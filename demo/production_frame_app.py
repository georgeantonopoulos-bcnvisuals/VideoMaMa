"""
Production sequence UI for multi-keyframe SAM 3 prompting + VideoMaMa inference.

This app is intended for Python 3.12+ environments because SAM 3 requires a
newer Python/PyTorch stack than the base VideoMaMa inference environment.
"""

import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for search_path in (THIS_DIR, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

try:
    import bz2  # noqa: F401
except ImportError:
    import types

    bz2 = types.ModuleType("bz2")

    class _MissingBZ2:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("bz2 support is unavailable in this local Python build")

    def _missing_bz2_open(*args, **kwargs):
        raise RuntimeError("bz2 support is unavailable in this local Python build")

    bz2.BZ2File = _MissingBZ2
    bz2.BZ2Compressor = _MissingBZ2
    bz2.BZ2Decompressor = _MissingBZ2
    bz2.open = _missing_bz2_open
    sys.modules["bz2"] = bz2

# Gradio resolves some cache/temp paths relative to the process working directory
# and calls os.getcwd() on them per request (e.g. get_cache_folder() ->
# abspath("gradio_cached_examples")). This repo lives on a network mount whose
# working directory can be swapped out mid-session; getcwd() then raises
# FileNotFoundError and Gradio's file routes 500. Pin these to absolute local
# paths before importing gradio so resolution never depends on the cwd.
import tempfile as _tempfile

_GRADIO_TMP = Path(os.environ.setdefault(
    "GRADIO_TEMP_DIR", str(Path(_tempfile.gettempdir()) / "videomama-gradio")
))
os.environ.setdefault("GRADIO_EXAMPLES_CACHE", str(_GRADIO_TMP / "examples"))
try:
    _GRADIO_TMP.mkdir(parents=True, exist_ok=True)
    Path(os.environ["GRADIO_EXAMPLES_CACHE"]).mkdir(parents=True, exist_ok=True)
except OSError:
    pass

import gradio as gr
import numpy as np
from PIL import Image

try:
    import OpenEXR
    import Imath
except ImportError as exc:
    raise RuntimeError("OpenEXR and Imath are required for production_frame_app.py") from exc

from sam3_wrapper_hf import load_sam3_tracker
from videomama_wrapper import load_videomama_pipeline, videomama
from tools.painter import mask_painter, point_painter

DEFAULT_SEQUENCE_DIR = "/mnt/production/project/ntf_fire/work/Editing/102_CameraSpin_4K"
DEFAULT_EXR_GAMMA = 1.0
WORK_WIDTH = 1024
WORK_HEIGHT = 576
MASK_COLOR = 3
MASK_ALPHA = 0.7
CONTOUR_COLOR = 1
CONTOUR_WIDTH = 5
POINT_COLOR_POS = 8
POINT_COLOR_NEG = 1
POINT_ALPHA = 0.9
POINT_RADIUS = 15
SUPPORTED_IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Anchor the app's working area to the repo (not the launch CWD) so settings and
# prior runs are found reliably and outputs land where AGENTS.md documents them.
APP_TMP_ROOT = REPO_ROOT / 'tmp' / 'production_sequence_app'
SETTINGS_PATH = APP_TMP_ROOT / 'ui_settings.json'

DEFAULT_SETTINGS = {
    'sequence_dir': DEFAULT_SEQUENCE_DIR,
    'exr_gamma': DEFAULT_EXR_GAMMA,
    'point_mode': 'Positive',
    'chunk_size': 16,
    'overlap': 4,
    'resume_from_tmp': True,
    'range_start': 0,
    'range_end': -1,
}

sam3_tracker = None
videomama_pipeline = None


def _load_ui_settings():
    """Load persisted UI settings, falling back to defaults for missing keys."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with SETTINGS_PATH.open('r', encoding='utf-8') as f:
            saved = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return settings
    if isinstance(saved, dict):
        for key in DEFAULT_SETTINGS:
            if saved.get(key) is not None:
                settings[key] = saved[key]
    return settings


def _save_ui_settings(sequence_dir, exr_gamma, point_mode, chunk_size, overlap, resume_from_tmp,
                      range_start, range_end):
    """Persist the current input panel so the next launch restores it."""
    APP_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        'sequence_dir': str(sequence_dir),
        'exr_gamma': float(exr_gamma),
        'point_mode': str(point_mode),
        'chunk_size': int(chunk_size),
        'overlap': int(overlap),
        'resume_from_tmp': bool(resume_from_tmp),
        'range_start': int(range_start),
        'range_end': int(range_end),
    }
    try:
        with SETTINGS_PATH.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        print(f"Warning: could not save UI settings: {exc}")


def _free_cuda_cache():
    """Return cached-but-unused CUDA blocks to the driver. Safe no-op on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _read_exr_rgb(image_path: str, exr_gamma: float = DEFAULT_EXR_GAMMA, exr_exposure: float = 0.0) -> np.ndarray:
    exr_file = OpenEXR.InputFile(image_path)
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = []
    for channel_name in ("R", "G", "B"):
        raw = exr_file.channel(channel_name, float_type)
        channels.append(np.frombuffer(raw, dtype=np.float32).reshape(height, width))

    image = np.stack(channels, axis=-1)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image * (2.0 ** exr_exposure), 0.0, None)
    image = np.clip(image, 0.0, 1.0) ** (1.0 / exr_gamma)
    return (image * 255.0).round().astype(np.uint8)


def _load_rgb_frame(image_path: str, exr_gamma: float) -> np.ndarray:
    suffix = Path(image_path).suffix.lower()
    if suffix == ".exr":
        return _read_exr_rgb(image_path, exr_gamma=exr_gamma)
    return np.array(Image.open(image_path).convert("RGB"))


def _resize_rgb_frame(frame: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize((WORK_WIDTH, WORK_HEIGHT), Image.Resampling.BILINEAR))


def _discover_sequence_files(sequence_dir: str):
    directory = Path(sequence_dir)
    if not directory.is_dir():
        raise gr.Error(f"Sequence directory does not exist: {sequence_dir}")
    frame_paths = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    )
    if not frame_paths:
        raise gr.Error(f"No supported image sequence files found in {sequence_dir}")
    return frame_paths


def _read_png(path: Path):
    if not path.exists():
        return None
    return np.array(Image.open(path))


def _safe_name(base_name: str):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in base_name) or 'sequence'


def _run_root(base_name: str):
    return APP_TMP_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{_safe_name(base_name)}"


def _write_session_meta(run_root: Path, sequence_dir: str, exr_gamma: float, frame_names):
    """Record which sequence a run belongs to so it can be matched on resume."""
    meta = {
        'sequence_dir': str(Path(sequence_dir).resolve()),
        'exr_gamma': float(exr_gamma),
        'frame_count': len(frame_names),
        'frame_names': list(frame_names),
    }
    try:
        with (Path(run_root) / 'session.json').open('w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    except OSError as exc:
        print(f"Warning: could not write session metadata: {exc}")


def _find_existing_run(sequence_dir: str, frame_names):
    """Find the most recent prior run for this sequence that has data to restore."""
    if not APP_TMP_ROOT.is_dir():
        return None
    resolved = str(Path(sequence_dir).resolve())
    safe = _safe_name(Path(sequence_dir).name)
    frame_count = len(frame_names)

    run_dirs = sorted((d for d in APP_TMP_ROOT.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for run_dir in run_dirs:
        meta_path = run_dir / 'session.json'
        matches = False
        if meta_path.exists():
            try:
                with meta_path.open('r', encoding='utf-8') as f:
                    meta = json.load(f)
                matches = str(meta.get('sequence_dir')) == resolved and int(meta.get('frame_count', -1)) == frame_count
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                matches = False
        elif run_dir.name.endswith('_' + safe):
            # Older runs predate session.json; fall back to name + cached frame count.
            cache = run_dir / 'sam3_frames'
            matches = cache.is_dir() and len(list(cache.glob('*.jpg'))) == frame_count

        if not matches:
            continue

        has_prompts = (run_dir / 'keyframe_prompts.json').exists()
        masks_dir = run_dir / 'sam3_masks'
        has_masks = masks_dir.is_dir() and any(masks_dir.glob('*.png'))
        if has_prompts or has_masks:
            return run_dir
    return None


def _load_prompts_from_run(run_root: Path):
    """Restore keyframe prompts saved by a prior generate_sam3_masks run."""
    prompts_path = Path(run_root) / 'keyframe_prompts.json'
    if not prompts_path.exists():
        return {}
    try:
        with prompts_path.open('r', encoding='utf-8') as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    restored = {}
    for key, value in (saved or {}).items():
        if not isinstance(value, dict):
            continue
        points = value.get('points') or []
        labels = value.get('labels') or []
        if len(points) != len(labels):
            continue
        restored[str(int(key))] = {
            'points': [[int(round(float(x))), int(round(float(y)))] for x, y in points],
            'labels': [int(label) for label in labels],
        }
    return restored


def initialize_models():
    global sam3_tracker, videomama_pipeline
    if sam3_tracker is not None and videomama_pipeline is not None:
        return "Models already loaded."

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_version = os.environ.get("SAM3_MODEL_VERSION", "sam3")
    sam3_tracker = load_sam3_tracker(device=device, model_version=sam3_version)
    videomama_pipeline = load_videomama_pipeline(device=device)
    return f"Loaded SAM 3 ({sam3_version}) and VideoMaMa on {device}."


def _keyframe_summary(state):
    if state is None:
        return "No sequence loaded."
    keyframes = sorted(int(idx) for idx, prompt_data in state.get("prompts_by_frame", {}).items() if prompt_data.get("points"))
    if not keyframes:
        return "No keyframes yet."
    labels = [f"{frame_idx}:{state['frame_names'][frame_idx]}" for frame_idx in keyframes[:12]]
    if len(keyframes) > 12:
        labels.append(f"... (+{len(keyframes) - 12} more)")
    return f"{len(keyframes)} keyframe(s): " + ', '.join(labels)


def _frame_info(state):
    if state is None:
        return "No sequence loaded."
    frame_idx = state['current_frame_idx']
    return f"Frame {frame_idx + 1} / {len(state['frame_paths'])}: {state['frame_names'][frame_idx]}"


def _prompt_data_for_frame(state, frame_idx):
    prompts = state['prompts_by_frame'].setdefault(str(frame_idx), {'points': [], 'labels': []})
    return prompts


def _invalidate_generated_results(state):
    state['generated_masks_dir'] = None
    state['videomama_output_dir'] = None
    state['alpha_output_dir'] = None


def _load_cached_frame(state, frame_idx):
    return np.array(Image.open(state['cache_frame_paths'][frame_idx]).convert('RGB'))


def _mask_output_path(state, frame_idx):
    if not state.get('generated_masks_dir'):
        return None
    frame_stem = Path(state['frame_names'][frame_idx]).stem
    return Path(state['generated_masks_dir']) / f"{frame_stem}.png"


def _videomama_output_path(state, frame_idx):
    if not state.get('videomama_output_dir'):
        return None
    frame_stem = Path(state['frame_names'][frame_idx]).stem
    return Path(state['videomama_output_dir']) / f"{frame_stem}.png"


def _load_current_mask(state, frame_idx):
    mask_path = _mask_output_path(state, frame_idx)
    if mask_path is not None and mask_path.exists():
        return np.array(Image.open(mask_path).convert('L'))

    preview_masks = state.get('preview_masks', {})
    cached_mask = preview_masks.get(str(frame_idx))
    if cached_mask is not None:
        return np.array(cached_mask, dtype=np.uint8)
    return None


def _compute_preview_mask(state, frame_idx):
    prompts = _prompt_data_for_frame(state, frame_idx)
    preview_masks = state.setdefault('preview_masks', {})
    if prompts['points'] and sam3_tracker is not None:
        frame = _load_cached_frame(state, frame_idx)
        mask = sam3_tracker.get_frame_mask(frame, prompts['points'], prompts['labels'])
        preview_masks[str(frame_idx)] = mask.astype(np.uint8).tolist()
        return mask

    preview_masks.pop(str(frame_idx), None)
    return None


def _load_current_output(state, frame_idx):
    output_path = _videomama_output_path(state, frame_idx)
    if output_path is not None and output_path.exists():
        return np.array(Image.open(output_path).convert('RGB'))
    return None


def _render_frame_preview(state, frame_idx):
    frame = _load_cached_frame(state, frame_idx)
    prompts = _prompt_data_for_frame(state, frame_idx)
    mask = _load_current_mask(state, frame_idx)
    preview = frame.copy()
    if mask is not None:
        preview = mask_painter(preview, mask, MASK_COLOR, MASK_ALPHA, CONTOUR_COLOR, CONTOUR_WIDTH)

    positive_points = np.array([prompts['points'][i] for i in range(len(prompts['points'])) if prompts['labels'][i] == 1], dtype=np.int32)
    negative_points = np.array([prompts['points'][i] for i in range(len(prompts['points'])) if prompts['labels'][i] == 0], dtype=np.int32)

    if len(positive_points) > 0:
        preview = point_painter(preview, positive_points, POINT_COLOR_POS, POINT_ALPHA, POINT_RADIUS, CONTOUR_COLOR, 2)
    if len(negative_points) > 0:
        preview = point_painter(preview, negative_points, POINT_COLOR_NEG, POINT_ALPHA, POINT_RADIUS, CONTOUR_COLOR, 2)

    output_preview = _load_current_output(state, frame_idx)
    return preview, mask, output_preview


def _ui_state_payload(state, status_message):
    preview, mask, output_preview = _render_frame_preview(state, state['current_frame_idx'])
    return (
        preview,
        mask,
        output_preview,
        state,
        gr.update(value=state['current_frame_idx']),
        _frame_info(state),
        _keyframe_summary(state),
        state.get('generated_masks_dir', ''),
        state.get('videomama_output_dir', ''),
        status_message,
    )


def load_sequence(sequence_dir: str, exr_gamma: float, resume_from_tmp: bool = True):
    exr_gamma = float(exr_gamma)
    frame_paths = _discover_sequence_files(sequence_dir)
    frame_names = [path.name for path in frame_paths]

    existing_run = _find_existing_run(sequence_dir, frame_names) if resume_from_tmp else None
    run_root = existing_run if existing_run is not None else _run_root(Path(sequence_dir).name)
    cache_dir = run_root / 'sam3_frames'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the cached working frames when an existing run already has a complete,
    # gamma-matching set; otherwise (re)build the 1024x576 JPEG cache.
    cached_jpgs = sorted(cache_dir.glob('*.jpg'))
    cached_gamma = None
    if existing_run is not None and (run_root / 'session.json').exists():
        try:
            with (run_root / 'session.json').open('r', encoding='utf-8') as f:
                cached_gamma = float(json.load(f).get('exr_gamma'))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            cached_gamma = None
    reuse_cache = (
        existing_run is not None
        and len(cached_jpgs) == len(frame_paths)
        and (cached_gamma is None or abs(cached_gamma - exr_gamma) < 1e-6)
    )

    if reuse_cache:
        cache_frame_paths = [str(path) for path in cached_jpgs]
        print(f"Reusing {len(cache_frame_paths)} cached frames from {cache_dir}")
    else:
        cache_frame_paths = []
        print(f"Preparing {len(frame_paths)} frames for SAM 3/UI cache...")
        for idx, frame_path in enumerate(frame_paths):
            frame = _load_rgb_frame(str(frame_path), exr_gamma=exr_gamma)
            working_frame = _resize_rgb_frame(frame)
            cache_path = cache_dir / f"{idx:05d}.jpg"
            Image.fromarray(working_frame).save(cache_path, quality=95)
            cache_frame_paths.append(str(cache_path))

    _write_session_meta(run_root, sequence_dir, exr_gamma, frame_names)

    state = {
        'sequence_dir': str(Path(sequence_dir).resolve()),
        'frame_paths': [str(path) for path in frame_paths],
        'frame_names': frame_names,
        'cache_dir': str(cache_dir),
        'cache_frame_paths': cache_frame_paths,
        'current_frame_idx': 0,
        'prompts_by_frame': {},
        'preview_masks': {},
        'generated_masks_dir': None,
        'videomama_output_dir': None,
        'alpha_output_dir': None,
        'run_root': str(run_root),
        'exr_gamma': exr_gamma,
    }

    # Restore prior points / masks / outputs from the matched run, if any.
    restored = []
    if existing_run is not None:
        prompts = _load_prompts_from_run(run_root)
        if prompts:
            state['prompts_by_frame'] = prompts
            restored.append(f"{len(prompts)} keyframe(s)")

        masks_dir = run_root / 'sam3_masks'
        first_stem = Path(frame_names[0]).stem
        if masks_dir.is_dir() and (masks_dir / f"{first_stem}.png").exists():
            state['generated_masks_dir'] = str(masks_dir)
            restored.append('SAM 3 masks')

        outputs_dir = run_root / 'videomama_frames'
        if outputs_dir.is_dir() and (outputs_dir / f"{first_stem}.png").exists():
            state['videomama_output_dir'] = str(outputs_dir)
            restored.append('VideoMaMa outputs')
        alpha_dir = run_root / 'alpha_frames'
        if alpha_dir.is_dir() and any(alpha_dir.glob('*.png')):
            state['alpha_output_dir'] = str(alpha_dir)

    if restored:
        status_message = (
            f"Loaded {len(frame_paths)} frames from {sequence_dir}. "
            f"Resumed from {run_root.name}: restored {', '.join(restored)}."
        )
    else:
        status_message = (
            f"Loaded {len(frame_paths)} frames from {sequence_dir}. "
            "Click any frame to add keyframe prompts."
        )

    preview, mask, output_preview = _render_frame_preview(state, 0)
    return (
        preview,
        mask,
        output_preview,
        state,
        gr.update(minimum=0, maximum=len(frame_paths) - 1, value=0, step=1, interactive=True),
        _frame_info(state),
        _keyframe_summary(state),
        state.get('generated_masks_dir') or '',
        state.get('videomama_output_dir') or '',
        status_message,
    )


def select_frame(state, frame_index):
    if state is None:
        raise gr.Error('Load a sequence first.')
    frame_index = int(frame_index)
    frame_index = max(0, min(frame_index, len(state['frame_paths']) - 1))
    state['current_frame_idx'] = frame_index
    return _ui_state_payload(state, f"Viewing {_frame_info(state)}")


def step_frame(state, step):
    if state is None:
        raise gr.Error('Load a sequence first.')
    return select_frame(state, state['current_frame_idx'] + int(step))


def prev_frame(state):
    return step_frame(state, -1)


def next_frame(state):
    return step_frame(state, 1)


def add_point(state, point_mode, evt: gr.SelectData):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if sam3_tracker is None:
        raise gr.Error('Initialize models first.')
    if evt is None or evt.index is None:
        raise gr.Error('Click data was not received by Gradio.')

    frame_idx = state['current_frame_idx']
    prompts = _prompt_data_for_frame(state, frame_idx)
    x, y = int(evt.index[0]), int(evt.index[1])
    prompts['points'].append([x, y])
    prompts['labels'].append(1 if point_mode == 'Positive' else 0)
    _compute_preview_mask(state, frame_idx)
    _invalidate_generated_results(state)
    return _ui_state_payload(state, f"Added {point_mode.lower()} point at ({x}, {y}) on frame {frame_idx}.")


def undo_point(state):
    if state is None:
        raise gr.Error('Load a sequence first.')
    prompts = _prompt_data_for_frame(state, state['current_frame_idx'])
    if not prompts['points']:
        return _ui_state_payload(state, 'No points to undo on this frame.')
    prompts['points'].pop()
    prompts['labels'].pop()
    _compute_preview_mask(state, state['current_frame_idx'])
    _invalidate_generated_results(state)
    return _ui_state_payload(state, 'Removed last point on current frame.')


def clear_frame_points(state):
    if state is None:
        raise gr.Error('Load a sequence first.')
    frame_idx = state['current_frame_idx']
    state['prompts_by_frame'][str(frame_idx)] = {'points': [], 'labels': []}
    state.setdefault('preview_masks', {}).pop(str(frame_idx), None)
    _invalidate_generated_results(state)
    return _ui_state_payload(state, 'Cleared prompts on current frame.')


def clear_all_points(state):
    if state is None:
        raise gr.Error('Load a sequence first.')
    state['prompts_by_frame'] = {}
    state['preview_masks'] = {}
    _invalidate_generated_results(state)
    return _ui_state_payload(state, 'Cleared prompts on all frames.')


def _normalized_prompt_dict(state):
    prompts = {}
    for raw_frame_idx, prompt_data in state['prompts_by_frame'].items():
        if prompt_data.get('points'):
            prompts[int(raw_frame_idx)] = {
                'points': prompt_data['points'],
                'labels': prompt_data['labels'],
            }
    if not prompts:
        raise gr.Error('Add prompts on at least one frame first.')
    return prompts


def generate_sam3_masks(state):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if sam3_tracker is None:
        raise gr.Error('Initialize models first.')

    prompts = _normalized_prompt_dict(state)
    run_root = Path(state['run_root'])
    mask_dir = run_root / 'sam3_masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    with (run_root / 'keyframe_prompts.json').open('w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in prompts.items()}, f, indent=2)

    masks = sam3_tracker.track_video_from_dir(state['cache_dir'], prompts)

    # SAM 3 propagation leaves a large block of cached VRAM behind. Release it so
    # the co-resident VideoMaMa matting pass has room on the same GPU.
    _free_cuda_cache()

    for frame_name, mask in zip(state['frame_names'], masks):
        Image.fromarray(mask).save(mask_dir / f"{Path(frame_name).stem}.png")

    state['generated_masks_dir'] = str(mask_dir)
    state['videomama_output_dir'] = None
    state['alpha_output_dir'] = None
    return _ui_state_payload(state, f"Generated {len(masks)} SAM 3 masks using {_keyframe_summary(state)}")


def _resolve_range(range_start, range_end, total_frames):
    """Clamp a user-supplied [start, end] (inclusive) range to valid frame indices.

    An end of -1 (or any value >= total) means "through the last frame", so the
    default leaves the whole sequence selected.
    """
    start = max(0, min(int(range_start), total_frames - 1))
    end = int(range_end)
    if end < 0 or end >= total_frames:
        end = total_frames - 1
    end = max(end, start)
    return start, end


def run_sequence(state, chunk_size, overlap, range_start=0, range_end=-1):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if videomama_pipeline is None:
        raise gr.Error('Initialize models first.')

    if not state.get('generated_masks_dir'):
        state = generate_sam3_masks(state)[3]

    total_frames = len(state['frame_paths'])
    range_start, range_end = _resolve_range(range_start, range_end, total_frames)
    is_subrange = not (range_start == 0 and range_end == total_frames - 1)

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    step = max(1, chunk_size - overlap)

    run_root = Path(state['run_root'])
    output_dir = run_root / 'videomama_frames'
    alpha_dir = run_root / 'alpha_frames'
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    print(
        f"Running VideoMaMa over frames [{range_start}, {range_end}] "
        f"({range_end - range_start + 1} of {total_frames}) with chunk_size={chunk_size}, overlap={overlap}"
    )
    # `end` is exclusive within the loop; the range is inclusive of range_end.
    for start in range(range_start, range_end + 1, step):
        end = min(range_end + 1, start + chunk_size)
        chunk_frame_paths = state['frame_paths'][start:end]
        chunk_frame_names = state['frame_names'][start:end]

        frames_np = [_load_rgb_frame(frame_path, exr_gamma=state['exr_gamma']) for frame_path in chunk_frame_paths]
        masks_np = []
        for frame_name in chunk_frame_names:
            mask_path = Path(state['generated_masks_dir']) / f"{Path(frame_name).stem}.png"
            if not mask_path.exists():
                raise gr.Error(f"Missing SAM 3 mask for {frame_name}: {mask_path}")
            masks_np.append(np.array(Image.open(mask_path).convert('L')))

        output_frames = videomama(videomama_pipeline, frames_np, masks_np)
        # Keep every frame of the first chunk in this run; for later chunks drop the
        # leading `overlap` frames, which the previous chunk already wrote (the
        # overlap re-processes shared frames so segment boundaries blend cleanly).
        keep_from = 0 if start == range_start else overlap
        for local_idx in range(keep_from, len(output_frames)):
            frame_name = chunk_frame_names[local_idx]
            output_frame = output_frames[local_idx]
            output_path = output_dir / f"{Path(frame_name).stem}.png"
            alpha_path = alpha_dir / f"{Path(frame_name).stem}.png"
            Image.fromarray(output_frame).save(output_path)
            Image.fromarray(output_frame).convert('L').save(alpha_path)
            processed += 1

        # Drop this chunk's cached activations before the next chunk so peak VRAM
        # does not creep up (and fragment) across a long sequence.
        _free_cuda_cache()

        if end == range_end + 1:
            break

    state['videomama_output_dir'] = str(output_dir)
    state['alpha_output_dir'] = str(alpha_dir)
    # Jump the viewer to the start of what was just processed.
    state['current_frame_idx'] = range_start
    scope = f"frames [{range_start}, {range_end}]" if is_subrange else "the whole sequence"
    return _ui_state_payload(
        state,
        f"Saved VideoMaMa outputs for {scope} ({processed} frame(s) written) to {output_dir}.",
    )


with gr.Blocks(title='VideoMaMa Production Sequence App') as demo:
    gr.Markdown('# VideoMaMa Production Sequence App')
    gr.Markdown('Load an EXR or image sequence, add SAM 3 prompts on multiple keyframes, generate the full mask track, then run VideoMaMa over the whole shot.')
    gr.Markdown('Workflow: Initialize models, load the sequence, navigate frames, add prompts on the frames that need correction, generate SAM 3 masks, then run the whole sequence.')

    state = gr.State(None)

    _settings = _load_ui_settings()

    with gr.Row():
        sequence_dir = gr.Textbox(label='Sequence Directory', value=_settings['sequence_dir'])
        exr_gamma = gr.Number(label='EXR Gamma', value=_settings['exr_gamma'], precision=2)
        point_mode = gr.Radio(['Positive', 'Negative'], value=_settings['point_mode'], label='Point Mode')

    with gr.Row():
        chunk_size = gr.Number(label='VideoMaMa Chunk Size', value=_settings['chunk_size'], precision=0)
        overlap = gr.Number(label='Chunk Overlap', value=_settings['overlap'], precision=0)
        resume_from_tmp = gr.Checkbox(
            label='Resume from tmp (reuse cached frames, points, masks)',
            value=_settings['resume_from_tmp'],
        )

    with gr.Row():
        range_start = gr.Number(label='Process Start Frame', value=_settings['range_start'], precision=0)
        range_end = gr.Number(label='Process End Frame (-1 = last)', value=_settings['range_end'], precision=0)

    with gr.Row():
        init_btn = gr.Button('Initialize Models')
        load_btn = gr.Button('Load Sequence')
        prev_btn = gr.Button('Prev Frame')
        next_btn = gr.Button('Next Frame')
        undo_btn = gr.Button('Undo Point')
        clear_frame_btn = gr.Button('Clear Frame Points')
        clear_all_btn = gr.Button('Clear All Prompts')
        gen_masks_btn = gr.Button('Generate SAM 3 Masks')
        run_btn = gr.Button('Run VideoMaMa (Selected Range)')

    frame_slider = gr.Slider(label='Current Frame', minimum=0, maximum=0, value=0, step=1, interactive=False)
    frame_info = gr.Textbox(label='Frame Info')
    keyframes_info = gr.Textbox(label='Keyframes')

    with gr.Row():
        preview = gr.Image(
            label='Sequence Preview / Prompt Overlay',
            type='numpy',
            interactive=False,
            sources=[],
            show_download_button=False,
            show_fullscreen_button=False,
        )
        mask_img = gr.Image(label='Current SAM 3 Mask', type='numpy')
        output_img = gr.Image(label='Current VideoMaMa Output', type='numpy')

    mask_dir = gr.Textbox(label='Saved Mask Directory')
    output_dir = gr.Textbox(label='Saved VideoMaMa Output Directory')
    status = gr.Textbox(label='Status')

    # Persist the input panel whenever any setting changes, so the next launch
    # restores the same session configuration.
    settings_inputs = [sequence_dir, exr_gamma, point_mode, chunk_size, overlap, resume_from_tmp,
                       range_start, range_end]
    for _component in settings_inputs:
        _component.change(_save_ui_settings, inputs=settings_inputs, outputs=None)

    init_btn.click(initialize_models, outputs=status)
    load_btn.click(
        load_sequence,
        inputs=[sequence_dir, exr_gamma, resume_from_tmp],
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    frame_slider.release(
        select_frame,
        inputs=[state, frame_slider],
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    prev_btn.click(
        prev_frame,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    next_btn.click(
        next_frame,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    preview.select(
        add_point,
        inputs=[state, point_mode],
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    undo_btn.click(
        undo_point,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    clear_frame_btn.click(
        clear_frame_points,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    clear_all_btn.click(
        clear_all_points,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    gen_masks_btn.click(
        generate_sam3_masks,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    run_btn.click(
        run_sequence,
        inputs=[state, chunk_size, overlap, range_start, range_end],
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )


if __name__ == '__main__':
    # Anchor the long-running server to a stable absolute directory. The repo's
    # network-mount cwd can vanish mid-session, after which Gradio's request
    # handlers crash on os.getcwd(); cd-ing to an absolute local dir avoids that.
    try:
        os.chdir(_GRADIO_TMP)
    except OSError:
        pass

    server_name = os.environ.get('VIDEOMAMA_UI_HOST', '127.0.0.1')
    server_port = int(os.environ.get('VIDEOMAMA_UI_PORT', '7861'))
    share = os.environ.get('VIDEOMAMA_UI_SHARE', '1') not in {'0', 'false', 'False'}
    demo.launch(server_name=server_name, server_port=server_port, share=share)
