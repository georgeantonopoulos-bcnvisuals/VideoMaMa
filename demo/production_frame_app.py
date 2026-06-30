"""
Production sequence UI for multi-keyframe SAM 3 prompting + VideoMaMa inference.

This app is intended for Python 3.12+ environments because SAM 3 requires a
newer Python/PyTorch stack than the base VideoMaMa inference environment.
"""

import json
import os
import shutil
import sys
import time
import gc
import threading
from collections import deque
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
from PIL import Image, ImageFilter

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
DEFAULT_PROCESSING_RESOLUTION = f"{WORK_WIDTH}x{WORK_HEIGHT}"
PROCESSING_RESOLUTIONS = {
    "1024x576": (1024, 576),
    "1280x720 (experimental)": (1280, 720),
    "1536x864 (experimental)": (1536, 864),
    "2048x1152 (experimental)": (2048, 1152),
}
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
    'prompt_mode': 'Point keyframes',
    'concept_prompt': '',
    'point_mode': 'Positive',
    'quality_preset': 'Balanced',
    'sam_output_prob_thresh': 0.5,
    'videomama_mask_cond_mode': 'vae',
    'videomama_seed': 42,
    'videomama_fps': 7,
    'videomama_motion_bucket_id': 127,
    'videomama_noise_aug_strength': 0.0,
    'processing_resolution': DEFAULT_PROCESSING_RESOLUTION,
    'refine_edges_against_plate': False,
    'free_gpu_after_run': True,
    'chunk_size': 16,
    'overlap': 4,
    'custom_output_dir': '',
    'resume_from_tmp': True,
    'range_start': 0,
    'range_end': -1,
}

QUALITY_PRESETS = {
    'Balanced': {
        'sam_output_prob_thresh': 0.5,
        'videomama_mask_cond_mode': 'vae',
        'videomama_fps': 7,
        'videomama_motion_bucket_id': 127,
        'videomama_noise_aug_strength': 0.0,
    },
    'Fine Detail': {
        'sam_output_prob_thresh': 0.35,
        'videomama_mask_cond_mode': 'vae',
        'videomama_fps': 7,
        'videomama_motion_bucket_id': 127,
        'videomama_noise_aug_strength': 0.0,
    },
    'Tight Matte': {
        'sam_output_prob_thresh': 0.65,
        'videomama_mask_cond_mode': 'vae',
        'videomama_fps': 7,
        'videomama_motion_bucket_id': 127,
        'videomama_noise_aug_strength': 0.0,
    },
    'Custom': {},
}

APP_CSS = """
.accurate-preview img,
.accurate-preview canvas,
.accurate-preview video {
    object-fit: contain !important;
}
#large_prompt_preview img,
#large_prompt_preview canvas {
    object-fit: contain !important;
    max-height: 78vh !important;
}
.danger-button {
    background: #8f2f2f !important;
    border-color: #7a2929 !important;
    color: #fff7f7 !important;
}
.danger-button:hover {
    background: #a43a3a !important;
    border-color: #8f2f2f !important;
}
.debug-console textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
    font-size: 12px !important;
    line-height: 1.35 !important;
}
"""

sam3_tracker = None
videomama_pipeline = None
_DEBUG_LINES = deque(maxlen=500)
_DEBUG_LOCK = threading.Lock()
UI_OUTPUT_STATUS_INDEX = 11
UI_OUTPUT_DEBUG_INDEX = 12
UI_OUTPUT_COUNT = 13
CLEARED_RUN_PREFIX = "cleared"


def _append_debug_line(line):
    line = str(line).replace('\r', '\n')
    for part in line.splitlines():
        part = part.strip()
        if not part:
            continue
        timestamp = time.strftime('%H:%M:%S')
        with _DEBUG_LOCK:
            _DEBUG_LINES.append(f"[{timestamp}] {part}")


def _debug_console_text():
    with _DEBUG_LOCK:
        return '\n'.join(_DEBUG_LINES)


class _DebugTee:
    def __init__(self, wrapped, label):
        self._wrapped = wrapped
        self._label = label
        self._buffer = ''

    def write(self, data):
        self._wrapped.write(data)
        self._wrapped.flush()
        if not data:
            return
        self._buffer += str(data)
        while '\n' in self._buffer or '\r' in self._buffer:
            newline_positions = [pos for pos in (self._buffer.find('\n'), self._buffer.find('\r')) if pos >= 0]
            split_at = min(newline_positions)
            chunk = self._buffer[:split_at]
            self._buffer = self._buffer[split_at + 1:]
            _append_debug_line(f"{self._label}: {chunk}")

    def flush(self):
        self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


if not isinstance(sys.stdout, _DebugTee):
    sys.stdout = _DebugTee(sys.stdout, 'OUT')
if not isinstance(sys.stderr, _DebugTee):
    sys.stderr = _DebugTee(sys.stderr, 'ERR')


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


def _save_ui_settings(sequence_dir, exr_gamma, prompt_mode, concept_prompt, point_mode,
                      quality_preset, sam_output_prob_thresh, videomama_mask_cond_mode,
                      videomama_seed, videomama_fps, videomama_motion_bucket_id,
                      videomama_noise_aug_strength, processing_resolution,
                      refine_edges_against_plate, free_gpu_after_run, chunk_size, overlap,
                      custom_output_dir, resume_from_tmp, range_start, range_end):
    """Persist the current input panel so the next launch restores it."""
    APP_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        'sequence_dir': str(sequence_dir),
        'exr_gamma': float(exr_gamma),
        'prompt_mode': str(prompt_mode),
        'concept_prompt': str(concept_prompt or ''),
        'point_mode': str(point_mode),
        'quality_preset': str(quality_preset or 'Balanced'),
        'sam_output_prob_thresh': float(sam_output_prob_thresh),
        'videomama_mask_cond_mode': str(videomama_mask_cond_mode or 'vae'),
        'videomama_seed': int(videomama_seed),
        'videomama_fps': int(videomama_fps),
        'videomama_motion_bucket_id': int(videomama_motion_bucket_id),
        'videomama_noise_aug_strength': float(videomama_noise_aug_strength),
        'processing_resolution': str(processing_resolution or DEFAULT_PROCESSING_RESOLUTION),
        'refine_edges_against_plate': bool(refine_edges_against_plate),
        'free_gpu_after_run': bool(free_gpu_after_run),
        'chunk_size': int(chunk_size),
        'overlap': int(overlap),
        'custom_output_dir': str(custom_output_dir or ''),
        'resume_from_tmp': bool(resume_from_tmp),
        'range_start': int(range_start),
        'range_end': int(range_end),
    }
    try:
        with SETTINGS_PATH.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        print(f"Warning: could not save UI settings: {exc}")


def _sam_output_prob_thresh(value):
    return max(0.01, min(float(value), 0.99))


def _processing_work_size(processing_resolution):
    label = str(processing_resolution or DEFAULT_PROCESSING_RESOLUTION)
    if label not in PROCESSING_RESOLUTIONS:
        label = DEFAULT_PROCESSING_RESOLUTION
    return PROCESSING_RESOLUTIONS[label]


def _work_size_from_state(state):
    if state and state.get('work_size'):
        return int(state['work_size'][0]), int(state['work_size'][1])
    return WORK_WIDTH, WORK_HEIGHT


def apply_quality_preset(quality_preset):
    preset = QUALITY_PRESETS.get(str(quality_preset or 'Balanced'), QUALITY_PRESETS['Balanced'])
    if not preset:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return (
        gr.update(value=preset['sam_output_prob_thresh']),
        gr.update(value=preset['videomama_mask_cond_mode']),
        gr.update(value=preset['videomama_fps']),
        gr.update(value=preset['videomama_motion_bucket_id']),
        gr.update(value=preset['videomama_noise_aug_strength']),
    )


def _free_cuda_cache():
    """Return cached-but-unused CUDA blocks to the driver. Safe no-op on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
    except Exception:
        pass


def _unload_videomama_pipeline():
    """Drop VideoMaMa weights when SAM 3 needs the GPU for propagation."""
    global videomama_pipeline

    if videomama_pipeline is None:
        return
    videomama_pipeline = None
    gc.collect()
    _free_cuda_cache()


def _unload_sam3_tracker():
    """Drop SAM 3 weights when VideoMaMa needs the GPU for matting."""
    global sam3_tracker

    if sam3_tracker is None:
        return
    try:
        predictor = getattr(sam3_tracker, 'predictor', None)
        if predictor is not None and hasattr(predictor, 'shutdown'):
            predictor.shutdown()
    except Exception as exc:
        print(f"Warning: failed to shutdown SAM 3 predictor cleanly: {exc}")
    sam3_tracker = None
    gc.collect()
    _free_cuda_cache()


def unload_models_for_gpu():
    """Release loaded model weights without clearing UI/session data on disk."""
    had_sam3 = sam3_tracker is not None
    had_videomama = videomama_pipeline is not None
    _unload_sam3_tracker()
    _unload_videomama_pipeline()
    status_message = "Unloaded SAM 3 and VideoMaMa models from GPU."
    if not had_sam3 and not had_videomama:
        status_message = "No SAM 3 or VideoMaMa model was loaded."
    _append_debug_line(status_message)
    return status_message, _debug_console_text()


def _ensure_sam3_tracker(device):
    global sam3_tracker

    if sam3_tracker is None:
        sam3_version = os.environ.get("SAM3_MODEL_VERSION", "sam3")
        sam3_tracker = load_sam3_tracker(device=device, model_version=sam3_version)
    return sam3_tracker


def _ensure_videomama_pipeline(device):
    global videomama_pipeline

    if videomama_pipeline is None:
        videomama_pipeline = load_videomama_pipeline(device=device)
    return videomama_pipeline


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


def _image_size(image_path: str):
    suffix = Path(image_path).suffix.lower()
    if suffix == ".exr":
        exr_file = OpenEXR.InputFile(str(image_path))
        data_window = exr_file.header()["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        return int(width), int(height)
    with Image.open(image_path) as image:
        return image.size


def _resize_mask(mask: np.ndarray, size, resample=Image.Resampling.NEAREST) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    if mask.shape[:2] == (height, width):
        return mask.astype(np.uint8)
    return np.array(Image.fromarray(mask.astype(np.uint8)).resize((width, height), resample))


def _box_mean_2d(image: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    padded = np.pad(image.astype(np.float32), ((radius, radius), (radius, radius)), mode='edge')
    integral = np.pad(padded, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    window = radius * 2 + 1
    sums = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return sums / float(window * window)


def _guided_filter_gray(guide: np.ndarray, alpha: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    guide = guide.astype(np.float32)
    alpha = alpha.astype(np.float32)
    mean_guide = _box_mean_2d(guide, radius)
    mean_alpha = _box_mean_2d(alpha, radius)
    corr_guide = _box_mean_2d(guide * guide, radius)
    corr_guide_alpha = _box_mean_2d(guide * alpha, radius)
    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_alpha = corr_guide_alpha - mean_guide * mean_alpha
    a = cov_guide_alpha / (var_guide + float(eps))
    b = mean_alpha - a * mean_guide
    mean_a = _box_mean_2d(a, radius)
    mean_b = _box_mean_2d(b, radius)
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def _refine_alpha_against_plate(plate_rgb: np.ndarray, alpha_rgb: np.ndarray, sam_mask: np.ndarray) -> np.ndarray:
    """Snap a VideoMaMa matte toward full-resolution plate edges, bounded by SAM."""
    if alpha_rgb.ndim == 3:
        alpha = np.array(Image.fromarray(alpha_rgb).convert('L'), dtype=np.float32) / 255.0
    else:
        alpha = alpha_rgb.astype(np.float32) / 255.0
    guide = np.array(Image.fromarray(plate_rgb).convert('L'), dtype=np.float32) / 255.0
    sam = np.array(Image.fromarray(sam_mask.astype(np.uint8)).convert('L'), dtype=np.uint8)

    # Work inside a slightly expanded SAM region so thin edges can move, while
    # suppressing unrelated plate edges far from the prompted object.
    dilated_sam = Image.fromarray(sam).filter(ImageFilter.MaxFilter(9))
    eroded_sam = Image.fromarray(sam).filter(ImageFilter.MinFilter(5))
    allowed = np.array(dilated_sam.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32) / 255.0
    confident_inside = np.array(eroded_sam, dtype=np.float32) / 255.0

    refined = _guided_filter_gray(guide, alpha, radius=8, eps=0.01)
    refined *= np.clip(allowed * 1.15, 0.0, 1.0)
    refined = np.where(confident_inside > 0.95, np.maximum(refined, alpha), refined)
    return (np.clip(refined, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _letterbox_transform(source_width: int, source_height: int, work_size=None):
    work_width, work_height = work_size or (WORK_WIDTH, WORK_HEIGHT)
    scale = min(work_width / source_width, work_height / source_height)
    content_width = max(1, int(round(source_width * scale)))
    content_height = max(1, int(round(source_height * scale)))
    pad_x = (work_width - content_width) // 2
    pad_y = (work_height - content_height) // 2
    return {
        'source_width': int(source_width),
        'source_height': int(source_height),
        'work_width': int(work_width),
        'work_height': int(work_height),
        'content_width': int(content_width),
        'content_height': int(content_height),
        'pad_x': int(pad_x),
        'pad_y': int(pad_y),
        'scale': float(scale),
    }


def _letterbox_rgb_frame(frame: np.ndarray, work_size=None):
    """Fit a source frame into the fixed SAM/UI work canvas without stretching."""
    work_width, work_height = work_size or (WORK_WIDTH, WORK_HEIGHT)
    source_height, source_width = frame.shape[:2]
    transform = _letterbox_transform(source_width, source_height, work_size=(work_width, work_height))
    content_width = transform['content_width']
    content_height = transform['content_height']
    pad_x = transform['pad_x']
    pad_y = transform['pad_y']
    resized = Image.fromarray(frame).resize((content_width, content_height), Image.Resampling.BILINEAR)
    canvas = np.zeros((work_height, work_width, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + content_height, pad_x:pad_x + content_width] = np.array(resized)
    return canvas, transform


def _work_mask_to_source(mask: np.ndarray, transform) -> np.ndarray:
    """Remove UI/SAM letterbox padding and resize a work mask to source size."""
    if not transform:
        return mask.astype(np.uint8)
    pad_x = int(transform.get('pad_x', 0))
    pad_y = int(transform.get('pad_y', 0))
    content_width = int(transform.get('content_width', mask.shape[1]))
    content_height = int(transform.get('content_height', mask.shape[0]))
    source_size = (int(transform['source_width']), int(transform['source_height']))
    cropped = mask[pad_y:pad_y + content_height, pad_x:pad_x + content_width]
    return _resize_mask(cropped, source_size)


def _source_mask_to_work(mask: np.ndarray, transform) -> np.ndarray:
    """Letterbox a source-resolution mask back to the fixed UI/SAM canvas."""
    if not transform:
        return _resize_mask(mask, (WORK_WIDTH, WORK_HEIGHT))
    work_width = int(transform.get('work_width', WORK_WIDTH))
    work_height = int(transform.get('work_height', WORK_HEIGHT))
    pad_x = int(transform.get('pad_x', 0))
    pad_y = int(transform.get('pad_y', 0))
    content_width = int(transform.get('content_width', work_width))
    content_height = int(transform.get('content_height', work_height))
    resized = _resize_mask(mask, (content_width, content_height))
    canvas = np.zeros((work_height, work_width), dtype=np.uint8)
    canvas[pad_y:pad_y + content_height, pad_x:pad_x + content_width] = resized
    return canvas


def _point_in_content(state, frame_idx, x, y):
    transform = (state.get('frame_transforms') or [{}])[frame_idx]
    work_width, work_height = _work_size_from_state(state)
    pad_x = int(transform.get('pad_x', 0))
    pad_y = int(transform.get('pad_y', 0))
    content_width = int(transform.get('content_width', work_width))
    content_height = int(transform.get('content_height', work_height))
    return pad_x <= x < pad_x + content_width and pad_y <= y < pad_y + content_height


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


def _write_session_meta(run_root: Path, sequence_dir: str, exr_gamma: float, frame_names,
                        frame_sizes, frame_transforms=None, work_size=None):
    """Record which sequence a run belongs to so it can be matched on resume."""
    work_width, work_height = work_size or (WORK_WIDTH, WORK_HEIGHT)
    meta = {
        'sequence_dir': str(Path(sequence_dir).resolve()),
        'exr_gamma': float(exr_gamma),
        'frame_count': len(frame_names),
        'frame_names': list(frame_names),
        'frame_sizes': [[int(width), int(height)] for width, height in frame_sizes],
        'frame_transforms': frame_transforms or [],
        'work_size': [int(work_width), int(work_height)],
    }
    try:
        with (Path(run_root) / 'session.json').open('w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    except OSError as exc:
        print(f"Warning: could not write session metadata: {exc}")


def _find_existing_run(sequence_dir: str, frame_names, work_size=None):
    """Find the most recent prior run for this sequence that has data to restore."""
    if not APP_TMP_ROOT.is_dir():
        return None
    resolved = str(Path(sequence_dir).resolve())
    safe = _safe_name(Path(sequence_dir).name)
    frame_count = len(frame_names)
    work_width, work_height = work_size or (WORK_WIDTH, WORK_HEIGHT)

    run_dirs = sorted((d for d in APP_TMP_ROOT.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for run_dir in run_dirs:
        meta_path = run_dir / 'session.json'
        matches = False
        if meta_path.exists():
            try:
                with meta_path.open('r', encoding='utf-8') as f:
                    meta = json.load(f)
                matches = (
                    str(meta.get('sequence_dir')) == resolved
                    and int(meta.get('frame_count', -1)) == frame_count
                    and (not meta.get('frame_names') or list(meta.get('frame_names')) == list(frame_names))
                    and (not meta.get('work_size') or list(meta.get('work_size')) == [work_width, work_height])
                )
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                matches = False
        elif run_dir.name.endswith('_' + safe) and (work_width, work_height) == (WORK_WIDTH, WORK_HEIGHT):
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


def _load_concept_prompt_from_run(run_root: Path):
    prompt_path = Path(run_root) / 'concept_prompt.json'
    if not prompt_path.exists():
        return ''
    try:
        with prompt_path.open('r', encoding='utf-8') as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ''
    return str(saved.get('text') or '').strip() if isinstance(saved, dict) else ''


def initialize_models():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sam3_tracker is not None:
        status_message = "SAM 3 already loaded. VideoMaMa will load when matting starts."
        _append_debug_line(status_message)
        return status_message, _debug_console_text()

    sam3_version = os.environ.get("SAM3_MODEL_VERSION", "sam3")
    _ensure_sam3_tracker(device)
    status_message = f"Loaded SAM 3 ({sam3_version}) on {device}. VideoMaMa will load when matting starts."
    _append_debug_line(status_message)
    return status_message, _debug_console_text()


def refresh_debug_console():
    return _debug_console_text()


def clear_debug_console():
    with _DEBUG_LOCK:
        _DEBUG_LINES.clear()
    return ''


def _clear_run_dir(run_root: Path):
    """Move an active run aside, then best-effort delete it.

    Some production mounts can report `Directory not empty` while a directory tree
    is being removed. Renaming first makes the clear-cache operation effective for
    the UI even if physical cleanup has to be retried later.
    """
    run_root = Path(run_root)
    if not run_root.exists() or run_root.parent != APP_TMP_ROOT or run_root.name == APP_TMP_ROOT.name:
        return None, None

    trash_root = APP_TMP_ROOT / f"{CLEARED_RUN_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}_{run_root.name}"
    suffix = 1
    while trash_root.exists():
        trash_root = APP_TMP_ROOT / f"{CLEARED_RUN_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}_{run_root.name}_{suffix}"
        suffix += 1

    try:
        run_root.rename(trash_root)
    except OSError as exc:
        cleanup_warning = _empty_run_dir(run_root)
        if cleanup_warning:
            return run_root.name, f"Could not move stale cache aside ({exc}); {cleanup_warning}"
        return run_root.name, f"Could not move stale cache aside ({exc}); cleared contents in place"

    cleanup_error = None
    for attempt in range(3):
        try:
            shutil.rmtree(trash_root)
            return run_root.name, None
        except OSError as exc:
            cleanup_error = exc
            time.sleep(0.1 * (attempt + 1))

    return run_root.name, f"Moved stale cache to {trash_root}; cleanup failed: {cleanup_error}"


def _empty_run_dir(run_root: Path):
    cleanup_error = None
    for attempt in range(3):
        try:
            for current_root, dirs, files in os.walk(run_root, topdown=False):
                current_root = Path(current_root)
                for filename in files:
                    try:
                        (current_root / filename).unlink()
                    except OSError as exc:
                        cleanup_error = exc
                for dirname in dirs:
                    try:
                        (current_root / dirname).rmdir()
                    except OSError as exc:
                        cleanup_error = exc
            try:
                run_root.rmdir()
                return None
            except OSError as exc:
                cleanup_error = exc
            time.sleep(0.1 * (attempt + 1))
        except OSError as exc:
            cleanup_error = exc
            time.sleep(0.1 * (attempt + 1))

    return f"cleared cache contents in place but could not remove root {run_root}: {cleanup_error}"


def _keyframe_summary(state):
    if state is None:
        return "No sequence loaded."
    if state.get('prompt_mode') == 'Text concept':
        text = str(state.get('concept_prompt') or '').strip()
        return f"Text concept: {text}" if text else "No concept prompt yet."
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
    state['generated_masks_sam_output_prob_thresh'] = None
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


def _resolve_videomama_output_dirs(run_root: Path, custom_output_dir: str):
    custom_output_dir = str(custom_output_dir or '').strip()
    if not custom_output_dir:
        output_dir = run_root / 'videomama_frames'
        return output_dir, run_root / 'alpha_frames'

    output_dir = Path(custom_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir = output_dir.resolve()

    if output_dir.name == 'videomama_frames':
        alpha_dir = output_dir.parent / 'alpha_frames'
    elif output_dir.name == 'alpha_frames':
        raise gr.Error('Custom VideoMaMa output directory cannot be named alpha_frames.')
    else:
        alpha_dir = output_dir / 'alpha_frames'
    return output_dir, alpha_dir


def _load_current_mask(state, frame_idx):
    mask_path = _mask_output_path(state, frame_idx)
    if mask_path is not None and mask_path.exists():
        mask = np.array(Image.open(mask_path).convert('L'))
        transform = (state.get('frame_transforms') or [{}])[frame_idx]
        return _source_mask_to_work(mask, transform)

    preview_masks = state.get('preview_masks', {})
    cached_mask = preview_masks.get(str(frame_idx))
    if cached_mask is not None:
        return np.array(cached_mask, dtype=np.uint8)
    return None


def _compute_preview_mask(state, frame_idx, sam_output_prob_thresh=None):
    sam_output_prob_thresh = _sam_output_prob_thresh(
        state.get('sam_output_prob_thresh', 0.5) if sam_output_prob_thresh is None else sam_output_prob_thresh
    )
    state['sam_output_prob_thresh'] = sam_output_prob_thresh
    if state.get('prompt_mode') == 'Text concept':
        text = str(state.get('concept_prompt') or '').strip()
        preview_masks = state.setdefault('preview_masks', {})
        if text and sam3_tracker is not None:
            frame = _load_cached_frame(state, frame_idx)
            mask = sam3_tracker.get_text_frame_mask(frame, text, output_prob_thresh=sam_output_prob_thresh)
            if not np.any(mask):
                stats = getattr(sam3_tracker, 'last_text_prompt_stats', {}) or {}
                raise gr.Error(
                    f"SAM 3 found no mask pixels for `{text}` on this frame. "
                    f"Mask count: {stats.get('mask_count', 0)}. "
                    "Try a simpler noun phrase such as person, girl, hair, arm, or elbow."
                )
            preview_masks[str(frame_idx)] = mask.astype(np.uint8).tolist()
            return mask
        preview_masks.pop(str(frame_idx), None)
        return None

    prompts = _prompt_data_for_frame(state, frame_idx)
    preview_masks = state.setdefault('preview_masks', {})
    if prompts['points'] and sam3_tracker is not None:
        frame = _load_cached_frame(state, frame_idx)
        mask = sam3_tracker.get_frame_mask(
            frame,
            prompts['points'],
            prompts['labels'],
            output_prob_thresh=sam_output_prob_thresh,
        )
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
    _append_debug_line(status_message)
    preview, mask, output_preview = _render_frame_preview(state, state['current_frame_idx'])
    return (
        preview,
        preview,
        mask,
        output_preview,
        state,
        gr.update(value=state['current_frame_idx']),
        gr.update(value=state['current_frame_idx']),
        _frame_info(state),
        _keyframe_summary(state),
        state.get('generated_masks_dir', ''),
        state.get('videomama_output_dir', ''),
        status_message,
        _debug_console_text(),
    )


def load_sequence(sequence_dir: str, exr_gamma: float, resume_from_tmp: bool = True,
                  prompt_mode: str = 'Point keyframes', concept_prompt: str = '',
                  processing_resolution: str = DEFAULT_PROCESSING_RESOLUTION):
    exr_gamma = float(exr_gamma)
    prompt_mode = str(prompt_mode or 'Point keyframes')
    concept_prompt = str(concept_prompt or '').strip()
    work_size = _processing_work_size(processing_resolution)
    work_width, work_height = work_size
    frame_paths = _discover_sequence_files(sequence_dir)
    frame_names = [path.name for path in frame_paths]
    frame_sizes = [_image_size(str(path)) for path in frame_paths]
    frame_transforms = [_letterbox_transform(width, height, work_size=work_size) for width, height in frame_sizes]

    existing_run = _find_existing_run(sequence_dir, frame_names, work_size=work_size) if resume_from_tmp else None
    run_root = existing_run if existing_run is not None else _run_root(Path(sequence_dir).name)
    cache_dir = run_root / 'sam3_frames'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the cached working frames when an existing run already has a complete,
    # gamma-matching set; otherwise (re)build the selected JPEG cache.
    cached_jpgs = sorted(cache_dir.glob('*.jpg'))
    cached_gamma = None
    cached_has_letterbox_meta = False
    if existing_run is not None and (run_root / 'session.json').exists():
        try:
            with (run_root / 'session.json').open('r', encoding='utf-8') as f:
                cached_meta = json.load(f)
            cached_gamma = float(cached_meta.get('exr_gamma'))
            cached_transforms = cached_meta.get('frame_transforms') or []
            cached_has_letterbox_meta = len(cached_transforms) == len(frame_paths)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            cached_gamma = None
    reuse_cache = (
        existing_run is not None
        and cached_has_letterbox_meta
        and len(cached_jpgs) == len(frame_paths)
        and (cached_gamma is None or abs(cached_gamma - exr_gamma) < 1e-6)
    )

    if reuse_cache:
        cache_frame_paths = [str(path) for path in cached_jpgs]
        print(f"Reusing {len(cache_frame_paths)} cached frames from {cache_dir}")
    else:
        cache_frame_paths = []
        print(f"Preparing {len(frame_paths)} frames for SAM 3/UI cache at {work_width}x{work_height}...")
        for idx, frame_path in enumerate(frame_paths):
            frame = _load_rgb_frame(str(frame_path), exr_gamma=exr_gamma)
            working_frame, frame_transforms[idx] = _letterbox_rgb_frame(frame, work_size=work_size)
            cache_path = cache_dir / f"{idx:05d}.jpg"
            Image.fromarray(working_frame).save(cache_path, quality=95)
            cache_frame_paths.append(str(cache_path))

    _write_session_meta(run_root, sequence_dir, exr_gamma, frame_names, frame_sizes, frame_transforms, work_size=work_size)

    state = {
        'sequence_dir': str(Path(sequence_dir).resolve()),
        'frame_paths': [str(path) for path in frame_paths],
        'frame_names': frame_names,
        'frame_sizes': [[int(width), int(height)] for width, height in frame_sizes],
        'frame_transforms': frame_transforms,
        'cache_dir': str(cache_dir),
        'cache_frame_paths': cache_frame_paths,
        'current_frame_idx': 0,
        'prompts_by_frame': {},
        'preview_masks': {},
        'prompt_mode': prompt_mode,
        'concept_prompt': concept_prompt,
        'generated_masks_dir': None,
        'generated_masks_sam_output_prob_thresh': None,
        'videomama_output_dir': None,
        'alpha_output_dir': None,
        'run_root': str(run_root),
        'exr_gamma': exr_gamma,
        'work_size': [int(work_width), int(work_height)],
        'processing_resolution': str(processing_resolution or DEFAULT_PROCESSING_RESOLUTION),
        'sam_output_prob_thresh': _sam_output_prob_thresh(_settings.get('sam_output_prob_thresh', 0.5)),
    }

    # Restore prior points / masks / outputs from the matched run, if any.
    restored = []
    if existing_run is not None:
        prompts = _load_prompts_from_run(run_root)
        if prompts:
            state['prompts_by_frame'] = prompts
            restored.append(f"{len(prompts)} keyframe(s)")
        restored_concept = _load_concept_prompt_from_run(run_root)
        if restored_concept and not concept_prompt:
            state['concept_prompt'] = restored_concept
            restored.append('concept prompt')

        masks_dir = run_root / 'sam3_masks'
        first_stem = Path(frame_names[0]).stem
        first_mask_path = masks_dir / f"{first_stem}.png"
        if masks_dir.is_dir() and first_mask_path.exists():
            with Image.open(first_mask_path) as first_mask:
                mask_size = first_mask.size
            if mask_size == tuple(frame_sizes[0]):
                state['generated_masks_dir'] = str(masks_dir)
                state['generated_masks_sam_output_prob_thresh'] = 0.5
                restored.append('SAM 3 masks')
            else:
                print(f"Ignoring stale SAM 3 masks with size {mask_size}; source is {frame_sizes[0]}")

        outputs_dir = run_root / 'videomama_frames'
        first_output_path = outputs_dir / f"{first_stem}.png"
        if outputs_dir.is_dir() and first_output_path.exists():
            with Image.open(first_output_path) as first_output:
                output_size = first_output.size
            if output_size == tuple(frame_sizes[0]):
                state['videomama_output_dir'] = str(outputs_dir)
                restored.append('VideoMaMa outputs')
            else:
                print(f"Ignoring stale VideoMaMa outputs with size {output_size}; source is {frame_sizes[0]}")
        alpha_dir = run_root / 'alpha_frames'
        if alpha_dir.is_dir() and any(alpha_dir.glob('*.png')):
            state['alpha_output_dir'] = str(alpha_dir)

    if restored:
        status_message = (
            f"Loaded {len(frame_paths)} frames from {sequence_dir}. "
            f"Resumed from {run_root.name} at {work_width}x{work_height}: restored {', '.join(restored)}."
        )
    else:
        status_message = (
            f"Loaded {len(frame_paths)} frames from {sequence_dir}. "
            f"Working at {work_width}x{work_height}. Add point keyframes or enter a text concept, then generate SAM 3 masks."
        )

    preview, mask, output_preview = _render_frame_preview(state, 0)
    _append_debug_line(status_message)
    return (
        preview,
        preview,
        mask,
        output_preview,
        state,
        gr.update(minimum=0, maximum=len(frame_paths) - 1, value=0, step=1, interactive=True),
        gr.update(minimum=0, maximum=len(frame_paths) - 1, value=0, step=1, interactive=True),
        _frame_info(state),
        _keyframe_summary(state),
        state.get('generated_masks_dir') or '',
        state.get('videomama_output_dir') or '',
        status_message,
        _debug_console_text(),
    )


def clear_sequence_cache_and_reload(state, sequence_dir: str, exr_gamma: float, prompt_mode: str,
                                    concept_prompt: str, processing_resolution: str):
    """Delete this sequence's current app cache/run directory, then load fresh frames."""
    removed = []
    cleanup_warnings = []
    candidate_roots = []
    if state and state.get('run_root'):
        candidate_roots.append(Path(state['run_root']))
    else:
        try:
            frame_paths = _discover_sequence_files(sequence_dir)
            frame_names = [path.name for path in frame_paths]
            existing_run = _find_existing_run(
                sequence_dir,
                frame_names,
                work_size=_processing_work_size(processing_resolution),
            )
            if existing_run is not None:
                candidate_roots.append(existing_run)
        except Exception:
            candidate_roots = []

    for run_root in dict.fromkeys(candidate_roots):
        run_root = Path(run_root)
        try:
            # Safety: only remove per-run directories anchored under APP_TMP_ROOT.
            removed_name, cleanup_warning = _clear_run_dir(run_root)
            if removed_name:
                removed.append(removed_name)
            if cleanup_warning:
                cleanup_warnings.append(cleanup_warning)
                print(f"Warning: {cleanup_warning}")
        except OSError as exc:
            raise gr.Error(f"Could not clear sequence cache {run_root}: {exc}") from exc

    _free_cuda_cache()
    payload = list(load_sequence(sequence_dir, exr_gamma, resume_from_tmp=False,
                                 prompt_mode=prompt_mode, concept_prompt=concept_prompt,
                                 processing_resolution=processing_resolution))
    if len(payload) != UI_OUTPUT_COUNT:
        raise gr.Error(f"Internal UI output mismatch: expected {UI_OUTPUT_COUNT}, got {len(payload)}")
    suffix = f" Cleared cache run(s): {', '.join(removed)}." if removed else " No prior cache run was found; loaded fresh."
    if cleanup_warnings:
        suffix += " Deferred cleanup warning: " + " | ".join(cleanup_warnings)
    payload[UI_OUTPUT_STATUS_INDEX] = str(payload[UI_OUTPUT_STATUS_INDEX]) + suffix
    _append_debug_line(payload[UI_OUTPUT_STATUS_INDEX])
    payload[UI_OUTPUT_DEBUG_INDEX] = _debug_console_text()
    return tuple(payload)


def delete_tmp_data(state):
    """Delete all production app tmp runs/settings, then clear the loaded UI state."""
    try:
        if APP_TMP_ROOT.exists():
            shutil.rmtree(APP_TMP_ROOT)
        APP_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise gr.Error(f"Could not delete tmp data under {APP_TMP_ROOT}: {exc}") from exc

    _free_cuda_cache()
    blank = None
    status_message = f"Deleted tmp data under {APP_TMP_ROOT}. Load a sequence to start fresh."
    _append_debug_line(status_message)
    return (
        None,
        None,
        None,
        None,
        blank,
        gr.update(minimum=0, maximum=0, value=0, interactive=False),
        gr.update(minimum=0, maximum=0, value=0, interactive=False),
        "No sequence loaded.",
        "No keyframes yet.",
        '',
        '',
        status_message,
        _debug_console_text(),
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


def set_prompt_mode(state, prompt_mode, concept_prompt):
    if state is None:
        return "Load a sequence first."
    state['prompt_mode'] = str(prompt_mode or 'Point keyframes')
    state['concept_prompt'] = str(concept_prompt or '').strip()
    state['preview_masks'] = {}
    _invalidate_generated_results(state)
    return _keyframe_summary(state)


def update_concept_prompt(state, concept_prompt):
    if state is None:
        return "Load a sequence first."
    state['concept_prompt'] = str(concept_prompt or '').strip()
    state['preview_masks'] = {}
    _invalidate_generated_results(state)
    return _keyframe_summary(state)


def preview_text_prompt_on_current_frame(state, prompt_mode, concept_prompt, sam_output_prob_thresh):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if sam3_tracker is None:
        raise gr.Error('Initialize models first.')
    if str(prompt_mode or '') != 'Text concept':
        raise gr.Error('Switch Prompt Mode to Text concept before previewing a text prompt.')
    text = str(concept_prompt or '').strip()
    if not text:
        raise gr.Error('Enter a concept prompt first.')
    state['prompt_mode'] = 'Text concept'
    state['concept_prompt'] = text
    _compute_preview_mask(state, state['current_frame_idx'], sam_output_prob_thresh=sam_output_prob_thresh)
    _invalidate_generated_results(state)
    return _ui_state_payload(
        state,
        f"Previewed text concept `{text}` on current frame with SAM threshold {_sam_output_prob_thresh(sam_output_prob_thresh):.3f}.",
    )


def add_point(state, prompt_mode, point_mode, sam_output_prob_thresh, evt: gr.SelectData):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if sam3_tracker is None:
        raise gr.Error('Initialize models first.')
    if str(prompt_mode or '') == 'Text concept':
        raise gr.Error('Switch Prompt Mode to Point keyframes before adding points.')
    state['prompt_mode'] = 'Point keyframes'
    if evt is None or evt.index is None:
        raise gr.Error('Click data was not received by Gradio.')

    frame_idx = state['current_frame_idx']
    prompts = _prompt_data_for_frame(state, frame_idx)
    x, y = int(evt.index[0]), int(evt.index[1])
    if not _point_in_content(state, frame_idx, x, y):
        raise gr.Error('Click inside the image area, not the letterbox padding.')
    prompts['points'].append([x, y])
    prompts['labels'].append(1 if point_mode == 'Positive' else 0)
    _compute_preview_mask(state, frame_idx, sam_output_prob_thresh=sam_output_prob_thresh)
    _invalidate_generated_results(state)
    return _ui_state_payload(
        state,
        f"Added {point_mode.lower()} point at ({x}, {y}) on frame {frame_idx} with SAM threshold {_sam_output_prob_thresh(sam_output_prob_thresh):.3f}.",
    )


def undo_point(state, sam_output_prob_thresh):
    if state is None:
        raise gr.Error('Load a sequence first.')
    prompts = _prompt_data_for_frame(state, state['current_frame_idx'])
    if not prompts['points']:
        return _ui_state_payload(state, 'No points to undo on this frame.')
    prompts['points'].pop()
    prompts['labels'].pop()
    _compute_preview_mask(state, state['current_frame_idx'], sam_output_prob_thresh=sam_output_prob_thresh)
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


def _ensure_loaded_processing_resolution(state, processing_resolution):
    if processing_resolution is None:
        return
    selected_size = _processing_work_size(processing_resolution)
    loaded_size = _work_size_from_state(state)
    if selected_size != loaded_size:
        raise gr.Error(
            f"Processing Resolution is set to {selected_size[0]}x{selected_size[1]}, "
            f"but the loaded sequence cache is {loaded_size[0]}x{loaded_size[1]}. "
            "Click Load Sequence after changing Processing Resolution."
        )


def generate_sam3_masks(state, prompt_mode=None, concept_prompt=None, sam_output_prob_thresh=0.5,
                        processing_resolution=None):
    if state is None:
        raise gr.Error('Load a sequence first.')

    _ensure_loaded_processing_resolution(state, processing_resolution)
    sam_output_prob_thresh = _sam_output_prob_thresh(sam_output_prob_thresh)
    state['sam_output_prob_thresh'] = sam_output_prob_thresh
    _unload_videomama_pipeline()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = _ensure_sam3_tracker(device)

    if prompt_mode is not None:
        state['prompt_mode'] = str(prompt_mode or 'Point keyframes')
    if concept_prompt is not None and (str(concept_prompt or '').strip() or not state.get('concept_prompt')):
        state['concept_prompt'] = str(concept_prompt or '').strip()

    run_root = Path(state['run_root'])
    mask_dir = run_root / 'sam3_masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    if state.get('prompt_mode') == 'Text concept':
        text = str(state.get('concept_prompt') or '').strip()
        if not text:
            raise gr.Error('Enter a concept prompt first.')
        with (run_root / 'concept_prompt.json').open('w', encoding='utf-8') as f:
            json.dump({'text': text, 'frame_index': state['current_frame_idx']}, f, indent=2)
        try:
            masks = tracker.track_video_from_dir_with_text(
                state['cache_dir'],
                text,
                frame_idx=state['current_frame_idx'],
                output_prob_thresh=sam_output_prob_thresh,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
    else:
        prompts = _normalized_prompt_dict(state)
        with (run_root / 'keyframe_prompts.json').open('w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in prompts.items()}, f, indent=2)
        masks = tracker.track_video_from_dir(
            state['cache_dir'],
            prompts,
            output_prob_thresh=sam_output_prob_thresh,
        )

    # SAM 3 propagation leaves a large block of cached VRAM behind. Release it so
    # the co-resident VideoMaMa matting pass has room on the same GPU.
    _free_cuda_cache()

    for frame_idx, (frame_name, mask) in enumerate(zip(state['frame_names'], masks)):
        transform = (state.get('frame_transforms') or [{}])[frame_idx]
        source_mask = _work_mask_to_source(mask, transform)
        Image.fromarray(source_mask).save(mask_dir / f"{Path(frame_name).stem}.png")

    state['generated_masks_dir'] = str(mask_dir)
    state['generated_masks_sam_output_prob_thresh'] = sam_output_prob_thresh
    state['videomama_output_dir'] = None
    state['alpha_output_dir'] = None
    return _ui_state_payload(
        state,
        f"Generated {len(masks)} SAM 3 masks at source resolution using {_keyframe_summary(state)} "
        f"(SAM threshold {sam_output_prob_thresh:.3f})",
    )


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


def run_sequence(state, chunk_size, overlap, range_start=0, range_end=-1, custom_output_dir='',
                 prompt_mode=None, concept_prompt=None, sam_output_prob_thresh=0.5,
                 videomama_mask_cond_mode='vae', videomama_seed=42, videomama_fps=7,
                 videomama_motion_bucket_id=127, videomama_noise_aug_strength=0.0,
                 processing_resolution=None, refine_edges_against_plate=False,
                 free_gpu_after_run=False):
    if state is None:
        raise gr.Error('Load a sequence first.')
    _ensure_loaded_processing_resolution(state, processing_resolution)
    if prompt_mode is not None:
        state['prompt_mode'] = str(prompt_mode or 'Point keyframes')
    if concept_prompt is not None and (str(concept_prompt or '').strip() or not state.get('concept_prompt')):
        state['concept_prompt'] = str(concept_prompt or '').strip()

    sam_output_prob_thresh = _sam_output_prob_thresh(sam_output_prob_thresh)
    state['sam_output_prob_thresh'] = sam_output_prob_thresh
    videomama_mask_cond_mode = str(videomama_mask_cond_mode or 'vae')
    if videomama_mask_cond_mode not in {'vae', 'interpolate'}:
        raise gr.Error(f"Unsupported VideoMaMa mask conditioning mode: {videomama_mask_cond_mode}")
    videomama_seed = int(videomama_seed)
    videomama_fps = max(1, int(videomama_fps))
    videomama_motion_bucket_id = max(0, min(int(videomama_motion_bucket_id), 255))
    videomama_noise_aug_strength = max(0.0, min(float(videomama_noise_aug_strength), 1.0))
    refine_edges_against_plate = bool(refine_edges_against_plate)
    free_gpu_after_run = bool(free_gpu_after_run)

    generated_mask_thresh = state.get('generated_masks_sam_output_prob_thresh')
    masks_need_regen = (
        not state.get('generated_masks_dir')
        or generated_mask_thresh is None
        or abs(float(generated_mask_thresh) - sam_output_prob_thresh) > 1e-6
    )
    if masks_need_regen:
        state = generate_sam3_masks(
            state,
            prompt_mode=state.get('prompt_mode'),
            concept_prompt=state.get('concept_prompt'),
            sam_output_prob_thresh=sam_output_prob_thresh,
        )[4]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _unload_sam3_tracker()
    pipeline = _ensure_videomama_pipeline(device)

    total_frames = len(state['frame_paths'])
    range_start, range_end = _resolve_range(range_start, range_end, total_frames)
    is_subrange = not (range_start == 0 and range_end == total_frames - 1)

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    step = max(1, chunk_size - overlap)

    run_root = Path(state['run_root'])
    work_size = _work_size_from_state(state)
    output_dir, alpha_dir = _resolve_videomama_output_dirs(run_root, custom_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    print(
        f"Running VideoMaMa over frames [{range_start}, {range_end}] "
        f"({range_end - range_start + 1} of {total_frames}) with chunk_size={chunk_size}, overlap={overlap}, "
        f"mask_cond_mode={videomama_mask_cond_mode}, seed={videomama_seed}, fps={videomama_fps}, "
        f"motion_bucket_id={videomama_motion_bucket_id}, noise_aug_strength={videomama_noise_aug_strength:.4f}, "
        f"processing_size={work_size[0]}x{work_size[1]}, refine_edges={refine_edges_against_plate}"
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

        output_frames = videomama(
            pipeline,
            frames_np,
            masks_np,
            seed=videomama_seed,
            mask_cond_mode=videomama_mask_cond_mode,
            fps=videomama_fps,
            motion_bucket_id=videomama_motion_bucket_id,
            noise_aug_strength=videomama_noise_aug_strength,
            target_size=work_size,
        )
        # Keep every frame of the first chunk in this run; for later chunks drop the
        # leading `overlap` frames, which the previous chunk already wrote (the
        # overlap re-processes shared frames so segment boundaries blend cleanly).
        keep_from = 0 if start == range_start else overlap
        for local_idx in range(keep_from, len(output_frames)):
            frame_name = chunk_frame_names[local_idx]
            output_frame = output_frames[local_idx]
            source_size = tuple(state['frame_sizes'][start + local_idx])
            if output_frame.shape[:2] != (source_size[1], source_size[0]):
                output_frame = np.array(Image.fromarray(output_frame).resize(source_size, Image.Resampling.BILINEAR))
            if refine_edges_against_plate:
                source_frame = frames_np[local_idx]
                source_mask = masks_np[local_idx]
                refined_alpha = _refine_alpha_against_plate(source_frame, output_frame, source_mask)
                output_frame = np.repeat(refined_alpha[:, :, None], 3, axis=2)
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
    if free_gpu_after_run:
        _unload_sam3_tracker()
        _unload_videomama_pipeline()
    # Jump the viewer to the start of what was just processed.
    state['current_frame_idx'] = range_start
    scope = f"frames [{range_start}, {range_end}]" if is_subrange else "the whole sequence"
    return _ui_state_payload(
        state,
        f"Saved VideoMaMa outputs for {scope} ({processed} frame(s) written) to {output_dir}. "
        f"VideoMaMa: {videomama_mask_cond_mode}, seed {videomama_seed}, fps {videomama_fps}, "
        f"motion bucket {videomama_motion_bucket_id}, noise {videomama_noise_aug_strength:.4f}, "
        f"processing {work_size[0]}x{work_size[1]}, edge refine {refine_edges_against_plate}, "
        f"free GPU after run {free_gpu_after_run}.",
    )


with gr.Blocks(title='VideoMaMa Production Sequence App', css=APP_CSS) as demo:
    gr.Markdown('# VideoMaMa Production Sequence App')

    state = gr.State(None)

    _settings = _load_ui_settings()

    status = gr.Textbox(label='Status', interactive=False)

    with gr.Row():
        with gr.Column(scale=4, min_width=360):
            with gr.Group():
                sequence_dir = gr.Textbox(label='Sequence Directory', value=_settings['sequence_dir'])
                with gr.Row():
                    exr_gamma = gr.Number(label='EXR Gamma', value=_settings['exr_gamma'], precision=2)
                    resume_from_tmp = gr.Checkbox(label='Resume from tmp', value=_settings['resume_from_tmp'])
                with gr.Row():
                    init_btn = gr.Button('Initialize Models', variant='primary')
                    load_btn = gr.Button('Load Sequence', variant='primary')
                with gr.Row():
                    clear_cache_reload_btn = gr.Button('Clear Sequence Cache + Reload')
                    delete_tmp_btn = gr.Button('Delete Tmp Data', elem_classes=['danger-button'])

            with gr.Group():
                prompt_mode = gr.Radio(
                    ['Point keyframes', 'Text concept'],
                    value=_settings['prompt_mode'],
                    label='Prompt Mode',
                )
                concept_prompt = gr.Textbox(label='Concept Prompt', value=_settings['concept_prompt'])
                point_mode = gr.Radio(['Positive', 'Negative'], value=_settings['point_mode'], label='Point Mode')
                with gr.Row():
                    undo_btn = gr.Button('Undo Point')
                    clear_frame_btn = gr.Button('Clear Frame Points')
                with gr.Row():
                    clear_all_btn = gr.Button('Clear All Prompts')
                    preview_text_btn = gr.Button('Preview Text Concept')
                gen_masks_btn = gr.Button('Generate SAM 3 Masks', variant='primary')
                keyframes_info = gr.Textbox(label='Prompt Summary', interactive=False)

            with gr.Group():
                quality_preset = gr.Radio(
                    ['Balanced', 'Fine Detail', 'Tight Matte', 'Custom'],
                    value=_settings['quality_preset'],
                    label='Quality Preset',
                )
                sam_output_prob_thresh = gr.Slider(
                    label='SAM Mask Threshold',
                    minimum=0.1,
                    maximum=0.9,
                    value=_settings['sam_output_prob_thresh'],
                    step=0.01,
                )
                videomama_mask_cond_mode = gr.Radio(
                    ['vae', 'interpolate'],
                    value=_settings['videomama_mask_cond_mode'],
                    label='VideoMaMa Mask Guide',
                )
                processing_resolution = gr.Dropdown(
                    list(PROCESSING_RESOLUTIONS.keys()),
                    value=_settings['processing_resolution'],
                    label='Processing Resolution',
                )
                refine_edges_against_plate = gr.Checkbox(
                    label='Refine Edges Against Plate',
                    value=_settings['refine_edges_against_plate'],
                )
                free_gpu_after_run = gr.Checkbox(
                    label='Free GPU After Run',
                    value=_settings['free_gpu_after_run'],
                )
                with gr.Row():
                    videomama_seed = gr.Number(label='VideoMaMa Seed', value=_settings['videomama_seed'], precision=0)
                    videomama_fps = gr.Number(label='VideoMaMa FPS', value=_settings['videomama_fps'], precision=0)
                with gr.Row():
                    videomama_motion_bucket_id = gr.Number(
                        label='VideoMaMa Motion Bucket',
                        value=_settings['videomama_motion_bucket_id'],
                        precision=0,
                    )
                    videomama_noise_aug_strength = gr.Number(
                        label='VideoMaMa Noise',
                        value=_settings['videomama_noise_aug_strength'],
                        precision=3,
                    )

            with gr.Group():
                with gr.Row():
                    chunk_size = gr.Number(label='Chunk Size', value=_settings['chunk_size'], precision=0)
                    overlap = gr.Number(label='Overlap', value=_settings['overlap'], precision=0)
                with gr.Row():
                    range_start = gr.Number(label='Start Frame', value=_settings['range_start'], precision=0)
                    range_end = gr.Number(label='End Frame (-1 = last)', value=_settings['range_end'], precision=0)
                custom_output_dir = gr.Textbox(
                    label='Custom Output Directory',
                    value=_settings['custom_output_dir'],
                )
                run_btn = gr.Button('Run VideoMaMa', variant='primary')
                unload_models_btn = gr.Button('Unload Models / Free GPU')
                mask_dir = gr.Textbox(label='Mask Directory', interactive=False)
                output_dir = gr.Textbox(label='VideoMaMa Output Directory', interactive=False)

            with gr.Accordion('Debug Console', open=False):
                debug_console = gr.Textbox(
                    label='Console',
                    value=_debug_console_text(),
                    lines=18,
                    max_lines=28,
                    interactive=False,
                    elem_classes=['debug-console'],
                )
                with gr.Row():
                    refresh_debug_btn = gr.Button('Refresh Console')
                    clear_debug_btn = gr.Button('Clear Console')

        with gr.Column(scale=7, min_width=560):
            frame_slider = gr.Slider(label='Current Frame', minimum=0, maximum=0, value=0, step=1, interactive=False)
            with gr.Row():
                prev_btn = gr.Button('Prev Frame')
                next_btn = gr.Button('Next Frame')
            frame_info = gr.Textbox(label='Frame Info', interactive=False)

            with gr.Tabs():
                with gr.Tab('Review'):
                    preview = gr.Image(
                        label='Prompt Overlay',
                        type='numpy',
                        interactive=False,
                        sources=[],
                        height=600,
                        show_download_button=False,
                        show_fullscreen_button=True,
                        elem_classes=['accurate-preview'],
                    )
                with gr.Tab('Mask'):
                    mask_img = gr.Image(label='Current SAM 3 Mask', type='numpy', height=600, elem_classes=['accurate-preview'])
                with gr.Tab('VideoMaMa'):
                    output_img = gr.Image(label='Current VideoMaMa Output', type='numpy', height=600, elem_classes=['accurate-preview'])
                with gr.Tab('Large Prompt'):
                    large_frame_slider = gr.Slider(label='Large View Frame', minimum=0, maximum=0, value=0, step=1, interactive=False)
                    preview_large = gr.Image(
                        label='Large Prompt Overlay',
                        type='numpy',
                        interactive=False,
                        sources=[],
                        height=720,
                        show_download_button=False,
                        show_fullscreen_button=True,
                        elem_id='large_prompt_preview',
                        elem_classes=['accurate-preview'],
                    )

    ui_outputs = [
        preview, preview_large, mask_img, output_img, state, frame_slider, large_frame_slider,
        frame_info, keyframes_info, mask_dir, output_dir, status, debug_console,
    ]

    # Persist the input panel whenever any setting changes, so the next launch
    # restores the same session configuration.
    settings_inputs = [
        sequence_dir, exr_gamma, prompt_mode, concept_prompt, point_mode,
        quality_preset, sam_output_prob_thresh, videomama_mask_cond_mode,
        videomama_seed, videomama_fps, videomama_motion_bucket_id,
        videomama_noise_aug_strength, processing_resolution,
        refine_edges_against_plate, free_gpu_after_run, chunk_size, overlap,
        custom_output_dir, resume_from_tmp, range_start, range_end,
    ]
    for _component in settings_inputs:
        _component.change(_save_ui_settings, inputs=settings_inputs, outputs=None)

    quality_preset.change(
        apply_quality_preset,
        inputs=quality_preset,
        outputs=[
            sam_output_prob_thresh,
            videomama_mask_cond_mode,
            videomama_fps,
            videomama_motion_bucket_id,
            videomama_noise_aug_strength,
        ],
    )

    init_btn.click(initialize_models, outputs=[status, debug_console])
    refresh_debug_btn.click(refresh_debug_console, outputs=debug_console)
    clear_debug_btn.click(clear_debug_console, outputs=debug_console)
    unload_models_btn.click(unload_models_for_gpu, outputs=[status, debug_console])
    load_btn.click(
        load_sequence,
        inputs=[sequence_dir, exr_gamma, resume_from_tmp, prompt_mode, concept_prompt, processing_resolution],
        outputs=ui_outputs,
    )
    clear_cache_reload_btn.click(
        clear_sequence_cache_and_reload,
        inputs=[state, sequence_dir, exr_gamma, prompt_mode, concept_prompt, processing_resolution],
        outputs=ui_outputs,
    )
    delete_tmp_btn.click(
        delete_tmp_data,
        inputs=state,
        outputs=ui_outputs,
    )
    prompt_mode.change(
        set_prompt_mode,
        inputs=[state, prompt_mode, concept_prompt],
        outputs=keyframes_info,
    )
    concept_prompt.change(
        update_concept_prompt,
        inputs=[state, concept_prompt],
        outputs=keyframes_info,
    )
    frame_slider.release(
        select_frame,
        inputs=[state, frame_slider],
        outputs=ui_outputs,
    )
    large_frame_slider.release(
        select_frame,
        inputs=[state, large_frame_slider],
        outputs=ui_outputs,
    )
    prev_btn.click(
        prev_frame,
        inputs=state,
        outputs=ui_outputs,
    )
    next_btn.click(
        next_frame,
        inputs=state,
        outputs=ui_outputs,
    )
    preview.select(
        add_point,
        inputs=[state, prompt_mode, point_mode, sam_output_prob_thresh],
        outputs=ui_outputs,
    )
    preview_large.select(
        add_point,
        inputs=[state, prompt_mode, point_mode, sam_output_prob_thresh],
        outputs=ui_outputs,
    )
    undo_btn.click(
        undo_point,
        inputs=[state, sam_output_prob_thresh],
        outputs=ui_outputs,
    )
    clear_frame_btn.click(
        clear_frame_points,
        inputs=state,
        outputs=ui_outputs,
    )
    clear_all_btn.click(
        clear_all_points,
        inputs=state,
        outputs=ui_outputs,
    )
    preview_text_btn.click(
        preview_text_prompt_on_current_frame,
        inputs=[state, prompt_mode, concept_prompt, sam_output_prob_thresh],
        outputs=ui_outputs,
    )
    gen_masks_btn.click(
        generate_sam3_masks,
        inputs=[state, prompt_mode, concept_prompt, sam_output_prob_thresh, processing_resolution],
        outputs=ui_outputs,
    )
    run_btn.click(
        run_sequence,
        inputs=[
            state, chunk_size, overlap, range_start, range_end, custom_output_dir,
            prompt_mode, concept_prompt, sam_output_prob_thresh,
            videomama_mask_cond_mode, videomama_seed, videomama_fps,
            videomama_motion_bucket_id, videomama_noise_aug_strength,
            processing_resolution, refine_edges_against_plate, free_gpu_after_run,
        ],
        outputs=ui_outputs,
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
