"""
Production sequence UI for multi-keyframe SAM2 prompting + VideoMaMa inference.

This app is intended for Python 3.10+ environments because the upstream `sam2`
package does not support Python 3.9.
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

import gradio as gr
import numpy as np
from PIL import Image

try:
    import OpenEXR
    import Imath
except ImportError as exc:
    raise RuntimeError("OpenEXR and Imath are required for production_frame_app.py") from exc

from sam2_wrapper_hf import load_sam2_tracker
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

sam2_tracker = None
videomama_pipeline = None


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


def _run_root(base_name: str):
    safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in base_name) or 'sequence'
    return Path('tmp/production_sequence_app') / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name}"


def initialize_models():
    global sam2_tracker, videomama_pipeline
    if sam2_tracker is not None and videomama_pipeline is not None:
        return "Models already loaded."

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_ckpt = os.environ.get("SAM2_CHECKPOINT_PATH", "checkpoints/sam2.1_hiera_large.pt")

    sam2_tracker = load_sam2_tracker(checkpoint_path=sam2_ckpt, device=device)
    videomama_pipeline = load_videomama_pipeline(device=device)
    return f"Loaded SAM2 and VideoMaMa on {device}."


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
    if prompts['points'] and sam2_tracker is not None:
        frame = _load_cached_frame(state, frame_idx)
        mask = sam2_tracker.get_frame_mask(frame, prompts['points'], prompts['labels'])
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


def load_sequence(sequence_dir: str, exr_gamma: float):
    frame_paths = _discover_sequence_files(sequence_dir)
    run_root = _run_root(Path(sequence_dir).name)
    cache_dir = run_root / 'sam2_frames'
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_frame_paths = []
    original_sizes = []
    print(f"Preparing {len(frame_paths)} frames for SAM2/UI cache...")
    for idx, frame_path in enumerate(frame_paths):
        frame = _load_rgb_frame(str(frame_path), exr_gamma=exr_gamma)
        original_sizes.append([int(frame.shape[1]), int(frame.shape[0])])
        working_frame = _resize_rgb_frame(frame)
        cache_path = cache_dir / f"{idx:05d}.jpg"
        Image.fromarray(working_frame).save(cache_path, quality=95)
        cache_frame_paths.append(str(cache_path))

    state = {
        'sequence_dir': str(Path(sequence_dir).resolve()),
        'frame_paths': [str(path) for path in frame_paths],
        'frame_names': [path.name for path in frame_paths],
        'cache_dir': str(cache_dir),
        'cache_frame_paths': cache_frame_paths,
        'current_frame_idx': 0,
        'prompts_by_frame': {},
        'preview_masks': {},
        'generated_masks_dir': None,
        'videomama_output_dir': None,
        'alpha_output_dir': None,
        'run_root': str(run_root),
        'exr_gamma': float(exr_gamma),
        'original_sizes': original_sizes,
    }

    preview, mask, output_preview = _render_frame_preview(state, 0)
    return (
        preview,
        mask,
        output_preview,
        state,
        gr.update(minimum=0, maximum=len(frame_paths) - 1, value=0, step=1, interactive=True),
        _frame_info(state),
        _keyframe_summary(state),
        '',
        '',
        f"Loaded {len(frame_paths)} frames from {sequence_dir}. Click any frame to add keyframe prompts.",
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
    if sam2_tracker is None:
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


def generate_sam2_masks(state):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if sam2_tracker is None:
        raise gr.Error('Initialize models first.')

    prompts = _normalized_prompt_dict(state)
    run_root = Path(state['run_root'])
    mask_dir = run_root / 'sam2_masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    with (run_root / 'keyframe_prompts.json').open('w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in prompts.items()}, f, indent=2)

    masks = sam2_tracker.track_video_from_dir(state['cache_dir'], prompts)
    for frame_name, mask in zip(state['frame_names'], masks):
        Image.fromarray(mask).save(mask_dir / f"{Path(frame_name).stem}.png")

    state['generated_masks_dir'] = str(mask_dir)
    state['videomama_output_dir'] = None
    state['alpha_output_dir'] = None
    return _ui_state_payload(state, f"Generated {len(masks)} SAM2 masks using {_keyframe_summary(state)}")


def run_sequence(state, chunk_size, overlap):
    if state is None:
        raise gr.Error('Load a sequence first.')
    if videomama_pipeline is None:
        raise gr.Error('Initialize models first.')

    if not state.get('generated_masks_dir'):
        state = generate_sam2_masks(state)[3]

    total_frames = len(state['frame_paths'])
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

    print(f"Running VideoMaMa over {total_frames} frames with chunk_size={chunk_size}, overlap={overlap}")
    for start in range(0, total_frames, step):
        end = min(total_frames, start + chunk_size)
        chunk_frame_paths = state['frame_paths'][start:end]
        chunk_frame_names = state['frame_names'][start:end]

        frames_np = [_load_rgb_frame(frame_path, exr_gamma=state['exr_gamma']) for frame_path in chunk_frame_paths]
        masks_np = []
        for frame_name in chunk_frame_names:
            mask_path = Path(state['generated_masks_dir']) / f"{Path(frame_name).stem}.png"
            if not mask_path.exists():
                raise gr.Error(f"Missing SAM2 mask for {frame_name}: {mask_path}")
            masks_np.append(np.array(Image.open(mask_path).convert('L')))

        output_frames = videomama(videomama_pipeline, frames_np, masks_np)
        keep_from = 0 if start == 0 else overlap
        for local_idx in range(keep_from, len(output_frames)):
            frame_name = chunk_frame_names[local_idx]
            output_frame = output_frames[local_idx]
            output_path = output_dir / f"{Path(frame_name).stem}.png"
            alpha_path = alpha_dir / f"{Path(frame_name).stem}.png"
            Image.fromarray(output_frame).save(output_path)
            Image.fromarray(output_frame).convert('L').save(alpha_path)

        if end == total_frames:
            break

    state['videomama_output_dir'] = str(output_dir)
    state['alpha_output_dir'] = str(alpha_dir)
    return _ui_state_payload(
        state,
        f"Saved VideoMaMa sequence outputs to {output_dir} and alpha previews to {alpha_dir}",
    )


with gr.Blocks(title='VideoMaMa Production Sequence App') as demo:
    gr.Markdown('# VideoMaMa Production Sequence App')
    gr.Markdown('Load an EXR or image sequence, add SAM2 prompts on multiple keyframes, generate the full mask track, then run VideoMaMa over the whole shot.')
    gr.Markdown('Workflow: Initialize models, load the sequence, navigate frames, add prompts on the frames that need correction, generate SAM2 masks, then run the whole sequence.')

    state = gr.State(None)

    with gr.Row():
        sequence_dir = gr.Textbox(label='Sequence Directory', value=DEFAULT_SEQUENCE_DIR)
        exr_gamma = gr.Number(label='EXR Gamma', value=DEFAULT_EXR_GAMMA, precision=2)
        point_mode = gr.Radio(['Positive', 'Negative'], value='Positive', label='Point Mode')

    with gr.Row():
        chunk_size = gr.Number(label='VideoMaMa Chunk Size', value=16, precision=0)
        overlap = gr.Number(label='Chunk Overlap', value=4, precision=0)

    with gr.Row():
        init_btn = gr.Button('Initialize Models')
        load_btn = gr.Button('Load Sequence')
        prev_btn = gr.Button('Prev Frame')
        next_btn = gr.Button('Next Frame')
        undo_btn = gr.Button('Undo Point')
        clear_frame_btn = gr.Button('Clear Frame Points')
        clear_all_btn = gr.Button('Clear All Prompts')
        gen_masks_btn = gr.Button('Generate SAM2 Masks')
        run_btn = gr.Button('Run Whole Sequence')

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
        mask_img = gr.Image(label='Current SAM2 Mask', type='numpy')
        output_img = gr.Image(label='Current VideoMaMa Output', type='numpy')

    mask_dir = gr.Textbox(label='Saved Mask Directory')
    output_dir = gr.Textbox(label='Saved VideoMaMa Output Directory')
    status = gr.Textbox(label='Status')

    init_btn.click(initialize_models, outputs=status)
    load_btn.click(
        load_sequence,
        inputs=[sequence_dir, exr_gamma],
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
        generate_sam2_masks,
        inputs=state,
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )
    run_btn.click(
        run_sequence,
        inputs=[state, chunk_size, overlap],
        outputs=[preview, mask_img, output_img, state, frame_slider, frame_info, keyframes_info, mask_dir, output_dir, status],
    )


if __name__ == '__main__':
    server_name = os.environ.get('VIDEOMAMA_UI_HOST', '127.0.0.1')
    server_port = int(os.environ.get('VIDEOMAMA_UI_PORT', '7861'))
    share = os.environ.get('VIDEOMAMA_UI_SHARE', '1') not in {'0', 'false', 'False'}
    demo.launch(server_name=server_name, server_port=server_port, share=share)
