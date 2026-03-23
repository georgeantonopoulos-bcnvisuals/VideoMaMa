"""
Single-frame production UI for SAM2 point prompting + VideoMaMa inference.

This app is intended for Python 3.10+ environments because the upstream `sam2`
package does not support Python 3.9.
"""

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

DEFAULT_FRAME = "/mnt/production/project/ntf_fire/work/Editing/102_CameraSpin_4K/102_CameraSpin_4k1001.exr"
DEFAULT_EXR_GAMMA = 1.0
MASK_COLOR = 3
MASK_ALPHA = 0.7
CONTOUR_COLOR = 1
CONTOUR_WIDTH = 5
POINT_COLOR_POS = 8
POINT_COLOR_NEG = 1
POINT_ALPHA = 0.9
POINT_RADIUS = 15

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


def load_frame(frame_path: str, exr_gamma: float):
    if not frame_path:
        raise gr.Error("Provide an EXR frame path.")
    frame = _read_exr_rgb(frame_path, exr_gamma=exr_gamma)
    state = {
        "frame_path": frame_path,
        "frame": frame,
        "points": [],
        "labels": [],
        "mask": None,
    }
    return frame, state, "Frame loaded. Click positive/negative points on the image."


def _render_preview(frame: np.ndarray, points, labels, mask):
    preview = frame.copy()
    if mask is not None:
        preview = mask_painter(preview, mask, MASK_COLOR, MASK_ALPHA, CONTOUR_COLOR, CONTOUR_WIDTH)

    positive_points = np.array([points[i] for i in range(len(points)) if labels[i] == 1], dtype=np.int32)
    negative_points = np.array([points[i] for i in range(len(points)) if labels[i] == 0], dtype=np.int32)

    if len(positive_points) > 0:
        preview = point_painter(preview, positive_points, POINT_COLOR_POS, POINT_ALPHA, POINT_RADIUS, CONTOUR_COLOR, 2)
    if len(negative_points) > 0:
        preview = point_painter(preview, negative_points, POINT_COLOR_NEG, POINT_ALPHA, POINT_RADIUS, CONTOUR_COLOR, 2)

    return preview


def add_point(state, point_mode, evt: gr.SelectData):
    if state is None or state.get("frame") is None:
        raise gr.Error("Load a frame first.")
    if sam2_tracker is None:
        raise gr.Error("Initialize models first.")

    if evt is None or evt.index is None:
        raise gr.Error("Click data was not received by Gradio.")

    # Gradio image select events provide (x, y) pixel coordinates for clicks.
    x, y = int(evt.index[0]), int(evt.index[1])
    label = 1 if point_mode == "Positive" else 0
    state["points"].append([x, y])
    state["labels"].append(label)
    state["mask"] = sam2_tracker.get_first_frame_mask(state["frame"], state["points"], state["labels"])

    preview = _render_preview(state["frame"], state["points"], state["labels"], state["mask"])
    return preview, state["mask"], state, f"Added {point_mode.lower()} point at ({x}, {y})."


def undo_point(state):
    if state is None or state.get("frame") is None:
        raise gr.Error("Load a frame first.")
    if not state["points"]:
        preview = _render_preview(state["frame"], [], [], None)
        return preview, None, state, "No points to undo."

    state["points"].pop()
    state["labels"].pop()
    if state["points"] and sam2_tracker is not None:
        state["mask"] = sam2_tracker.get_first_frame_mask(state["frame"], state["points"], state["labels"])
    else:
        state["mask"] = None

    preview = _render_preview(state["frame"], state["points"], state["labels"], state["mask"])
    return preview, state["mask"], state, "Removed last point."


def clear_points(state):
    if state is None or state.get("frame") is None:
        raise gr.Error("Load a frame first.")
    state["points"] = []
    state["labels"] = []
    state["mask"] = None
    return state["frame"], None, state, "Cleared all points."


def run_single_frame(state):
    if state is None or state.get("frame") is None:
        raise gr.Error("Load a frame first.")
    if state.get("mask") is None:
        raise gr.Error("Add at least one point and generate a mask first.")
    if videomama_pipeline is None:
        raise gr.Error("Initialize models first.")

    output = videomama(videomama_pipeline, [state["frame"]], [state["mask"]])[0]

    out_dir = Path("tmp/production_frame_app") / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(state["frame"]).save(out_dir / "input.png")
    Image.fromarray(state["mask"]).save(out_dir / "mask.png")
    Image.fromarray(output).save(out_dir / "output.png")

    return output, str(out_dir), f"Saved input/mask/output to {out_dir}"


with gr.Blocks(title="VideoMaMa Production Frame App") as demo:
    gr.Markdown("# VideoMaMa Production Frame App")
    gr.Markdown("Load one EXR frame, click positive/negative prompts for SAM2, then run one-frame VideoMaMa inference.")
    gr.Markdown("After loading the frame, click directly on the left preview image to add points. Initialize models first.")

    state = gr.State(None)

    with gr.Row():
        frame_path = gr.Textbox(label="EXR Frame Path", value=DEFAULT_FRAME)
        exr_gamma = gr.Number(label="EXR Gamma", value=DEFAULT_EXR_GAMMA, precision=2)
        point_mode = gr.Radio(["Positive", "Negative"], value="Positive", label="Point Mode")

    with gr.Row():
        init_btn = gr.Button("Initialize Models")
        load_btn = gr.Button("Load Frame")
        undo_btn = gr.Button("Undo Point")
        clear_btn = gr.Button("Clear Points")
        run_btn = gr.Button("Run VideoMaMa")

    with gr.Row():
        preview = gr.Image(
            label="Frame / Prompt Preview",
            type="numpy",
            interactive=False,
            sources=[],
            show_download_button=False,
            show_fullscreen_button=False,
        )
        mask_img = gr.Image(label="SAM2 Mask", type="numpy")
        output_img = gr.Image(label="VideoMaMa Output", type="numpy")

    save_dir = gr.Textbox(label="Saved Output Directory")
    status = gr.Textbox(label="Status")

    init_btn.click(initialize_models, outputs=status)
    load_btn.click(load_frame, inputs=[frame_path, exr_gamma], outputs=[preview, state, status])
    preview.select(add_point, inputs=[state, point_mode], outputs=[preview, mask_img, state, status])
    undo_btn.click(undo_point, inputs=state, outputs=[preview, mask_img, state, status])
    clear_btn.click(clear_points, inputs=state, outputs=[preview, mask_img, state, status])
    run_btn.click(run_single_frame, inputs=state, outputs=[output_img, save_dir, status])


if __name__ == "__main__":
    server_name = os.environ.get("VIDEOMAMA_UI_HOST", "127.0.0.1")
    server_port = int(os.environ.get("VIDEOMAMA_UI_PORT", "7861"))
    share = os.environ.get("VIDEOMAMA_UI_SHARE", "1") not in {"0", "false", "False"}
    demo.launch(server_name=server_name, server_port=server_port, share=share)
