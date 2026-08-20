"""
Client side of the SAM2Matting integration.

Runs inside the SAM 3 production UI process. It never imports SAM2Matting — it
stages inputs on disk, launches demo/sam2matting_worker.py under the isolated
SAM2Matting venv, streams the worker's NDJSON progress, and yields float32
alphas back to the caller one frame at a time.

Keeping this boundary a process boundary rather than a module boundary is the
whole point: SAM2Matting ships its own modified top-level `sam2` and `sam3`
packages plus a different Torch build, and either would shadow or break the
official SAM 3 install that the tracking half of the pipeline depends on.
"""

import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import matting_backends as backends


class SAM2MattingError(RuntimeError):
    """Raised for any failure attributable to the SAM2Matting subprocess."""

    def __init__(self, message, kind="runtime", detail=""):
        super().__init__(message)
        self.kind = kind
        self.detail = detail


WORKER_SCRIPT = Path(__file__).resolve().parent / "sam2matting_worker.py"


@dataclass
class StagedWindow:
    """Where a window's inputs live and how they map back to source pixels."""

    frames_dir: Path
    masks_dir: Path
    alpha_dir: Path
    frame_indices: list  # global frame indices, in processing order
    mask_paths: dict  # local index -> staged mask path
    roi: dict  # {'enabled', 'x', 'y', 'width', 'height'}
    staged_size: tuple  # (width, height) actually written to disk

    def local_index(self, global_index):
        return self.frame_indices.index(int(global_index))


def preflight(repo_root, spec, checkpoints_root):
    """Verify the isolated runtime exists before promising the artist a matte.

    Returns the runtime info dict (upstream commit, torch version, checkpoint
    digests) recorded by scripts/bootstrap_sam2matting.sh.
    """
    spec = backends.resolve_backend(spec)
    if spec.family != "sam2matting":
        raise ValueError(f"{spec.backend_id} is not a SAM2Matting backend.")

    python_bin = backends.sam2matting_python(repo_root)
    home = backends.sam2matting_home(repo_root)
    checkpoint = backends.sam2matting_checkpoint_dir(checkpoints_root) / spec.checkpoint_name

    if not python_bin.is_file():
        raise SAM2MattingError(
            f"SAM2Matting environment is missing ({python_bin}). Build it with:\n"
            "  bash scripts/bootstrap_sam2matting.sh",
            kind="setup",
        )
    if not (home / "sam2").is_dir():
        raise SAM2MattingError(
            f"SAM2Matting source checkout is missing or incomplete ({home}). "
            "Re-run: bash scripts/bootstrap_sam2matting.sh",
            kind="setup",
        )
    if not checkpoint.is_file():
        raise SAM2MattingError(
            f"Missing SAM2Matting checkpoint {checkpoint}. Fetch it with:\n"
            "  bash scripts/bootstrap_sam2matting.sh checkpoints",
            kind="setup",
        )

    runtime = dict(backends.read_runtime_lock(repo_root))
    runtime.setdefault("commit", backends.SAM2MATTING_PINNED_COMMIT)
    runtime["python"] = str(python_bin)
    runtime["home"] = str(home)
    runtime["checkpoint_path"] = str(checkpoint)
    return runtime


def _resized_size(width, height, long_edge_cap):
    """Scale a ROI down only if it exceeds the cap; never upscale."""
    long_edge = max(int(width), int(height))
    cap = int(long_edge_cap or 0)
    if cap <= 0 or long_edge <= cap:
        return int(width), int(height)
    scale = cap / float(long_edge)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def stage_window(work_dir, frame_indices, load_rgb, load_mask, cond_indices, roi,
                 long_edge_cap=2048, progress=None):
    """Write one window's plates and conditioning masks where the worker can read them.

    `load_rgb(global_idx) -> HxWx3 uint8` and `load_mask(global_idx) -> HxW uint8`
    are supplied by the caller because plate decoding (EXR, OCIO, exposure) and
    mask lookup belong to the app, not here.

    Frames are named `%05d.jpg` because SAM2Matting's loader sorts a frame folder
    with `int(os.path.splitext(name)[0])` and would raise on any other naming.
    """
    work_dir = Path(work_dir)
    frames_dir = work_dir / "frames"
    masks_dir = work_dir / "masks"
    alpha_dir = work_dir / "alpha"
    for directory in (frames_dir, masks_dir, alpha_dir):
        directory.mkdir(parents=True, exist_ok=True)

    roi = dict(roi or {"enabled": False})
    cond_indices = sorted(set(int(i) for i in cond_indices))
    mask_paths = {}
    staged_size = None

    for local_idx, global_idx in enumerate(frame_indices):
        frame = np.asarray(load_rgb(global_idx))
        if roi.get("enabled"):
            x, y = int(roi["x"]), int(roi["y"])
            width, height = int(roi["width"]), int(roi["height"])
            frame = frame[y:y + height, x:x + width]
        target = _resized_size(frame.shape[1], frame.shape[0], long_edge_cap)
        if staged_size is None:
            staged_size = target
        elif target != staged_size:
            raise SAM2MattingError(
                "Frames in one window must share a size; "
                f"frame {global_idx} staged to {target}, expected {staged_size}."
            )
        image = Image.fromarray(frame)
        if (image.width, image.height) != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        # Maximum quality, no chroma subsampling: this JPEG is the model's only
        # view of the plate, and subsampled chroma smears fine colored edges.
        image.save(frames_dir / f"{local_idx:05d}.jpg", quality=100, subsampling=0)

        if global_idx in cond_indices:
            mask = np.asarray(load_mask(global_idx))
            if roi.get("enabled"):
                mask = mask[y:y + height, x:x + width]
            mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
            if (mask_image.width, mask_image.height) != target:
                mask_image = mask_image.resize(target, Image.Resampling.NEAREST)
            mask_path = masks_dir / f"{local_idx:05d}.png"
            mask_image.save(mask_path)
            mask_paths[local_idx] = str(mask_path)

        if progress is not None:
            progress(local_idx + 1, len(frame_indices))

    if not mask_paths:
        raise SAM2MattingError("No conditioning masks were staged for this window.")

    return StagedWindow(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        alpha_dir=alpha_dir,
        frame_indices=[int(i) for i in frame_indices],
        mask_paths=mask_paths,
        roi=roi,
        staged_size=staged_size,
    )


def build_job(spec, params, runtime, staged, device="auto"):
    """Serialize everything the worker needs into a plain dict."""
    spec = backends.resolve_backend(spec)
    params = params.normalized() if hasattr(params, "normalized") else params
    return {
        "backend_id": spec.backend_id,
        "variant": spec.variant,
        "config_name": spec.config_name,
        "checkpoint_path": runtime["checkpoint_path"],
        "sam2matting_home": runtime["home"],
        "frames_dir": str(staged.frames_dir),
        "alpha_dir": str(staged.alpha_dir),
        "masks": {str(k): v for k, v in staged.mask_paths.items()},
        # "auto" lets the worker detect CUDA in its own environment. The UI's
        # torch (if it even has one) says nothing about the worker's.
        "device": device,
        "obj_id": 1,
        "mask_threshold": int(params.mask_threshold),
        "offload_video_to_cpu": bool(params.offload_video_to_cpu),
        "offload_state_to_cpu": bool(params.offload_state_to_cpu),
        "bf16": bool(params.bf16),
        "async_loading_frames": False,
        "compile_image_encoder": False,
        "use_tensorrt": False,
    }


def _worker_env(repo_root):
    """A minimal environment for the child.

    PYTHONPATH is cleared deliberately: the UI process has the repo root and
    demo/ on sys.path, and inheriting that would let our modules shadow the
    ones the SAM2Matting checkout expects to import.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["VIDEOMAMA_ROOT"] = str(repo_root)
    return env


def run_window(repo_root, job, python_bin, log=None, popen=None):
    """Run the worker over one staged window, yielding (local_index, alpha).

    Alphas are yielded as float32 arrays at the staged resolution and their .npy
    files are deleted as they are consumed, so a long window never accumulates
    a second full copy of the matte on disk.
    """
    log = log or (lambda line: None)
    # Resolved at call time, not bound as a default, so tests can substitute a
    # fake process by patching subprocess.Popen on this module.
    popen = popen or subprocess.Popen
    job_dir = Path(job["alpha_dir"]).parent
    job_path = job_dir / f"job.{uuid.uuid4().hex}.json"
    with job_path.open("w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2, sort_keys=True)

    command = [str(python_bin), str(WORKER_SCRIPT), "--job", str(job_path)]
    log(f"Launching SAM2Matting worker: {' '.join(command)}")
    process = popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_env(repo_root),
        cwd=str(Path(job["sam2matting_home"])),
        text=True,
        bufsize=1,
    )

    stderr_tail = []

    def drain_stderr():
        for line in process.stderr:
            line = line.rstrip()
            if not line:
                continue
            stderr_tail.append(line)
            del stderr_tail[:-80]
            log(line)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    error = None
    ready = None
    done = None
    try:
        for raw_line in process.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                log(raw_line)
                continue
            kind = message.get("type")
            if kind == "ready":
                ready = message
                log(
                    f"SAM2Matting ready: {message.get('frames')} frames at "
                    f"{message.get('video_size')}, model {message.get('model_resolution')}px, "
                    f"bf16={message.get('bf16')}, offload_video={message.get('offload_video_to_cpu')}, "
                    f"offload_state={message.get('offload_state_to_cpu')}"
                )
            elif kind == "cond":
                log(f"SAM2Matting conditioning frames (window-local): {message.get('frames')}")
            elif kind == "frame":
                path = Path(message["path"])
                alpha = np.load(path).astype(np.float32, copy=False)
                path.unlink(missing_ok=True)
                yield int(message["index"]), alpha, message
            elif kind == "done":
                done = message
            elif kind == "error":
                error = message
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass
        returncode = process.wait()
        stderr_thread.join(timeout=5)
        job_path.unlink(missing_ok=True)

    if error is not None:
        raise SAM2MattingError(
            error.get("message", "SAM2Matting worker failed."),
            kind=error.get("kind", "runtime"),
            detail=error.get("traceback", ""),
        )
    if returncode != 0:
        raise SAM2MattingError(
            f"SAM2Matting worker exited with code {returncode}.",
            kind="runtime",
            detail="\n".join(stderr_tail[-40:]),
        )
    if done is None:
        raise SAM2MattingError(
            "SAM2Matting worker finished without reporting completion.",
            kind="runtime",
            detail="\n".join(stderr_tail[-40:]),
        )
    if ready is None:
        raise SAM2MattingError("SAM2Matting worker never reported readiness.", kind="runtime")


def cleanup_window(work_dir):
    shutil.rmtree(Path(work_dir), ignore_errors=True)


def plan_windows(frame_indices, window_size, window_overlap):
    """Split a range into overlapping windows of SAM2Matting propagation.

    Each window is one predictor state. Overlap exists so the boundary between
    two states can be cross-faded rather than cut, and is clamped to half a
    window so any frame belongs to at most two windows.
    """
    indices = [int(i) for i in frame_indices]
    if not indices:
        return []
    size = int(window_size or 0)
    if size <= 0 or size >= len(indices):
        return [list(indices)]
    overlap = max(0, int(window_overlap or 0))
    overlap = min(overlap, size // 2)
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(indices), step):
        window = indices[start:start + size]
        if not window:
            break
        windows.append(window)
        if start + size >= len(indices):
            break
    return windows
