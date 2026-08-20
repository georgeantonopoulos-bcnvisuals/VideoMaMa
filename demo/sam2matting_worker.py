#!/usr/bin/env python
"""
SAM2Matting worker — runs ONLY inside the isolated SAM2Matting venv.

Do not import this module from the production Gradio app. SAM2Matting vendors
modified top-level `sam2` and `sam3` packages and expects its own Torch build;
importing it into the SAM 3 UI process would shadow our official SAM 3 install.
The UI drives this file as a subprocess instead (see demo/sam2matting_client.py).

Protocol
--------
in:   a JSON job file, path given as --job
out:  NDJSON on stdout, one object per line:
        {"type": "ready",  "video_size": [w, h], "frames": n, ...}
        {"type": "cond",   "frames": [...]}
        {"type": "frame",  "index": i, "path": "...npy", "min": f, "max": f, "mean": f}
        {"type": "done",   "written": n, "peak_vram_bytes": n}
        {"type": "error",  "message": "...", "kind": "oom|runtime", "traceback": "..."}
      diagnostics go to stderr and are teed into the UI debug console.
alpha: one float32 .npy per frame in the job's alpha_dir, at the frame-cache
      resolution. Float, never quantized here — the caller owns the final format.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _log(message):
    sys.stderr.write(f"[sam2matting-worker] {message}\n")
    sys.stderr.flush()


def _prepare_sys_path(job):
    """Put the pinned SAM2Matting checkout first on sys.path.

    Upstream's sam2/build_sam.py refuses to run when a `sam2/sam2` directory is
    visible, and its `sam2/__init__.py` registers a Hydra config module, so the
    checkout must be importable as a plain top-level package from its own root.
    """
    home = job.get("sam2matting_home")
    if not home:
        raise RuntimeError("Job is missing sam2matting_home.")
    home = str(Path(home).resolve())
    if not Path(home, "sam2").is_dir():
        raise RuntimeError(f"SAM2Matting checkout looks wrong (no sam2/ in {home}).")
    if home in sys.path:
        sys.path.remove(home)
    sys.path.insert(0, home)
    # Hydra resolves the sam2 config module relative to the process CWD in some
    # code paths; anchor to the checkout so `configs/...` names always resolve.
    os.chdir(home)
    return home


def _load_conditioning_mask(path, threshold, target_size, torch, np):
    """Read a binary SAM 3 mask and encode it the way the predictor expects.

    add_new_mask() re-binarizes whatever it receives, so what matters is only
    that positive pixels land above the threshold it uses. We hand it logits at
    the model's own mask resolution (matching upstream's demo encoding) rather
    than a 0/1 image, which keeps the behaviour identical whether or not the
    predictor takes its interpolate-and-threshold branch.
    """
    from PIL import Image

    with Image.open(path) as image:
        mask = np.array(image.convert("L"))
    binary = (mask > int(threshold)).astype("float32")
    tensor = torch.from_numpy(binary)[None, None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(int(target_size), int(target_size)), mode="bilinear", align_corners=False
    )
    # >0.5 after bilinear keeps thin structures that a nearest resample drops.
    return (tensor > 0.5).float() * 20.0 - 10.0


def _build_sam2_predictor(job, torch):
    from sam2.build_sam import build_sam2matting_video_predictor

    overrides = []
    if job.get("compile_image_encoder"):
        overrides.append("++model.compile_image_encoder=True")
    predictor = build_sam2matting_video_predictor(
        job["config_name"],
        job["checkpoint_path"],
        device=job.get("device", "cuda"),
        hydra_overrides_extra=overrides,
    )
    if job.get("compile_image_encoder") and job.get("use_tensorrt"):
        from sam2.utils.trt import replace_unknown_alpha_predictor_with_trt

        predictor = replace_unknown_alpha_predictor_with_trt(predictor)
    return predictor


def _build_sam3_predictor(job, torch):
    """Build SAM2Matting's own vendored SAM 3 tracker variant.

    Unrelated to our official facebookresearch/sam3 install: every weight comes
    out of the SAM2Matting-SAM3.pt checkpoint, so no gated download is involved.
    """
    from iopath.common.file_io import g_pathmgr
    from sam3.model.sam3matting_video_predictor import build_sam3matting_video_predictor

    with g_pathmgr.open(job["checkpoint_path"], "rb") as handle:
        ckpt = torch.load(handle, map_location="cpu", weights_only=True)
    state_dict = {}
    for key, value in ckpt["model"].items():
        if key.startswith("detector.backbone.vision_backbone."):
            state_dict[key[len("detector."):]] = value
        elif key.startswith("tracker."):
            state_dict[key[len("tracker."):]] = value
    predictor = build_sam3matting_video_predictor(checkpoint=None, device=job.get("device", "cuda"))
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    _log(f"sam3 variant load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    return predictor


def _init_state(predictor, job):
    """Call init_state with only the offload kwargs this predictor accepts."""
    import inspect

    kwargs = {"video_path": job["frames_dir"]}
    signature = inspect.signature(predictor.init_state)
    for name, value in (
        ("offload_video_to_cpu", bool(job.get("offload_video_to_cpu", True))),
        ("offload_state_to_cpu", bool(job.get("offload_state_to_cpu", True))),
        ("async_loading_frames", bool(job.get("async_loading_frames", False))),
    ):
        if name in signature.parameters:
            kwargs[name] = value
        elif value:
            _log(f"predictor.init_state does not accept {name}; ignoring")
    return predictor.init_state(**kwargs)


class _DeviceFrameView:
    """Hands out one frame at a time on the compute device, keeping the rest on CPU.

    Works around a real upstream bug: SAM2Matting's _run_single_frame_inference
    reads ``inference_state["images"][frame_idx]`` and feeds it straight to the
    alpha heads without moving it to the compute device, unlike every other
    reader, which calls ``.to(device)``. With ``offload_video_to_cpu=True`` the
    alpha head then tries to concatenate a CPU image with CUDA features and
    raises a device mismatch.

    Disabling CPU offload would "fix" it by keeping every decoded frame resident
    on the GPU, which is exactly what makes a long shot OOM on a 24 GB L4. This
    keeps the offload and materializes a single frame per access instead.

    Only ``__getitem__`` and ``__len__`` are needed: those are the only ways the
    predictors touch this tensor (verified against sam2_video_predictor.py,
    sam2matting_video_predictor.py and sam3_tracking_predictor.py).
    """

    def __init__(self, images, device):
        self._images = images
        self._device = device

    def __getitem__(self, index):
        return self._images[index].to(self._device, non_blocking=True)

    def __len__(self):
        return len(self._images)

    @property
    def shape(self):
        return self._images.shape


def _apply_offload_device_shim(inference_state, log=_log):
    """Wrap the offloaded video tensor so per-frame reads land on the right device."""
    images = inference_state.get("images")
    device = inference_state.get("device")
    if images is None or device is None:
        return False
    if getattr(images, "device", device) == device:
        return False  # not offloaded; upstream's own .to(device) calls suffice
    inference_state["images"] = _DeviceFrameView(images, device)
    log(f"applied CPU-offload device shim (video stays on {images.device}, frames served on {device})")
    return True


def _prune_alpha_cache(inference_state, frame_idx, is_conditioning):
    """Drop the stored per-frame alpha once it has been written out.

    Upstream keeps `alpha` at full video resolution inside the per-frame output
    dict and, unlike maskmem, never moves it to the storage device. Left alone
    that is roughly 4 MB/frame at 1024x1024 on the GPU and grows without bound,
    which is what turns a long shot into an OOM. Conditioning frames are kept
    because propagate_in_video re-reads their stored alpha.
    """
    if is_conditioning or os.environ.get("SAM2MATTING_DISABLE_ALPHA_PRUNE") == "1":
        return
    for per_obj in (inference_state.get("output_dict_per_obj") or {}).values():
        entry = (per_obj.get("non_cond_frame_outputs") or {}).get(frame_idx)
        if entry is not None:
            entry["alpha"] = None
    # SAM 3 variant keeps a single flat output dict instead of a per-object one.
    flat = inference_state.get("output_dict")
    if isinstance(flat, dict):
        entry = (flat.get("non_cond_frame_outputs") or {}).get(frame_idx)
        if isinstance(entry, dict) and "alpha" in entry:
            entry["alpha"] = None


def _alpha_to_numpy(alpha, np):
    """Normalize the predictor's alpha to a 2-D float32 array."""
    if hasattr(alpha, "detach"):
        alpha = alpha.detach().float().cpu().numpy()
    array = np.asarray(alpha, dtype="float32")
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2-D alpha, got shape {array.shape}.")
    return np.clip(array, 0.0, 1.0)


def run_job(job):
    _prepare_sys_path(job)

    import numpy as np
    import torch

    # The worker decides this, not the caller. The UI process runs in a different
    # environment whose torch may not even be installed, so a CUDA check there
    # says nothing about this process and must never silently downgrade to CPU.
    requested = str(job.get("device") or "auto")
    if requested == "cpu":
        device = "cpu"
        _log("device=cpu was requested explicitly; inference will be very slow")
    elif torch.cuda.is_available():
        device = "cuda"
    elif requested == "cuda":
        raise RuntimeError("SAM2Matting requires a CUDA GPU; none is visible to this worker.")
    else:
        device = "cpu"
        _log("no CUDA GPU visible to the worker; falling back to CPU (very slow)")
    job = dict(job, device=device)

    frames_dir = Path(job["frames_dir"])
    alpha_dir = Path(job["alpha_dir"])
    alpha_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(sorted(frames_dir.glob("*.jpg"))) or len(sorted(frames_dir.glob("*.png")))
    if frame_count == 0:
        raise RuntimeError(f"No frames to process in {frames_dir}.")

    variant = job.get("variant", "sam2.1base+")
    _log(f"building predictor for variant={variant} on {device}")
    if variant == "sam3":
        predictor = _build_sam3_predictor(job, torch)
        mask_size = 288
    else:
        predictor = _build_sam2_predictor(job, torch)
        mask_size = 256
    mask_size = int(job.get("cond_mask_size") or mask_size)

    autocast_dtype = torch.bfloat16 if job.get("bf16", True) else torch.float32
    obj_id = int(job.get("obj_id", 1))
    masks = {int(k): v for k, v in (job.get("masks") or {}).items()}
    if not masks:
        raise RuntimeError("At least one conditioning mask is required.")
    for frame_idx in masks:
        if not 0 <= frame_idx < frame_count:
            raise RuntimeError(
                f"Conditioning frame {frame_idx} is outside the {frame_count}-frame window."
            )

    written = set()

    def write_alpha(frame_idx, alpha):
        array = _alpha_to_numpy(alpha, np)
        path = alpha_dir / f"{frame_idx:06d}.npy"
        # Write through a file object: np.save() appends ".npy" to a *path* that
        # does not already end in it, which would silently defeat the rename.
        temp_path = alpha_dir / f".{frame_idx:06d}.partial.npy"
        with temp_path.open("wb") as handle:
            np.save(handle, array)
        os.replace(temp_path, path)
        written.add(int(frame_idx))
        _emit({
            "type": "frame",
            "index": int(frame_idx),
            "path": str(path),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
        })

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode(), torch.autocast(
        device_type="cuda" if device == "cuda" else "cpu", dtype=autocast_dtype,
        enabled=bool(job.get("bf16", True)) and device == "cuda",
    ):
        inference_state = _init_state(predictor, job)
        predictor.reset_state(inference_state)
        _apply_offload_device_shim(inference_state)
        _log(
            "state: images device="
            f"{getattr(inference_state.get('images'), 'device', 'n/a')}, "
            f"storage={inference_state.get('storage_device')}, "
            f"compute={inference_state.get('device')}"
        )
        video_height = int(inference_state["video_height"])
        video_width = int(inference_state["video_width"])
        _emit({
            "type": "ready",
            "frames": int(frame_count),
            "video_size": [video_width, video_height],
            "variant": variant,
            "model_resolution": int(getattr(predictor, "image_size", 0)),
            "offload_video_to_cpu": bool(job.get("offload_video_to_cpu", True)),
            "offload_state_to_cpu": bool(job.get("offload_state_to_cpu", True)),
            "bf16": bool(job.get("bf16", True)),
        })

        threshold = int(job.get("mask_threshold", 127))
        for frame_idx in sorted(masks):
            mask_tensor = _load_conditioning_mask(masks[frame_idx], threshold, mask_size, torch, np)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=obj_id,
                mask=mask_tensor.to(device),
            )
        _emit({"type": "cond", "frames": sorted(int(i) for i in masks)})

        earliest = min(masks)

        def consume(stream, conditioning_frames):
            for output in stream:
                frame_idx = int(output[0])
                alpha = output[3]
                if frame_idx not in written:
                    write_alpha(frame_idx, alpha)
                del alpha
                _prune_alpha_cache(inference_state, frame_idx, frame_idx in conditioning_frames)

        cond_set = set(int(i) for i in masks)
        # SAM2Matting propagates in one direction per call, unlike our SAM 3
        # tracker. Run backwards from the earliest conditioning frame first so
        # frames before it are covered, then forwards over the rest.
        if earliest > 0:
            _log(f"reverse pass from frame {earliest}")
            consume(
                predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=earliest,
                    max_frame_num_to_track=earliest,
                    reverse=True,
                ),
                cond_set,
            )
        _log(f"forward pass from frame {earliest}")
        consume(
            predictor.propagate_in_video(inference_state, start_frame_idx=earliest, reverse=False),
            cond_set,
        )

    missing = sorted(set(range(frame_count)) - written)
    if missing:
        raise RuntimeError(
            f"SAM2Matting did not produce alpha for {len(missing)} frame(s): "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
        )

    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    _emit({"type": "done", "written": len(written), "peak_vram_bytes": peak})


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAM2Matting isolated worker")
    parser.add_argument("--job", required=True, help="path to the JSON job file")
    args = parser.parse_args(argv)

    with open(args.job, "r", encoding="utf-8") as handle:
        job = json.load(handle)

    try:
        run_job(job)
    except BaseException as exc:  # noqa: BLE001 - the parent needs every failure mode
        kind = "runtime"
        name = type(exc).__name__
        if "OutOfMemory" in name or "CUDA out of memory" in str(exc):
            kind = "oom"
        _emit({
            "type": "error",
            "kind": kind,
            "message": f"{name}: {exc}",
            "traceback": traceback.format_exc(),
        })
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
