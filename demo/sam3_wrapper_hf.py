"""
SAM 3 wrapper for production sequence mask tracking.

This adapter preserves the small interface used by production_frame_app.py:
preview a prompted frame, and propagate prompted keyframes through a JPEG
sequence directory.
"""

import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image


def _ui_venv_path() -> str:
    """Where this UI environment lives, for actionable error messages."""
    venv = os.environ.get("VIDEOMAMA_UI_VENV")
    if venv:
        return venv
    root = os.environ.get("VIDEOMAMA_VENV_ROOT", "/mnt/temporal/VideoMama")
    return f"{root}/videomama-sam3-ui-venv"


def _assert_hf_sam3_access(repo_id: str):
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for SAM 3 checkpoint access. Refresh the SAM 3 UI environment."
        ) from exc

    try:
        hf_hub_download(repo_id=repo_id, filename="config.json")
    except GatedRepoError as exc:
        raise RuntimeError(
            f"SAM 3 checkpoint repo `{repo_id}` is gated. Request access at "
            f"https://huggingface.co/{repo_id}, then authenticate this machine with "
            f"`{_ui_venv_path()}/bin/hf auth login`."
        ) from exc
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise RuntimeError(
                f"SAM 3 checkpoint repo `{repo_id}` requires Hugging Face authentication. "
                f"Run `{_ui_venv_path()}/bin/hf auth login` with a token that has access."
            ) from exc
        raise
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"Could not reach or cache SAM 3 checkpoint repo `{repo_id}`. Check network access to "
            f"huggingface.co and authenticate with `{_ui_venv_path()}/bin/hf auth login`."
        ) from exc


class SAM3VideoTracker:
    def __init__(self, device="cuda", gpus_to_use: Optional[List[int]] = None, model_version="sam3"):
        try:
            from sam3.model_builder import build_sam3_predictor
        except ModuleNotFoundError as exc:
            missing_module = exc.name or "unknown"
            if missing_module == "sam3":
                raise RuntimeError(
                    "SAM 3 is not installed. Build the production UI environment with "
                    "`bash scripts/bootstrap_tmp_venv.sh sam3-ui`."
                ) from exc
            raise RuntimeError(
                f"SAM 3 dependency import failed because `{missing_module}` is missing. "
                "Refresh the SAM 3 UI environment with "
                f"`{_ui_venv_path()}/bin/python -m pip install -r scripts/requirements-sam3-ui.txt`."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 failed to import. Refresh the production UI environment with "
                f"`{_ui_venv_path()}/bin/python -m pip install -r scripts/requirements-sam3-ui.txt`."
            ) from exc

        self.device = device
        self.model_version = str(model_version)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("SAM 3 requires PyTorch in the production UI environment.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3 video prediction requires a CUDA GPU; none is visible to PyTorch.")

        if gpus_to_use is None:
            gpus_to_use = [torch.cuda.current_device()]

        repo_id = "facebook/sam3.1" if model_version == "sam3.1" else "facebook/sam3"
        _assert_hf_sam3_access(repo_id)

        kwargs = {}
        # The legacy SAM 3 video predictor supports explicit multi-GPU IDs.
        # SAM 3.1's multiplex predictor is single-device and rejects this kwarg.
        if model_version == "sam3" and gpus_to_use is not None:
            kwargs["gpus_to_use"] = gpus_to_use
        if model_version == "sam3.1":
            # FA3 is an optional Hopper-only extension and is not installed in
            # the production venv. PyTorch attention is slower but equivalent.
            kwargs["use_fa3"] = False
        kwargs["version"] = model_version
        self.predictor = build_sam3_predictor(**kwargs)
        self.last_text_prompt_stats = {}
        print(f"SAM 3 video predictor initialized on {device}")

    def _normalize_prompts_by_frame(self, prompts_by_frame: Dict[int, Dict[str, List[List[int]]]]):
        normalized = {}
        for raw_frame_idx, prompt_data in (prompts_by_frame or {}).items():
            if prompt_data is None:
                continue
            frame_idx = int(raw_frame_idx)
            points = prompt_data.get("points", [])
            labels = prompt_data.get("labels", [])
            if not points and not labels:
                continue
            if len(points) != len(labels):
                raise ValueError(f"Frame {frame_idx} has mismatched points and labels.")
            normalized[frame_idx] = {
                "points": [[float(x), float(y)] for x, y in points],
                "labels": [int(label) for label in labels],
            }
        if not normalized:
            raise ValueError("At least one prompted frame is required.")
        return dict(sorted(normalized.items()))

    def _frame_shape_from_dir(self, frames_dir: str):
        frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
        if not frame_paths:
            raise ValueError(f"No JPEG frames found in {frames_dir}")
        with Image.open(frame_paths[0]) as image:
            width, height = image.size
        return height, width, len(frame_paths)

    def _relative_points(self, points, width, height):
        return [[x / width, y / height] for x, y in points]

    @staticmethod
    def _mask_from_logits(mask_logits, frame_shape, mask_threshold: float) -> np.ndarray:
        """Resize raw SAM mask logits and apply a real probability threshold."""
        height, width = frame_shape
        threshold = max(0.01, min(float(mask_threshold), 0.99))
        logit_threshold = math.log(threshold / (1.0 - threshold))

        if hasattr(mask_logits, "detach"):
            import torch
            import torch.nn.functional as torch_functional

            logits = mask_logits.detach().float()
            while logits.ndim > 2 and logits.shape[0] == 1:
                logits = logits[0]
            if logits.ndim != 2:
                logits = logits.squeeze()
            if logits.ndim != 2:
                raise ValueError(f"Unexpected SAM mask-logit shape: {tuple(logits.shape)}")
            if tuple(logits.shape) != (height, width):
                logits = torch_functional.interpolate(
                    logits[None, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            return ((logits >= logit_threshold).to(torch.uint8).cpu().numpy() * 255)

        logits = np.asarray(mask_logits, dtype=np.float32)
        while logits.ndim > 2 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 2 and logits.size == height * width:
            logits = logits.reshape(height, width)
        if logits.ndim != 2:
            raise ValueError(f"Unexpected SAM mask-logit shape: {logits.shape}")
        if logits.shape != (height, width):
            logits = np.asarray(
                Image.fromarray(logits, mode="F").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        return ((logits >= logit_threshold).astype(np.uint8) * 255)

    def _point_mask_logits(self, session_id: str, frame_idx: int, obj_id: int):
        """Read the raw point-mask logits retained by SAM's interactive tracker."""
        sessions = getattr(self.predictor, "_all_inference_states", {})
        session = sessions.get(session_id)
        if session is None and hasattr(self.predictor, "_get_session"):
            session = self.predictor._get_session(session_id)
        inference_state = session.get("state", {}) if isinstance(session, dict) else {}

        tracker_states = list(inference_state.get("tracker_inference_states") or [])
        tracker_states.extend(inference_state.get("sam2_inference_states") or [])
        for tracker_state in tracker_states:
            obj_id_to_idx = tracker_state.get("obj_id_to_idx") or {}
            if obj_id not in obj_id_to_idx:
                continue
            obj_idx = int(obj_id_to_idx[obj_id])
            for outputs_by_kind in (
                tracker_state.get("temp_output_dict_per_obj", {}).get(obj_idx, {}),
                tracker_state.get("output_dict_per_obj", {}).get(obj_idx, {}),
            ):
                for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    frame_output = outputs_by_kind.get(storage_key, {}).get(int(frame_idx))
                    if not frame_output:
                        continue
                    logits = frame_output.get("pred_masks_video_res")
                    if logits is None:
                        logits = frame_output.get("pred_masks")
                    if logits is not None:
                        return logits
        return None

    def _attach_point_mask_logits(self, session_id: str, frame_idx: int, obj_id: int, response):
        """Add private raw logits to an in-process response for thresholding."""
        logits = self._point_mask_logits(session_id, frame_idx, obj_id)
        outputs = response.get("outputs") if isinstance(response, dict) else None
        if logits is not None and isinstance(outputs, dict):
            outputs["_point_mask_logits"] = logits
            outputs["_point_mask_logits_obj_id"] = int(obj_id)
        return response

    def _extract_mask(
        self,
        outputs,
        frame_shape,
        obj_id: int = 1,
        mask_threshold: float = 0.5,
    ):
        height, width = frame_shape
        if not outputs:
            return np.zeros((height, width), dtype=np.uint8)

        mask_logits = outputs.get("_point_mask_logits")
        logits_obj_id = int(outputs.get("_point_mask_logits_obj_id", obj_id))
        if mask_logits is not None and logits_obj_id == int(obj_id):
            return self._mask_from_logits(mask_logits, frame_shape, mask_threshold)

        object_ids = outputs.get("out_obj_ids")
        masks = outputs.get("out_binary_masks")
        if object_ids is None or masks is None or len(object_ids) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        object_ids = object_ids.tolist() if hasattr(object_ids, "tolist") else list(object_ids)
        mask_index = object_ids.index(obj_id) if obj_id in object_ids else 0
        mask = masks[mask_index]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).squeeze()
        if mask.shape != (height, width):
            mask = np.array(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST))
            return mask.astype(np.uint8)
        return ((mask > 0) * 255).astype(np.uint8)

    def _extract_combined_mask(self, outputs, frame_shape):
        height, width = frame_shape
        if not outputs:
            return np.zeros((height, width), dtype=np.uint8)

        masks = outputs.get("out_binary_masks")
        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        combined = np.zeros((height, width), dtype=bool)
        for mask in masks:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask).squeeze()
            if mask.shape != (height, width):
                mask = np.array(
                    Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                )
            combined |= mask > 0
        return (combined.astype(np.uint8) * 255)

    def _output_mask_stats(self, outputs):
        if not outputs:
            return {"mask_count": 0, "areas": [], "keys": []}
        masks = outputs.get("out_binary_masks")
        if masks is None:
            return {"mask_count": 0, "areas": [], "keys": sorted(outputs.keys())}
        areas = []
        for mask in masks:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask).squeeze()
            areas.append(int((mask > 0).sum()))
        return {
            "mask_count": len(areas),
            "areas": areas,
            "keys": sorted(outputs.keys()),
        }

    def _start_session(self, frames_dir: str):
        if self.model_version == "sam3.1":
            # The upstream base predictor currently forwards
            # ``offload_state_to_cpu`` to the multiplex model, whose init_state
            # signature does not accept it. Initialize and register the session
            # directly until the upstream predictor signatures converge.
            init_kwargs = {
                "resource_path": str(frames_dir),
                "offload_video_to_cpu": False,
            }
            if hasattr(self.predictor, "async_loading_frames"):
                init_kwargs["async_loading_frames"] = self.predictor.async_loading_frames
            if hasattr(self.predictor, "video_loader_type"):
                init_kwargs["video_loader_type"] = self.predictor.video_loader_type
            inference_state = self.predictor.model.init_state(**init_kwargs)
            session_id = str(uuid.uuid4())
            now = time.time()
            self.predictor._all_inference_states[session_id] = {
                "state": inference_state,
                "session_id": session_id,
                "start_time": now,
                "last_use_time": now,
            }
            return session_id
        response = self.predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(frames_dir),
            }
        )
        return response["session_id"]

    def _close_session(self, session_id: str):
        try:
            self.predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )
        except Exception as exc:
            print(f"Warning: failed to close SAM 3 session {session_id}: {exc}")

    def _add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        prompt_data,
        width: int,
        height: int,
        obj_id: int,
        output_prob_thresh: float = 0.5,
    ):
        import torch

        points = self._relative_points(prompt_data["points"], width, height)
        labels = [int(label) for label in prompt_data["labels"]]
        response = self.predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(frame_idx),
                "points": torch.tensor(points, dtype=torch.float32),
                "point_labels": torch.tensor(labels, dtype=torch.int32),
                "obj_id": int(obj_id),
                "output_prob_thresh": float(output_prob_thresh),
            }
        )
        return self._attach_point_mask_logits(session_id, frame_idx, obj_id, response)

    def _add_text_prompt(self, session_id: str, frame_idx: int, text: str, output_prob_thresh: float = 0.5):
        text = str(text or "").strip()
        if not text:
            raise ValueError("A non-empty text prompt is required.")
        response = self.predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(frame_idx),
                "text": text,
                "output_prob_thresh": float(output_prob_thresh),
            }
        )
        stats = self._output_mask_stats(response.get("outputs", {}))
        self.last_text_prompt_stats = {
            "text": text,
            "frame_index": int(frame_idx),
            "output_prob_thresh": float(output_prob_thresh),
            **stats,
        }
        print(
            "SAM 3 text prompt "
            f"`{text}` on frame {frame_idx}: {stats['mask_count']} mask(s), "
            f"areas={stats['areas'][:8]}, keys={stats['keys']}"
        )
        return response

    def track_video_from_dir(
        self,
        frames_dir: str,
        prompts_by_frame,
        obj_id: int = 1,
        output_prob_thresh: float = 0.5,
        mask_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        collect_masks: bool = True,
    ) -> List[np.ndarray]:
        prompts_by_frame = self._normalize_prompts_by_frame(prompts_by_frame)
        height, width, num_frames = self._frame_shape_from_dir(frames_dir)
        keyframes = sorted(prompts_by_frame)
        if keyframes[0] < 0 or keyframes[-1] >= num_frames:
            raise ValueError(f"Prompt frame indices must be within [0, {num_frames - 1}].")

        session_id = self._start_session(frames_dir)
        try:
            keyframe_masks = {}
            for frame_idx, prompt_data in prompts_by_frame.items():
                response = self._add_prompt(
                    session_id,
                    frame_idx,
                    prompt_data,
                    width,
                    height,
                    obj_id=obj_id,
                    output_prob_thresh=output_prob_thresh,
                )
                # SAM 3 reuses internal output buffers as later corrections are
                # added. Convert and copy immediately; retaining the response's
                # tensors until the loop ends lets later keyframes mutate an
                # earlier artist-approved result.
                keyframe_masks[frame_idx] = self._extract_mask(
                    response.get("outputs", {}),
                    (height, width),
                    obj_id=obj_id,
                    mask_threshold=output_prob_thresh,
                ).copy()

            final_masks = [
                np.zeros((height, width), dtype=np.uint8) if collect_masks else None
                for _ in range(num_frames)
            ]
            mask_areas = {}

            def record_mask(frame_idx, mask):
                mask_areas[int(frame_idx)] = int((mask > 0).sum())
                if collect_masks:
                    final_masks[int(frame_idx)] = mask
                if mask_callback is not None:
                    mask_callback(int(frame_idx), mask)

            # ``add_prompt`` is the authoritative result on an artist-authored
            # keyframe. Keep those masks separate because the propagation stream
            # also emits conditioning frames and can return a different tracked
            # result for them after later corrections have been added.
            for frame_idx, mask in keyframe_masks.items():
                record_mask(frame_idx, mask)

            propagated_keyframe_drift = {}

            # SAM 3's add_prompt path runs the model under a bf16 autocast context,
            # but the library's propagate_in_video (sam3_base_predictor) does not wrap
            # the model forward. Its detector casts the backbone features to bfloat16
            # while the conv layers keep float32 weights, so the propagation forward
            # only has matching dtypes inside an autocast context. Without this the
            # streamed forward raises in conv_s0:
            #   "Input type (c10::BFloat16) and bias type (float) should be the same".
            # The generator runs the model lazily per `next()`, so autocast must stay
            # active across the whole iteration, not just generator creation.
            import torch

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for response in self.predictor.handle_stream_request(
                    request={
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        # Point prompts are stored in the interactive tracker state,
                        # not in SAM 3's VG `previous_stages_out`. If this is left
                        # unset, SAM 3 checks only `previous_stages_out` and raises
                        # "No prompts are received" even though point prompts were
                        # accepted. Start from the earliest prompted keyframe and
                        # let bidirectional propagation cover the shot.
                        "start_frame_index": keyframes[0],
                        "propagation_direction": "both",
                        "output_prob_thresh": float(output_prob_thresh),
                    }
                ):
                    frame_idx = int(response["frame_index"])
                    self._attach_point_mask_logits(session_id, frame_idx, obj_id, response)
                    propagated_mask = self._extract_mask(
                        response.get("outputs", {}),
                        (height, width),
                        obj_id=obj_id,
                        mask_threshold=output_prob_thresh,
                    )
                    if frame_idx in keyframe_masks:
                        exact_mask = keyframe_masks[frame_idx]
                        propagated_keyframe_drift[frame_idx] = int(
                            np.count_nonzero((propagated_mask > 0) != (exact_mask > 0))
                        )
                        continue
                    record_mask(frame_idx, propagated_mask)
        finally:
            self._close_session(session_id)

        frames_with_pixels = sum(1 for area in mask_areas.values() if area > 0)
        total_pixels = int(sum(mask_areas.values()))
        self.last_tracking_stats = {
            "frame_count": int(num_frames),
            "keyframes": keyframes,
            "keyframe_areas": {
                int(frame_idx): int((mask > 0).sum())
                for frame_idx, mask in keyframe_masks.items()
            },
            "propagated_keyframe_drift_pixels": propagated_keyframe_drift,
            "frames_with_pixels": int(frames_with_pixels),
            "total_pixels": total_pixels,
        }
        print(
            f"Generated {len(final_masks)} SAM 3 masks from {len(keyframes)} keyframes; "
            f"frames_with_pixels={frames_with_pixels}/{num_frames}, total_pixels={total_pixels}, "
            f"preserved_keyframes={len(keyframe_masks)}, "
            f"output_prob_thresh={float(output_prob_thresh):.3f}"
        )
        return final_masks

    def track_video_from_dir_with_text(
        self,
        frames_dir: str,
        text: str,
        frame_idx: int = 0,
        output_prob_thresh: float = 0.5,
        mask_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        collect_masks: bool = True,
    ) -> List[np.ndarray]:
        height, width, num_frames = self._frame_shape_from_dir(frames_dir)
        frame_idx = max(0, min(int(frame_idx), num_frames - 1))

        session_id = self._start_session(frames_dir)
        try:
            response = self._add_text_prompt(session_id, frame_idx, text, output_prob_thresh=output_prob_thresh)
            final_masks = [
                np.zeros((height, width), dtype=np.uint8) if collect_masks else None
                for _ in range(num_frames)
            ]
            mask_areas = {}

            def record_mask(out_frame_idx, mask):
                mask_areas[int(out_frame_idx)] = int((mask > 0).sum())
                if collect_masks:
                    final_masks[int(out_frame_idx)] = mask
                if mask_callback is not None:
                    mask_callback(int(out_frame_idx), mask)

            record_mask(frame_idx, self._extract_combined_mask(response.get("outputs", {}), (height, width)))

            import torch

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for response in self.predictor.handle_stream_request(
                    request={
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "start_frame_index": frame_idx,
                        "propagation_direction": "both",
                        "output_prob_thresh": float(output_prob_thresh),
                    }
                ):
                    out_frame_idx = int(response["frame_index"])
                    record_mask(
                        out_frame_idx,
                        self._extract_combined_mask(response.get("outputs", {}), (height, width)),
                    )
        finally:
            self._close_session(session_id)

        frames_with_pixels = sum(1 for area in mask_areas.values() if area > 0)
        total_pixels = int(sum(mask_areas.values()))
        self.last_text_prompt_stats.update(
            {
                "frames_with_pixels": int(frames_with_pixels),
                "total_pixels": total_pixels,
                "num_frames": int(num_frames),
                "output_prob_thresh": float(output_prob_thresh),
            }
        )
        if total_pixels == 0:
            raise ValueError(
                f"SAM 3 returned no mask pixels for text prompt `{text}`. "
                "Try a simpler noun phrase such as `person`, `girl`, `hair`, or `arm`."
            )

        print(
            f"Generated {len(final_masks)} SAM 3 masks from text prompt `{text}` "
            f"with output_prob_thresh={float(output_prob_thresh):.3f}"
        )
        return final_masks

    def track_video_with_keyframes(
        self,
        frames: List[np.ndarray],
        prompts_by_frame,
        obj_id: int = 1,
        output_prob_thresh: float = 0.5,
    ) -> List[np.ndarray]:
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        try:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(frames_dir / f"{i:05d}.jpg", quality=95)
            return self.track_video_from_dir(
                str(frames_dir),
                prompts_by_frame,
                obj_id=obj_id,
                output_prob_thresh=output_prob_thresh,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def track_video(
        self,
        frames: List[np.ndarray],
        points: List[List[int]],
        labels: List[int],
        output_prob_thresh: float = 0.5,
    ) -> List[np.ndarray]:
        return self.track_video_with_keyframes(
            frames,
            {0: {"points": points, "labels": labels}},
            obj_id=1,
            output_prob_thresh=output_prob_thresh,
        )

    def get_frame_mask(
        self,
        frame: np.ndarray,
        points: List[List[int]],
        labels: List[int],
        output_prob_thresh: float = 0.5,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        try:
            Image.fromarray(frame).save(frames_dir / "00000.jpg", quality=95)
            session_id = self._start_session(str(frames_dir))
            try:
                response = self._add_prompt(
                    session_id,
                    0,
                    {"points": points, "labels": labels},
                    width,
                    height,
                    obj_id=1,
                    output_prob_thresh=output_prob_thresh,
                )
                return self._extract_mask(
                    response.get("outputs", {}),
                    (height, width),
                    obj_id=1,
                    mask_threshold=output_prob_thresh,
                )
            finally:
                self._close_session(session_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_frame_mask_from_path(
        self,
        frame_path: str,
        points: List[List[int]],
        labels: List[int],
        output_prob_thresh: float = 0.5,
    ) -> np.ndarray:
        """Preview the exact cached JPEG bytes used by video propagation."""
        source_path = Path(frame_path)
        if not source_path.is_file():
            raise ValueError(f"SAM 3 preview frame does not exist: {source_path}")
        with Image.open(source_path) as image:
            width, height = image.size

        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        try:
            destination = frames_dir / "00000.jpg"
            if source_path.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copy2(source_path, destination)
            else:
                with Image.open(source_path) as image:
                    image.convert("RGB").save(destination, quality=95)
            session_id = self._start_session(str(frames_dir))
            try:
                response = self._add_prompt(
                    session_id,
                    0,
                    {"points": points, "labels": labels},
                    width,
                    height,
                    obj_id=1,
                    output_prob_thresh=output_prob_thresh,
                )
                return self._extract_mask(
                    response.get("outputs", {}),
                    (height, width),
                    obj_id=1,
                    mask_threshold=output_prob_thresh,
                )
            finally:
                self._close_session(session_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_text_frame_mask(self, frame: np.ndarray, text: str, output_prob_thresh: float = 0.5) -> np.ndarray:
        height, width = frame.shape[:2]
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        try:
            Image.fromarray(frame).save(frames_dir / "00000.jpg", quality=95)
            session_id = self._start_session(str(frames_dir))
            try:
                response = self._add_text_prompt(session_id, 0, text, output_prob_thresh=output_prob_thresh)
                return self._extract_combined_mask(response.get("outputs", {}), (height, width))
            finally:
                self._close_session(session_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_first_frame_mask(
        self,
        frame: np.ndarray,
        points: List[List[int]],
        labels: List[int],
        output_prob_thresh: float = 0.5,
    ) -> np.ndarray:
        return self.get_frame_mask(frame, points, labels, output_prob_thresh=output_prob_thresh)


def load_sam3_tracker(device="cuda", model_version="sam3"):
    return SAM3VideoTracker(device=device, model_version=model_version)
