"""
SAM 3 wrapper for production sequence mask tracking.

This adapter preserves the small interface used by production_frame_app.py:
preview a prompted frame, and propagate prompted keyframes through a JPEG
sequence directory.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


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
            "`/tmp/videomama-sam3-ui-venv/bin/hf auth login`."
        ) from exc
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise RuntimeError(
                f"SAM 3 checkpoint repo `{repo_id}` requires Hugging Face authentication. "
                "Run `/tmp/videomama-sam3-ui-venv/bin/hf auth login` with a token that has access."
            ) from exc
        raise
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"Could not reach or cache SAM 3 checkpoint repo `{repo_id}`. Check network access to "
            "huggingface.co and authenticate with `/tmp/videomama-sam3-ui-venv/bin/hf auth login`."
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
                "`/tmp/videomama-sam3-ui-venv/bin/python -m pip install -r scripts/requirements-sam3-ui.txt`."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 failed to import. Refresh the production UI environment with "
                "`/tmp/videomama-sam3-ui-venv/bin/python -m pip install -r scripts/requirements-sam3-ui.txt`."
            ) from exc

        self.device = device
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
        if gpus_to_use is not None:
            kwargs["gpus_to_use"] = gpus_to_use
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

    def _extract_mask(self, outputs, frame_shape, obj_id: int = 1):
        height, width = frame_shape
        if not outputs:
            return np.zeros((height, width), dtype=np.uint8)

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
        return self.predictor.handle_request(
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
    ) -> List[np.ndarray]:
        prompts_by_frame = self._normalize_prompts_by_frame(prompts_by_frame)
        height, width, num_frames = self._frame_shape_from_dir(frames_dir)
        keyframes = sorted(prompts_by_frame)
        if keyframes[0] < 0 or keyframes[-1] >= num_frames:
            raise ValueError(f"Prompt frame indices must be within [0, {num_frames - 1}].")

        session_id = self._start_session(frames_dir)
        try:
            latest_outputs = {}
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
                latest_outputs[frame_idx] = response.get("outputs", {})

            final_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]
            for frame_idx, outputs in latest_outputs.items():
                final_masks[frame_idx] = self._extract_mask(outputs, (height, width), obj_id=obj_id)

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
                    final_masks[frame_idx] = self._extract_mask(response.get("outputs", {}), (height, width), obj_id=obj_id)
        finally:
            self._close_session(session_id)

        frames_with_pixels = sum(1 for mask in final_masks if np.any(mask))
        total_pixels = int(sum((mask > 0).sum() for mask in final_masks))
        print(
            f"Generated {len(final_masks)} SAM 3 masks from {len(keyframes)} keyframes; "
            f"frames_with_pixels={frames_with_pixels}/{num_frames}, total_pixels={total_pixels}, "
            f"output_prob_thresh={float(output_prob_thresh):.3f}"
        )
        return final_masks

    def track_video_from_dir_with_text(
        self,
        frames_dir: str,
        text: str,
        frame_idx: int = 0,
        output_prob_thresh: float = 0.5,
    ) -> List[np.ndarray]:
        height, width, num_frames = self._frame_shape_from_dir(frames_dir)
        frame_idx = max(0, min(int(frame_idx), num_frames - 1))

        session_id = self._start_session(frames_dir)
        try:
            response = self._add_text_prompt(session_id, frame_idx, text, output_prob_thresh=output_prob_thresh)
            final_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]
            final_masks[frame_idx] = self._extract_combined_mask(response.get("outputs", {}), (height, width))

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
                    final_masks[out_frame_idx] = self._extract_combined_mask(
                        response.get("outputs", {}), (height, width)
                    )
        finally:
            self._close_session(session_id)

        frames_with_pixels = sum(1 for mask in final_masks if np.any(mask))
        total_pixels = int(sum((mask > 0).sum() for mask in final_masks))
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
                return self._extract_mask(response.get("outputs", {}), (height, width), obj_id=1)
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
