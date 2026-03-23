"""
SAM2 Wrapper for Video Mask Tracking - Hugging Face Space Version
Handles mask generation and propagation through video.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

# Add SAM2 to path if installed
try:
    import sam2  # noqa: F401
except ImportError:
    for path in ("/home/cvlab19/project/samuel/CVPR/sam2", "./sam2"):
        if os.path.exists(path):
            sys.path.append(path)
            break

import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


class SAM2VideoTracker:
    def __init__(self, checkpoint_path, config_file, device="cuda"):
        self.device = device
        self.predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device,
        )
        print(f"SAM2 video tracker initialized on {device}")

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

    def _extract_object_mask(self, object_ids, mask_logits, obj_id: int, frame_shape):
        obj_ids_list = object_ids.tolist() if hasattr(object_ids, "tolist") else list(object_ids)
        if obj_id in obj_ids_list:
            mask_idx = obj_ids_list.index(obj_id)
            mask = (mask_logits[mask_idx] > 0.0).cpu().numpy()
            return (mask.squeeze() * 255).astype(np.uint8)
        height, width = frame_shape
        return np.zeros((height, width), dtype=np.uint8)

    def _save_frames_to_temp_dir(self, frames: List[np.ndarray]) -> Path:
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"{i:05d}.jpg", quality=95)
        return temp_dir

    def _apply_prompts(self, inference_state, prompts_by_frame, obj_id: int):
        for frame_idx, prompt_data in prompts_by_frame.items():
            points_array = np.array(prompt_data["points"], dtype=np.float32)
            labels_array = np.array(prompt_data["labels"], dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_array,
                labels=labels_array,
            )

    def _propagate(self, inference_state, start_frame_idx, max_frame_num_to_track, reverse, obj_id, frame_shape):
        masks = {}
        iterator = self.predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        for frame_idx, object_ids, mask_logits in iterator:
            masks[frame_idx] = self._extract_object_mask(object_ids, mask_logits, obj_id=obj_id, frame_shape=frame_shape)
        return masks

    def track_video_from_dir(self, frames_dir: str, prompts_by_frame, obj_id: int = 1) -> List[np.ndarray]:
        prompts_by_frame = self._normalize_prompts_by_frame(prompts_by_frame)
        inference_state = self.predictor.init_state(video_path=str(frames_dir))
        self.predictor.reset_state(inference_state)

        num_frames = int(inference_state["num_frames"])
        frame_shape = (int(inference_state["video_height"]), int(inference_state["video_width"]))
        keyframes = sorted(prompts_by_frame)

        if keyframes[0] < 0 or keyframes[-1] >= num_frames:
            raise ValueError(f"Prompt frame indices must be within [0, {num_frames - 1}].")

        final_masks = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(num_frames)]

        def run_segment(segment_prompt_frames, start_frame_idx, max_track, reverse):
            self.predictor.reset_state(inference_state)
            segment_prompts = {frame_idx: prompts_by_frame[frame_idx] for frame_idx in segment_prompt_frames}
            self._apply_prompts(inference_state, segment_prompts, obj_id=obj_id)
            return self._propagate(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_track,
                reverse=reverse,
                obj_id=obj_id,
                frame_shape=frame_shape,
            )

        first_keyframe = keyframes[0]
        if first_keyframe > 0:
            reverse_masks = run_segment([first_keyframe], first_keyframe, first_keyframe, True)
            for frame_idx, mask in reverse_masks.items():
                final_masks[frame_idx] = mask

        for left_keyframe, right_keyframe in zip(keyframes, keyframes[1:]):
            segment_length = right_keyframe - left_keyframe
            forward_masks = run_segment([left_keyframe, right_keyframe], left_keyframe, segment_length, False)
            reverse_masks = run_segment([left_keyframe, right_keyframe], right_keyframe, segment_length, True)

            for frame_idx in range(left_keyframe, right_keyframe + 1):
                if frame_idx == left_keyframe:
                    chosen_mask = forward_masks.get(frame_idx)
                    if chosen_mask is None:
                        chosen_mask = reverse_masks.get(frame_idx)
                elif frame_idx == right_keyframe:
                    chosen_mask = reverse_masks.get(frame_idx)
                    if chosen_mask is None:
                        chosen_mask = forward_masks.get(frame_idx)
                else:
                    dist_left = frame_idx - left_keyframe
                    dist_right = right_keyframe - frame_idx
                    primary_masks = forward_masks if dist_left <= dist_right else reverse_masks
                    fallback_masks = reverse_masks if primary_masks is forward_masks else forward_masks
                    chosen_mask = primary_masks.get(frame_idx)
                    if chosen_mask is None:
                        chosen_mask = fallback_masks.get(frame_idx)
                if chosen_mask is not None:
                    final_masks[frame_idx] = chosen_mask

        last_keyframe = keyframes[-1]
        forward_masks = run_segment([last_keyframe], last_keyframe, num_frames - 1 - last_keyframe, False)
        for frame_idx, mask in forward_masks.items():
            final_masks[frame_idx] = mask

        if len(keyframes) == 1 and first_keyframe > 0:
            reverse_masks = run_segment([first_keyframe], first_keyframe, first_keyframe, True)
            for frame_idx, mask in reverse_masks.items():
                final_masks[frame_idx] = mask

        print(f"Generated {len(final_masks)} masks from {len(keyframes)} keyframes")
        return final_masks

    def track_video_with_keyframes(self, frames: List[np.ndarray], prompts_by_frame, obj_id: int = 1) -> List[np.ndarray]:
        temp_dir = self._save_frames_to_temp_dir(frames)
        try:
            return self.track_video_from_dir(temp_dir / 'frames', prompts_by_frame, obj_id=obj_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def track_video(self, frames: List[np.ndarray], points: List[List[int]], labels: List[int]) -> List[np.ndarray]:
        return self.track_video_with_keyframes(
            frames,
            {0: {"points": points, "labels": labels}},
            obj_id=1,
        )

    def get_frame_mask(self, frame: np.ndarray, points: List[List[int]], labels: List[int]) -> np.ndarray:
        temp_dir = self._save_frames_to_temp_dir([frame])
        try:
            inference_state = self.predictor.init_state(video_path=str(temp_dir / 'frames'))
            self.predictor.reset_state(inference_state)
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            _, object_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_array,
                labels=labels_array,
            )
            return self._extract_object_mask(object_ids, mask_logits, obj_id=1, frame_shape=frame.shape[:2])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_first_frame_mask(self, frame: np.ndarray, points: List[List[int]], labels: List[int]) -> np.ndarray:
        return self.get_frame_mask(frame, points, labels)


def load_sam2_tracker(checkpoint_path=None, device="cuda"):
    if checkpoint_path is None:
        checkpoint_path = 'checkpoints/sam2.1_hiera_large.pt'

    config_file = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    if not os.path.exists(config_file):
        config_file = 'sam2_hiera_l.yaml'

    print(f"Loading SAM2 from {checkpoint_path}...")
    print(f"Using config: {config_file}")
    return SAM2VideoTracker(checkpoint_path, config_file, device)
