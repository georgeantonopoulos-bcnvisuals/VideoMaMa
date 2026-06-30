"""
VideoMaMa Inference Wrapper
Handles video matting with mask conditioning
"""

import os
import sys
sys.path.append("../")
sys.path.append("../../")

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
import tqdm

from pipeline_svd_mask import VideoInferencePipeline


def videomama(
    pipeline,
    frames_np,
    mask_frames_np,
    seed=42,
    mask_cond_mode="vae",
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=0.0,
    target_size=(1024, 576),
):
    """
    Run VideoMaMa inference on video frames with mask conditioning
    
    Args:
        pipeline: VideoInferencePipeline instance
        frames_np: List of numpy arrays, [(H,W,3)]*n, uint8 RGB frames
        mask_frames_np: List of numpy arrays, [(H,W)]*n, uint8 grayscale masks
        seed: Random seed for reproducible VideoMaMa output
        mask_cond_mode: Mask conditioning mode ("vae" or "interpolate")
        fps: FPS conditioning value passed to the SVD time embedding
        motion_bucket_id: Motion bucket conditioning value
        noise_aug_strength: Conditioning noise strength
        target_size: Model processing size as (width, height)
        
    Returns:
        output_frames: List of numpy arrays, [(H,W,3)]*n, uint8 RGB outputs
    """
    # Convert numpy arrays to PIL Images
    frames_pil = [Image.fromarray(f) for f in frames_np]
    mask_frames_pil = [Image.fromarray(m, mode='L') for m in mask_frames_np]
    
    # Resize to model input size
    target_width, target_height = int(target_size[0]), int(target_size[1])
    frames_resized = [f.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                     for f in frames_pil]
    masks_resized = [m.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                    for m in mask_frames_pil]
    
    # Run inference
    print(
        f"Running VideoMaMa inference on {len(frames_resized)} frames "
        f"(seed={int(seed)}, mask_cond_mode={mask_cond_mode}, fps={int(fps)}, "
        f"motion_bucket_id={int(motion_bucket_id)}, noise_aug_strength={float(noise_aug_strength):.4f})..."
    )
    output_frames_pil = pipeline.run(
        cond_frames=frames_resized,
        mask_frames=masks_resized,
        seed=int(seed),
        mask_cond_mode=str(mask_cond_mode),
        fps=int(fps),
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug_strength),
    )
    
    # Resize each output back to its matching input resolution. Production shots
    # should be consistent, but pairing frame-by-frame prevents accidental first-
    # frame sizing from leaking into mixed-resolution or retimed test inputs.
    output_frames_resized = [
        out.resize(src.size, Image.Resampling.BILINEAR)
        for out, src in zip(output_frames_pil, frames_pil)
    ]

    # Convert back to numpy arrays
    output_frames_np = [np.array(f) for f in output_frames_resized]
    
    return output_frames_np


def load_videomama_pipeline(device="cuda"):
    """
    Load VideoMaMa pipeline with pretrained weights
    
    Args:
        device: Device to run on
        
    Returns:
        VideoInferencePipeline instance
    """
    checkpoints_root = os.environ.get(
        "VIDEOMAMA_CHECKPOINTS",
        "/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints",
    )
    base_model_path = os.environ.get(
        "VIDEOMAMA_BASE_MODEL_PATH",
        os.path.join(checkpoints_root, "stable-video-diffusion-img2vid-xt"),
    )
    unet_checkpoint_path = os.environ.get(
        "VIDEOMAMA_UNET_CHECKPOINT_PATH",
        os.path.join(checkpoints_root, "VideoMaMa"),
    )
    
    print(f"Loading VideoMaMa pipeline from {unet_checkpoint_path}...")
    
    pipeline = VideoInferencePipeline(
        base_model_path=base_model_path,
        unet_checkpoint_path=unet_checkpoint_path,
        weight_dtype=torch.float16,
        device=device
    )
    
    print("VideoMaMa pipeline loaded successfully!")
    
    return pipeline
