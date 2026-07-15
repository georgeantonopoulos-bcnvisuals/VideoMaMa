"""
VideoMaMa Inference Wrapper
Handles video matting with mask conditioning
"""

import os
import numpy as np
from PIL import Image, ImageOps


# The VAE reduces image dimensions by 8 and the VideoMaMa UNet downsamples the
# latent three more times. Inputs therefore need image dimensions divisible by
# 64 to avoid odd latent sizes at UNet skip connections (for example, 864px
# becomes a 108px latent and eventually produces the 27-vs-28 tensor mismatch).
MODEL_SIZE_ALIGNMENT = 64


def _model_padding(size, alignment=MODEL_SIZE_ALIGNMENT):
    """Return a centered (left, top, right, bottom) pad to aligned dimensions."""
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"VideoMaMa target size must be positive, got {width}x{height}.")
    aligned_width = ((width + alignment - 1) // alignment) * alignment
    aligned_height = ((height + alignment - 1) // alignment) * alignment
    pad_width = aligned_width - width
    pad_height = aligned_height - height
    left = pad_width // 2
    top = pad_height // 2
    return left, top, pad_width - left, pad_height - top


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
    frame_indices=None,
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
        output_frames: List of float32 numpy arrays, [(H,W,3)]*n, range [0, 1]
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

    model_padding = _model_padding((target_width, target_height))
    if any(model_padding):
        left, top, right, bottom = model_padding
        model_width = target_width + left + right
        model_height = target_height + top + bottom
        print(
            f"Padding VideoMaMa model input from {target_width}x{target_height} "
            f"to {model_width}x{model_height} for UNet alignment; output will be cropped back."
        )
        frames_resized = [ImageOps.expand(frame, border=model_padding, fill=0) for frame in frames_resized]
        masks_resized = [ImageOps.expand(mask, border=model_padding, fill=0) for mask in masks_resized]
    
    # Run inference
    print(
        f"Running VideoMaMa inference on {len(frames_resized)} frames "
        f"(seed={int(seed)}, mask_cond_mode={mask_cond_mode}, fps={int(fps)}, "
        f"motion_bucket_id={int(motion_bucket_id)}, noise_aug_strength={float(noise_aug_strength):.4f})..."
    )
    output_frames_float = pipeline.run(
        cond_frames=frames_resized,
        mask_frames=masks_resized,
        seed=int(seed),
        mask_cond_mode=str(mask_cond_mode),
        fps=int(fps),
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug_strength),
        frame_indices=frame_indices,
        output_type="numpy",
    )

    if len(output_frames_float) != len(frames_np):
        raise RuntimeError(
            f"VideoMaMa returned {len(output_frames_float)} frames for {len(frames_np)} inputs."
        )
    
    # Resize each output back to its matching input resolution. Production shots
    # should be consistent, but pairing frame-by-frame prevents accidental first-
    # frame sizing from leaking into mixed-resolution or retimed test inputs.
    output_frames_resized = []
    for out, src in zip(output_frames_float, frames_pil):
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 2:
            out = out[:, :, None]
        if any(model_padding):
            left, top, right, bottom = model_padding
            expected_height = target_height + top + bottom
            expected_width = target_width + left + right
            if out.shape[:2] != (expected_height, expected_width):
                raise RuntimeError(
                    "VideoMaMa returned an unexpected aligned frame size: "
                    f"expected {expected_width}x{expected_height}, got {out.shape[1]}x{out.shape[0]}."
                )
            out = out[top:top + target_height, left:left + target_width]
        resized_channels = [
            np.array(
                Image.fromarray(out[:, :, channel], mode='F').resize(src.size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            for channel in range(out.shape[2])
        ]
        output_frames_resized.append(np.clip(np.stack(resized_channels, axis=2), 0.0, 1.0))
    return output_frames_resized


def load_videomama_pipeline(device="cuda"):
    """
    Load VideoMaMa pipeline with pretrained weights
    
    Args:
        device: Device to run on
        
    Returns:
        VideoInferencePipeline instance
    """
    import torch
    from pipeline_svd_mask import VideoInferencePipeline

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
