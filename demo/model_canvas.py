"""Choosing the pixel canvas a diffusion stage actually runs on.

The old path resized every region onto one fixed canvas (1024x576, later
2048x1152) with a plain non-uniform `resize`. That did three unhelpful things
at once:

- It distorted the subject whenever the region's aspect differed from the
  canvas. A 2325x1641 ROI onto a 2048x1152 canvas stretches the subject 25.5%
  horizontally, which is out-of-distribution for a model trained on undistorted
  footage.
- It decimated the two axes by different amounts, so vertical detail was thrown
  away roughly 1.4x harder than horizontal.
- It happily *upscaled* a small ROI up to the canvas, spending VRAM to carry no
  extra information.

This module picks a canvas that matches the region's aspect, stays inside a
pixel budget derived from real free VRAM, and never upscales unless asked.

Pure stdlib. No torch, no PIL, no gradio -- the sizing rules are arithmetic and
are worth being able to test without a GPU in the room.
"""

import math

# The VAE downsamples by 8 and the VideoMaMa UNet by a further 8, so both canvas
# dimensions must be divisible by 64 or the UNet skip connections mismatch. See
# MODEL_SIZE_ALIGNMENT in videomama_wrapper.py, which pads to the same grid.
DEFAULT_ALIGNMENT = 64

# Never propose a canvas below the model's training size. Under this, the matte
# is worse than anything the resolution setting was meant to buy.
MIN_PIXEL_BUDGET = 1024 * 576

# Peak VRAM is affine in canvas area, not proportional: a large fixed cost for
# the resident weights plus a per-pixel-per-frame cost for activations. Both
# figures are fitted by scripts/calibrate_resolution.py; on an L4 the fit
# reproduces every measured peak to within 0.01 GiB.
#
#     peak = MEASURED_FIXED_OVERHEAD_BYTES
#          + MEASURED_BYTES_PER_PIXEL_FRAME * canvas_px * frames_in_chunk
#
# Treating this as purely proportional (dividing peak by pixels) folds the 4.2
# GiB of weights into the per-pixel term and over-estimates what fits by more
# than 2x, which OOMs mid-run instead of degrading.
MEASURED_FIXED_OVERHEAD_BYTES = 4_554_383_854   # 4.24 GiB, L4 / fp16 / VideoMaMa
MEASURED_BYTES_PER_PIXEL_FRAME = 2072.8

# Fraction of free VRAM the budget is allowed to claim. The pipeline allocates
# outside the accounted latents too (CLIP encode, VAE group-norm activations,
# allocator fragmentation), so the headroom is not optional.
#
# Lowered from 0.80 after a real OOM: the affine fit was measured across canvas
# sizes at a FIXED chunk_size of 2 and then extrapolated across chunk sizes,
# which it cannot model. `pipeline_svd_mask.py` encodes the VAE in slices of
# VIDEOMAMA_VAE_ENCODE_CHUNK (4) and decodes in slices of 8, so activation
# memory steps with frame count rather than scaling smoothly with px*frames.
# Until the fit is re-measured with chunk size varied, treat the prediction as
# an estimate and rely on the OOM retry path below, not on this number.
DEFAULT_SAFETY = 0.70

# HARD limit, and the one that is not about memory at all.
#
# The VideoMaMa/SVD temporal attention block reshapes to (batch * latent_h *
# latent_w, frames, channels) and launches a kernel with one grid slot per
# latent spatial token. CUDA caps that grid dimension at 65535. Above it,
# `scaled_dot_product_attention` fails with `CUDA error: invalid configuration
# argument` -- and that error poisons the CUDA context, so the process cannot
# retry at a smaller size; the run is simply lost.
#
# Measured on an L4 (scripts/calibrate_resolution.py, full-frame ladder):
#     2752x1408 -> 60,544 latent tokens -> ran, 19.20 GiB
#     3136x1664 -> 81,536 latent tokens -> CUDA-LIMIT
#
# The latent is canvas/8 per axis, so 65535 tokens is 65535*64 canvas pixels.
# This depends on neither GPU size nor chunk length: a bigger card does NOT
# raise it. Nothing may propose a canvas above this.
CUDA_MAX_GRID_DIM = 65535
HARD_CANVAS_PX_LIMIT = CUDA_MAX_GRID_DIM * 64   # 4,194,240 px

# Quality ceiling, independent of what the GPU could fit. VideoMaMa fine-tunes
# Stable Video Diffusion, which is trained at 1024x576 (0.59 Mpx), and the
# calibration ladder shows edge quality flattening well before the hard limit:
# on the 2325x1641 ROI, transition-band width improved 8.564 -> 7.191 px up to
# 1664x1152 (1.92 Mpx) and then stopped, while cost kept climbing (7.7s ->
# 18.4s per chunk). This sits just above the observed plateau and equals the
# previous fixed maximum's area, so it is a cap rather than a regression.
# Raise it with VIDEOMAMA_CANVAS_CEILING_PX to trade time for a gamble.
DEFAULT_CEILING_PX = 2048 * 1152   # 2.36 Mpx

# Free VRAM is bucketed before sizing so that another process taking a little
# memory does not silently change the model input between runs.
VRAM_BUCKET_BYTES = 512 << 20

# Stand-in for "no ceiling", used when a budget must express one constraint in
# isolation before the others are applied.
UNBOUNDED_PX = 1 << 62


def _align_down(value, alignment):
    return int(value) // alignment * alignment


def _align_up(value, alignment):
    return -(-int(value) // alignment) * alignment


def canvas_for(region_w, region_h, budget_px, alignment=DEFAULT_ALIGNMENT,
               allow_upscale=False):
    """Largest aligned canvas with `region`'s aspect that fits `budget_px`.

    Args:
        region_w, region_h: the source-pixel region being matted (the ROI crop,
            or the full frame when no ROI is set).
        budget_px: maximum canvas area, from `pixel_budget`.
        alignment: both dimensions land on this grid; 64 for the VideoMaMa UNet.
        allow_upscale: when False (the default) the canvas never exceeds the
            region in either axis, because enlarging the region before the model
            sees it adds cost and no detail.

    Returns:
        (width, height), both positive multiples of `alignment`.

    A region smaller than one alignment block is the single case where the
    canvas is allowed to exceed it regardless: the UNet cannot run below one
    block, so `(alignment, alignment)` is the floor.
    """
    region_w, region_h = int(region_w), int(region_h)
    if region_w <= 0 or region_h <= 0:
        raise ValueError(f"Region must be positive, got {region_w}x{region_h}.")
    budget_px = int(budget_px)
    if budget_px <= 0:
        raise ValueError(f"Pixel budget must be positive, got {budget_px}.")
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError(f"Alignment must be positive, got {alignment}.")

    scale = math.sqrt(budget_px / float(region_w * region_h))
    if not allow_upscale:
        scale = min(scale, 1.0)
    ideal_w = region_w * scale
    ideal_h = region_h * scale

    # Rounding both axes down keeps the budget but drifts the aspect, and
    # rounding both up breaks it. Score all four corners of the alignment cell
    # and let the aspect decide, which also keeps the result monotonic in the
    # budget.
    width_options = {_align_down(ideal_w, alignment), _align_up(ideal_w, alignment)}
    height_options = {_align_down(ideal_h, alignment), _align_up(ideal_h, alignment)}

    region_aspect = region_w / float(region_h)
    best = None
    for width in width_options:
        for height in height_options:
            if width < alignment or height < alignment:
                continue
            if width * height > budget_px:
                continue
            if not allow_upscale and (width > max(region_w, alignment)
                                      or height > max(region_h, alignment)):
                continue
            # Log-ratio so a 10% stretch scores the same either direction.
            aspect_error = abs(math.log((width / float(height)) / region_aspect))
            # Prefer the truer aspect; break ties toward more pixels.
            candidate = (aspect_error, -(width * height), width, height)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        # Region smaller than one block, or a budget below one block.
        return alignment, alignment
    return best[2], best[3]


def pixel_budget(available_vram_bytes, frames_in_chunk, ceiling_px=DEFAULT_CEILING_PX,
                 safety=DEFAULT_SAFETY,
                 fixed_overhead_bytes=MEASURED_FIXED_OVERHEAD_BYTES,
                 bytes_per_pixel_frame=MEASURED_BYTES_PER_PIXEL_FRAME):
    """How many model pixels one frame may occupy, given available VRAM.

    `available_vram_bytes` must be the VRAM the *peak* may occupy -- free memory
    plus whatever this process already holds -- because the peak includes the
    resident weights. `free_vram_for_peak()` computes that correctly; passing a
    bare `mem_get_info()` free figure after the pipeline is loaded double-counts
    the weights and under-budgets.

    The UNet consumes a whole chunk as one temporal batch, so activations scale
    with `frames_in_chunk * canvas_area`. Inverting the affine fit gives the
    per-frame area the GPU can afford; `ceiling_px` then caps it for quality
    reasons that have nothing to do with memory.
    """
    frames_in_chunk = int(frames_in_chunk)
    if frames_in_chunk <= 0:
        raise ValueError(f"frames_in_chunk must be positive, got {frames_in_chunk}.")
    if bytes_per_pixel_frame <= 0:
        raise ValueError("bytes_per_pixel_frame must be positive.")
    ceiling_px = int(ceiling_px)
    if ceiling_px <= 0:
        raise ValueError(f"ceiling_px must be positive, got {ceiling_px}.")

    for_activations = (float(available_vram_bytes) * float(safety)
                       - float(fixed_overhead_bytes))
    if for_activations <= 0:
        # Not even the weights fit inside the safety margin. Nothing this
        # function returns can save the run, so return the smallest runnable
        # area rather than something that pretends to fit.
        return DEFAULT_ALIGNMENT * DEFAULT_ALIGNMENT

    affordable = for_activations / (frames_in_chunk * float(bytes_per_pixel_frame))
    budget = min(affordable, float(ceiling_px))

    # MIN_PIXEL_BUDGET is a PREFERENCE, not a guarantee. Raising the budget to
    # meet it when VRAM cannot afford it is a predicted OOM: at chunk_size 16 on
    # a 17 GiB budget the floor asked for 589,824 px/frame, a 22.5 GiB peak.
    # A canvas below the model's training size is bad; dying mid-run is worse,
    # so VRAM wins and the caller is told via `below_min_budget`.
    return int(max(DEFAULT_ALIGNMENT * DEFAULT_ALIGNMENT, budget))


def free_vram_for_peak(device=None):
    """VRAM a peak allocation may occupy: free memory plus what we already hold.

    `torch.cuda.mem_get_info()` reports memory not currently allocated, but the
    measured peak includes the resident model weights. Adding back this
    process's own reserved bytes makes the two comparable.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return int(free) + int(torch.cuda.memory_reserved(device))


def free_vram_bytes(device=None):
    """Free VRAM in bytes, or None when there is no usable CUDA device.

    torch is imported lazily so the sizing rules above stay testable on a host
    with no GPU and no torch installed.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)


def describe(width, height, region_w, region_h):
    """One-line summary of a canvas choice, for status text and manifests."""
    region_aspect = region_w / float(region_h) if region_h else 0.0
    canvas_aspect = width / float(height) if height else 0.0
    scale_x = width / float(region_w) if region_w else 0.0
    scale_y = height / float(region_h) if region_h else 0.0
    anisotropy = (scale_x / scale_y) if scale_y else 0.0
    return (
        f"{region_w}x{region_h} region -> {width}x{height} canvas "
        f"({width * height / 1e6:.2f} Mpx, aspect {canvas_aspect:.3f} vs "
        f"{region_aspect:.3f}, scale {scale_x:.3f}x/{scale_y:.3f}x, "
        f"anisotropy {anisotropy:.3f})"
    )


def canvas_for_region(region_w, region_h, frames_in_chunk, available_vram_bytes=None,
                      ceiling_px=DEFAULT_CEILING_PX, alignment=DEFAULT_ALIGNMENT,
                      safety=DEFAULT_SAFETY):
    """Pick the canvas for one region: the single call the app makes.

    Four independent things can bind the result, and the report says which did:

      'vram'            free VRAM, via the affine peak-memory fit
      'quality-ceiling' the model stops improving before the GPU stops fitting
      'cuda-limit'      the 65535-latent-token kernel grid cap (hard)
      'region-size'     the region is smaller than any of the above; upscaling
                        into the model would add cost and no detail

    Returns (width, height, report). The dimensions are concrete ints so the
    caller can write them straight into the run manifest -- auto sizing must
    never reach the cache key as the string "auto", or a resumed run would
    reuse frames built at a different size.
    """
    region_w, region_h = int(region_w), int(region_h)
    if region_w <= 0 or region_h <= 0:
        raise ValueError(f"Region must be positive, got {region_w}x{region_h}.")

    if available_vram_bytes is None:
        available_vram_bytes = free_vram_for_peak()
    if available_vram_bytes is None:
        # No CUDA: fall back to the quality ceiling and let the region cap it.
        vram_px = ceiling_px
        bucketed = None
    else:
        bucketed = (int(available_vram_bytes) // VRAM_BUCKET_BYTES) * VRAM_BUCKET_BYTES
        # Deliberately uncapped: this must express what VRAM alone allows, so
        # the report can say which constraint actually bound the result. Capping
        # it here would make a VRAM-rich GPU report 'vram' when the real binding
        # constraint was the CUDA grid limit. The caps are applied below.
        vram_px = pixel_budget(bucketed, frames_in_chunk, ceiling_px=UNBOUNDED_PX,
                               safety=safety)

    region_px = region_w * region_h
    limits = {
        'vram': int(vram_px),
        'quality-ceiling': int(ceiling_px),
        'cuda-limit': int(HARD_CANVAS_PX_LIMIT),
        'region-size': int(region_px),
    }
    limited_by = min(limits, key=limits.get)
    budget = max(1, min(limits.values()))

    width, height = canvas_for(region_w, region_h, budget_px=budget,
                               alignment=alignment, allow_upscale=False)

    predicted_peak = int(MEASURED_FIXED_OVERHEAD_BYTES
                         + MEASURED_BYTES_PER_PIXEL_FRAME * width * height
                         * int(frames_in_chunk))
    report = {
        'canvas': [width, height],
        'canvas_px': width * height,
        'region': [region_w, region_h],
        'budget_px': int(budget),
        'limits_px': limits,
        'limited_by': limited_by,
        'frames_in_chunk': int(frames_in_chunk),
        'latent_tokens': (width // 8) * (height // 8),
        'available_vram_bytes': None if bucketed is None else int(bucketed),
        'predicted_peak_bytes': predicted_peak,
        # True when VRAM could not even afford the model's training resolution.
        # The actionable fix is a smaller chunk, not a smaller canvas.
        'below_min_budget': bool(budget < MIN_PIXEL_BUDGET),
        'summary': describe(width, height, region_w, region_h),
    }
    return width, height, report


def shrink_canvas(width, height, factor=0.85, alignment=DEFAULT_ALIGNMENT):
    """Next smaller aligned canvas with the same aspect, for OOM recovery.

    The peak-memory fit is an estimate, so a run must be able to survive it
    being wrong. A torch OOM is recoverable -- the allocator raises before
    corrupting anything -- unlike the CUDA grid-limit failure at
    HARD_CANVAS_PX_LIMIT, which poisons the context and cannot be retried at
    all. Only OOM should reach this function.

    Guarantees the result is strictly smaller until it bottoms out at one
    alignment block, so a retry loop always terminates.
    """
    width, height = int(width), int(height)
    if width <= alignment and height <= alignment:
        return alignment, alignment

    target_px = max(alignment * alignment, int(width * height * float(factor) ** 2))
    shrunk_w, shrunk_h = canvas_for(width, height, budget_px=target_px,
                                    alignment=alignment, allow_upscale=False)
    if shrunk_w * shrunk_h >= width * height:
        # Rounding kept it the same size; step down one block on the long axis.
        if width >= height:
            shrunk_w = max(alignment, width - alignment)
            shrunk_h = max(alignment, _align_down(shrunk_w * height / float(width), alignment))
        else:
            shrunk_h = max(alignment, height - alignment)
            shrunk_w = max(alignment, _align_down(shrunk_h * width / float(height), alignment))
    return int(max(alignment, shrunk_w)), int(max(alignment, shrunk_h))
