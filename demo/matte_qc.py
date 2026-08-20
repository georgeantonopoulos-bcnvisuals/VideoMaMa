"""
Lightweight matte QC: does the produced alpha agree with the SAM 3 mask?

This is deliberately not an automatic rescue system. There is no metric here
reliable enough to re-run a range on its own authority. What it does give the
artist is a short list of frames worth looking at first, which on a 300-frame
shot is most of the value.

The one signal that reliably catches real failures is disagreement between the
support of the SAM 3 segmentation (which we trust, because an artist prompted
it) and the support of the alpha. Tracking loss shows up as near-empty alpha;
identity swap and bleed show up as large alpha mass far outside the mask.
"""

import numpy as np


def _binary(mask, threshold=0):
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array > threshold


def _dilate_bool(mask, radius):
    """Square-kernel dilation via cumulative sums; avoids an OpenCV dependency."""
    radius = int(radius)
    if radius <= 0:
        return mask
    padded = np.pad(mask.astype(np.int32), radius, mode="constant")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    size = 2 * radius + 1
    height, width = mask.shape
    window = (
        integral[size:size + height, size:size + width]
        - integral[0:height, size:size + width]
        - integral[size:size + height, 0:width]
        + integral[0:height, 0:width]
    )
    return window > 0


def frame_metrics(sam_mask, alpha, alpha_threshold=0.5, tolerance_px=8):
    """Compare one SAM 3 mask against one float alpha.

    `tolerance_px` dilates the SAM mask before measuring outside-mass, because a
    correct soft matte legitimately extends past the binary mask wherever there
    is hair, motion blur or a translucent edge. Only alpha well beyond that band
    counts as bleed.
    """
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    mask = _binary(sam_mask)
    if mask.shape != alpha.shape:
        raise ValueError(f"QC shape mismatch: mask {mask.shape} vs alpha {alpha.shape}")

    mask_area = float(mask.sum())
    alpha_mass = float(alpha.sum())
    hard_alpha = alpha >= float(alpha_threshold)
    intersection = float(np.logical_and(hard_alpha, mask).sum())
    union = float(np.logical_or(hard_alpha, mask).sum())

    tolerant = _dilate_bool(mask, tolerance_px)
    outside_mass = float(alpha[~tolerant].sum())

    return {
        "mask_area": mask_area,
        "alpha_mass": alpha_mass,
        "alpha_coverage": (alpha_mass / mask_area) if mask_area > 0 else 0.0,
        "iou": (intersection / union) if union > 0 else (1.0 if mask_area == 0 else 0.0),
        "outside_ratio": (outside_mass / alpha_mass) if alpha_mass > 0 else 0.0,
        "soft_pixels": int(np.count_nonzero((alpha > 0.02) & (alpha < 0.98))),
        "empty_alpha": bool(alpha_mass < max(1.0, 0.01 * mask_area)),
        "mask_empty": bool(mask_area == 0),
    }


DEFAULT_THRESHOLDS = {
    "min_iou": 0.55,
    "max_outside_ratio": 0.30,
    "min_alpha_coverage": 0.35,
    "max_alpha_coverage": 3.0,
}


def flag_frame(metrics, thresholds=None):
    """Return a list of human-readable reasons this frame looks suspicious."""
    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    reasons = []
    if metrics.get("mask_empty"):
        return reasons  # nothing to disagree with; not a matting failure
    if metrics.get("empty_alpha"):
        reasons.append("alpha is empty where SAM 3 found the subject")
    if metrics.get("iou", 1.0) < limits["min_iou"]:
        reasons.append(f"low IoU vs SAM 3 mask ({metrics.get('iou', 0.0):.2f})")
    if metrics.get("outside_ratio", 0.0) > limits["max_outside_ratio"]:
        reasons.append(
            f"{metrics.get('outside_ratio', 0.0) * 100:.0f}% of alpha mass lies outside the mask"
        )
    coverage = metrics.get("alpha_coverage", 1.0)
    if coverage < limits["min_alpha_coverage"]:
        reasons.append(f"alpha covers only {coverage:.2f}x the mask area")
    elif coverage > limits["max_alpha_coverage"]:
        reasons.append(f"alpha covers {coverage:.2f}x the mask area")
    return reasons


def summarize(per_frame, thresholds=None):
    """Collapse per-frame metrics into flagged frames and contiguous ranges."""
    flagged = {}
    for frame_idx, metrics in sorted(per_frame.items(), key=lambda item: int(item[0])):
        reasons = flag_frame(metrics, thresholds)
        if reasons:
            flagged[int(frame_idx)] = reasons

    ranges = []
    for frame_idx in sorted(flagged):
        if ranges and frame_idx == ranges[-1][1] + 1:
            ranges[-1][1] = frame_idx
        else:
            ranges.append([frame_idx, frame_idx])

    return {
        "frames_checked": len(per_frame),
        "flagged_frames": {str(k): v for k, v in flagged.items()},
        "suspicious_ranges": [[int(a), int(b)] for a, b in ranges],
        "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})},
    }


def status_line(summary):
    """One sentence for the UI status box."""
    ranges = summary.get("suspicious_ranges") or []
    if not ranges:
        return f"QC: no suspicious frames in {summary.get('frames_checked', 0)} checked."
    formatted = ", ".join(
        f"{a}" if a == b else f"{a}-{b}" for a, b in ranges[:6]
    )
    more = "" if len(ranges) <= 6 else f" (+{len(ranges) - 6} more)"
    return (
        f"QC: {len(summary.get('flagged_frames') or {})} of "
        f"{summary.get('frames_checked', 0)} frames look suspicious; "
        f"review ranges {formatted}{more}."
    )
