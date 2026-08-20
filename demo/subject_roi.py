"""
Automatic, temporally stable subject ROI derived from generated SAM 3 masks.

Why this exists: SAM2Matting resizes whatever it is given into a square 1024x1024
and emits alpha from a 512x512 decoder head. Feeding it a whole 3840x2160 plate
throws away almost all subject detail before the network ever runs. Cropping a
source-space ROI around the subject and letting the model spend its fixed budget
there is the only way to get real 4K edge quality.

The ROI is computed once for the whole processed range and then held fixed. A
per-frame ROI would track the subject more tightly but makes the effective
sampling grid move every frame, which shows up as a matte that breathes and
crawls along hair edges. A single fixed ROI trades a little resolution for a
stationary sampling grid, which is what a compositor actually wants.

Pure numpy; no torch, no gradio.
"""

import numpy as np


def mask_bbox(mask, threshold=0):
    """Tight bounding box of a mask as (x0, y0, x1, y1), ends exclusive.

    Returns None for an empty mask so callers can distinguish "subject not
    present on this frame" from "subject fills the frame".
    """
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    occupied = array > threshold
    if not occupied.any():
        return None
    rows = np.flatnonzero(occupied.any(axis=1))
    cols = np.flatnonzero(occupied.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def union_bbox(bboxes):
    """Smallest box containing every non-empty box, or None if all are empty."""
    boxes = [box for box in bboxes if box is not None]
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return int(x0), int(y0), int(x1), int(y1)


def _align_up(value, alignment):
    if alignment <= 1:
        return int(value)
    return int(((int(value) + alignment - 1) // alignment) * alignment)


def _grow_to(center_lo, center_hi, target, limit):
    """Grow [center_lo, center_hi) to `target` length inside [0, limit)."""
    target = min(int(target), int(limit))
    length = center_hi - center_lo
    if target <= length:
        return center_lo, center_hi
    deficit = target - length
    lo = center_lo - deficit // 2
    hi = center_hi + (deficit - deficit // 2)
    if lo < 0:
        hi += -lo
        lo = 0
    if hi > limit:
        lo -= hi - limit
        hi = limit
    lo = max(0, lo)
    return int(lo), int(hi)


def compute_stable_roi(
    bboxes,
    source_width,
    source_height,
    padding_ratio=0.12,
    min_padding_px=24,
    aspect_limit=1.9,
    alignment=8,
    min_area_gain=1.35,
    min_size=64,
):
    """Turn per-frame subject boxes into one fixed, padded source-space ROI.

    Args:
        bboxes: iterable of per-frame (x0, y0, x1, y1) boxes; None entries (frames
            where SAM 3 found nothing) are ignored.
        padding_ratio: margin as a fraction of the union box's own size. Hair,
            motion blur and flyaways routinely fall outside the binary SAM mask,
            so the ROI must be looser than the mask that produced it.
        min_padding_px: absolute floor for that margin, for small subjects.
        aspect_limit: maximum width:height (or height:width) of the ROI. The
            model squashes its input to a square, so an extremely elongated ROI
            wastes the budget on distortion; the short side is grown instead.
        min_area_gain: minimum source-area / roi-area ratio required to bother
            cropping. Below this the crop buys little and full-frame is used.

    Returns a dict describing the decision, always safe to feed to the crop path.
    """
    source_width = int(source_width)
    source_height = int(source_height)
    frame_area = float(source_width * source_height)

    union = union_bbox(bboxes)
    if union is None:
        return {
            "enabled": False,
            "x": 0,
            "y": 0,
            "width": source_width,
            "height": source_height,
            "reason": "no SAM 3 mask pixels in range",
            "area_gain": 1.0,
            "union_bbox": None,
        }

    x0, y0, x1, y1 = union
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    pad_x = max(int(round(box_width * float(padding_ratio))), int(min_padding_px))
    pad_y = max(int(round(box_height * float(padding_ratio))), int(min_padding_px))

    left = max(0, x0 - pad_x)
    top = max(0, y0 - pad_y)
    right = min(source_width, x1 + pad_x)
    bottom = min(source_height, y1 + pad_y)

    # Enforce a minimum size and the aspect limit, then align. Every adjustment
    # only ever grows the ROI, so the padded subject can never be cropped away.
    left, right = _grow_to(left, right, max(min_size, _align_up(right - left, alignment)), source_width)
    top, bottom = _grow_to(top, bottom, max(min_size, _align_up(bottom - top, alignment)), source_height)

    if aspect_limit and aspect_limit > 0:
        width = right - left
        height = bottom - top
        if width > height * aspect_limit:
            top, bottom = _grow_to(top, bottom, int(round(width / aspect_limit)), source_height)
        elif height > width * aspect_limit:
            left, right = _grow_to(left, right, int(round(height / aspect_limit)), source_width)

    width = right - left
    height = bottom - top
    area_gain = frame_area / float(max(1, width * height))

    if area_gain < float(min_area_gain):
        return {
            "enabled": False,
            "x": 0,
            "y": 0,
            "width": source_width,
            "height": source_height,
            "reason": (
                f"ROI {width}x{height} only saves {area_gain:.2f}x area "
                f"(threshold {float(min_area_gain):.2f}x); using full frame"
            ),
            "area_gain": float(area_gain),
            "union_bbox": [int(v) for v in union],
        }

    return {
        "enabled": True,
        "x": int(left),
        "y": int(top),
        "width": int(width),
        "height": int(height),
        "reason": f"stable ROI from SAM 3 masks, {area_gain:.2f}x area gain",
        "area_gain": float(area_gain),
        "union_bbox": [int(v) for v in union],
    }


def roi_from_mask_paths(mask_paths, source_width, source_height, reader, **kwargs):
    """compute_stable_roi over masks read lazily one at a time.

    `reader` maps a path to a 2-D array. Reading one mask at a time matters: a
    100-frame 4K mask set is 800 MB if held resident.
    """
    boxes = []
    for path in mask_paths:
        boxes.append(mask_bbox(reader(path)))
    return compute_stable_roi(boxes, source_width, source_height, **kwargs)


def clamp_roi(roi, source_width, source_height, min_size=16):
    """Clamp a manually authored ROI to the frame, rejecting unusable crops."""
    source_width = int(source_width)
    source_height = int(source_height)
    if not roi or not roi.get("enabled"):
        return {"enabled": False, "x": 0, "y": 0, "width": source_width, "height": source_height}
    x = max(0, min(int(roi.get("x", 0)), source_width - 1))
    y = max(0, min(int(roi.get("y", 0)), source_height - 1))
    requested_width = int(roi.get("width", 0))
    requested_height = int(roi.get("height", 0))
    width = source_width - x if requested_width <= 0 else min(requested_width, source_width - x)
    height = source_height - y if requested_height <= 0 else min(requested_height, source_height - y)
    if width < min_size or height < min_size:
        raise ValueError(f"Matte ROI is too small after clamping: {width}x{height}.")
    return {"enabled": True, "x": x, "y": y, "width": int(width), "height": int(height)}
