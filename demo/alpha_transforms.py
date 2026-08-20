"""
Geometry for moving a matte between model space, ROI space and source space.

Pure numpy/PIL. This is the mapping that decides whether a matte lands on the
right pixels, so it is kept out of the Gradio app where it can be read and
tested on its own.

Two invariants hold everywhere below:

- Alpha never passes through 8 bits. Every resample is float32, because the
  production writers emit 16-bit PNG and half EXR and a byte round trip would
  throw away most of that range.
- Resampling never leaves [0, 1]. Bicubic overshoots at hard edges, and a matte
  with negative or >1 values is not a matte.
"""

import numpy as np
from PIL import Image


def crop_to_roi(array, roi):
    """Crop an array to a source-space ROI, or return it unchanged."""
    if not roi or not roi.get("enabled"):
        return array
    x, y = int(roi["x"]), int(roi["y"])
    width, height = int(roi["width"]), int(roi["height"])
    return array[y:y + height, x:x + width]


def resize_alpha(alpha, size):
    """Resample a float alpha to (width, height) without quantizing it."""
    width, height = int(size[0]), int(size[1])
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.shape == (height, width):
        return alpha
    resized = np.array(
        Image.fromarray(alpha, mode="F").resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    return np.clip(resized, 0.0, 1.0)


def place_alpha_from_roi(alpha, roi, source_width, source_height):
    """Map an alpha produced in ROI space back onto the full source frame.

    The alpha arriving here is at the *staged* size, which may be smaller than
    the ROI when the frame cap kicked in, so it is resized to the ROI first and
    then written into an otherwise-zero source-resolution frame.
    """
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if not roi or not roi.get("enabled"):
        return resize_alpha(alpha, (source_width, source_height))
    width, height = int(roi["width"]), int(roi["height"])
    alpha = resize_alpha(alpha, (width, height))
    full = np.zeros((int(source_height), int(source_width)), dtype=np.float32)
    x, y = int(roi["x"]), int(roi["y"])
    full[y:y + height, x:x + width] = alpha
    return full


def blend_overlap_alpha(previous, current, position, count):
    """Cross-fade the frames two processing units both produced.

    `position` is the frame's index within the overlap and `count` its length,
    so the weight ramps strictly between the two results without ever landing on
    0 or 1 — a hard cut at either end would reintroduce the seam.
    """
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.shape != current.shape:
        raise ValueError(
            f"Cannot blend overlap mattes with shapes {previous.shape} and {current.shape}."
        )
    weight = float(position + 1) / float(count + 1)
    return np.clip(previous * (1.0 - weight) + current * weight, 0.0, 1.0).astype(np.float32)


def downsample_for_metrics(array, long_edge=512, nearest=False):
    """Shrink a mask or alpha for QC.

    QC compares ratios, so measuring at full 4K costs a lot and buys nothing.
    Masks resample with nearest to stay binary; alpha resamples bilinearly.
    """
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= long_edge:
        return array
    scale = long_edge / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if nearest:
        return np.array(
            Image.fromarray(array.astype(np.uint8), mode="L").resize(
                size, Image.Resampling.NEAREST
            )
        )
    return np.array(
        Image.fromarray(array.astype(np.float32), mode="F").resize(
            size, Image.Resampling.BILINEAR
        )
    )
