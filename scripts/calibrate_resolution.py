#!/usr/bin/env python
"""Measure how VideoMaMa matte quality and VRAM behave as the canvas grows.

Why this exists: "run at the maximum resolution the GPU allows" is only the
right goal if quality actually improves with resolution. VideoMaMa fine-tunes
Stable Video Diffusion, which is trained at 1024x576, and diffusion UNets pushed
far past their training scale usually degrade rather than sharpen. This script
answers, on a real plate, two questions the code cannot assume:

  1. Where does the GPU stop fitting?  -> the bytes-per-pixel-per-frame constant
     that `model_canvas.pixel_budget` inverts.
  2. Where does the matte stop improving?  -> the quality ceiling.

It is a measurement tool, not part of normal operation. Nothing here runs during
a production matte.

Usage (from the repo root, inside the SAM 3 UI venv):

    python scripts/calibrate_resolution.py --run-dir tmp/production_sequence_app/<run>

It reuses the SAM 3 masks that run already produced, so it never needs a GPU
tracker pass of its own.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / 'demo'
for path in (str(DEMO_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _lazy_imports():
    """Imported inside a function so --help works without torch or a GPU."""
    import torch
    import model_canvas
    import production_frame_app as app
    from videomama_wrapper import videomama, load_videomama_pipeline
    return torch, model_canvas, app, videomama, load_videomama_pipeline


# ---------------------------------------------------------------- metrics


def _alpha_metrics(alpha_frames):
    """Edge-quality signals for a list of float alphas, all in ROI space.

    Every ladder step returns alpha at the same ROI resolution (the wrapper
    resizes its output back to the input size), so these numbers are directly
    comparable across steps.
    """
    alphas = [np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) for a in alpha_frames]
    if alphas and alphas[0].ndim == 3:
        alphas = [a.mean(axis=2) for a in alphas]

    first = alphas[0]
    soft = (first > 0.05) & (first < 0.95)
    hard = first >= 0.5

    # Perimeter of the hard matte, so band width is per unit of edge rather than
    # per unit of subject area: a bigger subject must not look like a softer one.
    edge = np.zeros_like(hard)
    edge[:-1, :] |= hard[:-1, :] != hard[1:, :]
    edge[:, :-1] |= hard[:, :-1] != hard[:, 1:]
    perimeter = float(edge.sum())

    band_px = float(soft.sum())
    band_width = band_px / perimeter if perimeter else 0.0

    # Total variation inside the transition band. Real hair raises this; a
    # blurred-up low-res matte lowers it.
    grad_y = np.abs(np.diff(first, axis=0, prepend=first[:1, :]))
    grad_x = np.abs(np.diff(first, axis=1, prepend=first[:, :1]))
    detail_energy = float((grad_x + grad_y)[soft].sum()) / perimeter if perimeter else 0.0

    # Frame-to-frame instability inside the band: the failure mode expected if
    # the canvas outgrows the model's training scale.
    crawl = 0.0
    if len(alphas) > 1:
        deltas = [np.abs(alphas[i + 1] - alphas[i]) for i in range(len(alphas) - 1)]
        union_band = soft
        for a in alphas[1:]:
            union_band = union_band | ((a > 0.05) & (a < 0.95))
        if union_band.any():
            crawl = float(np.mean([d[union_band].mean() for d in deltas]))

    return {
        'band_width_px': round(band_width, 4),
        'detail_energy': round(detail_energy, 4),
        'temporal_crawl': round(crawl, 6),
        'soft_px': int(band_px),
        'perimeter_px': int(perimeter),
        'alpha_mass': round(float(first.sum()), 1),
    }


# ---------------------------------------------------------------- ladder


def _build_ladder(region_w, region_h, model_canvas, max_px, include_legacy=True):
    """Aspect-matched canvases at growing budgets, plus the legacy stretch.

    The legacy 2048x1152 entry is what the pipeline does today. Keeping it in
    the ladder is the whole point: it makes "aspect-matched vs stretched"
    a measured comparison rather than an argument.
    """
    steps = []
    seen = set()
    for budget in (0.6e6, 1.2e6, 2.0e6, 2.9e6, 4.0e6, 5.3e6, 6.8e6, 8.5e6, 10.5e6):
        if budget > max_px:
            break
        width, height = model_canvas.canvas_for(
            region_w, region_h, budget_px=int(budget), allow_upscale=False
        )
        if (width, height) in seen:
            continue
        seen.add((width, height))
        steps.append({'label': f"{width}x{height}", 'width': width, 'height': height,
                      'kind': 'aspect-matched'})
    if include_legacy and (2048, 1152) not in seen:
        steps.append({'label': '2048x1152 (legacy stretch)', 'width': 2048,
                      'height': 1152, 'kind': 'legacy-fixed'})
    return steps


def _classify_failure(exc):
    """Name the ways a canvas can be too big, or None if this is a real bug.

    Two distinct ceilings exist and they are NOT the same number:

    - 'oom'          the allocator ran out of VRAM. Predictable from the affine
                     peak-memory fit, and what `pixel_budget` guards against.
    - 'cuda-limit'   a kernel launch exceeded a CUDA grid/block limit, which
                     surfaces as `invalid configuration argument` from
                     `scaled_dot_product_attention` inside the temporal
                     attention block. This depends on the spatial token count,
                     not on free memory, so a bigger GPU does NOT lift it.

    The second one matters because it is not a graceful failure: it poisons the
    CUDA context, so the process cannot simply retry smaller.
    """
    text = str(exc).lower()
    if exc.__class__.__name__ == 'OutOfMemoryError' or 'out of memory' in text:
        return 'oom'
    if 'invalid configuration argument' in text or 'invalid argument' in text:
        return 'cuda-limit'
    if 'cuda error' in text or exc.__class__.__name__ == 'AcceleratorError':
        return 'cuda-error'
    return None


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True,
                        help='A previous run under tmp/production_sequence_app with sam3_masks.')
    parser.add_argument('--frames', type=int, default=2,
                        help='Frames per chunk to matte (default 2, matching the UI default).')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--out', default=str(REPO_ROOT / 'tmp' / 'calibration'))
    parser.add_argument('--full-frame', action='store_true',
                        help='Ignore the run ROI and calibrate on the whole plate.')
    args = parser.parse_args()

    torch, model_canvas, app, videomama, load_videomama_pipeline = _lazy_imports()

    run_dir = Path(args.run_dir)
    session = json.loads((run_dir / 'session.json').read_text())
    manifest_path = run_dir / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    sequence_dir = Path(session['sequence_dir'])
    frame_names = session['frame_names']
    masks_dir = run_dir / 'sam3_masks'
    if not masks_dir.is_dir():
        raise SystemExit(f"No sam3_masks in {run_dir}; run SAM 3 for this sequence first.")

    roi = (manifest.get('videomama') or {}).get('matte_roi')
    if args.full_frame or not roi or not roi.get('enabled'):
        width, height = session['frame_sizes'][0]
        roi = {'enabled': False, 'x': 0, 'y': 0, 'width': width, 'height': height}

    start = int(args.start_frame)
    indices = list(range(start, min(start + int(args.frames), len(frame_names))))
    print(f"Sequence : {sequence_dir}")
    print(f"Frames   : {indices} ({[frame_names[i] for i in indices]})")
    print(f"ROI      : {roi}")

    # Decode plates with the session's colour settings so the calibration sees
    # exactly what a production run would.
    colour = session.get('exr_color_settings') or {}
    frames, masks = [], []
    for idx in indices:
        frame = app._load_rgb_frame(
            str(sequence_dir / frame_names[idx]),
            exr_gamma=session.get('exr_gamma', 1.0),
            exr_exposure=session.get('exr_exposure', 0.0),
            exr_color_mode=colour.get('mode', app.DEFAULT_EXR_COLOR_MODE),
            ocio_input_colorspace=colour.get('input_colorspace', app.DEFAULT_OCIO_INPUT_COLORSPACE),
            ocio_display=colour.get('display', ''),
            ocio_view=colour.get('view', ''),
        )
        mask_path = masks_dir / f"{Path(frame_names[idx]).stem}.png"
        from PIL import Image
        with Image.open(mask_path) as image:
            mask = np.array(image.convert('L'))
        frames.append(app._crop_to_roi(frame, roi))
        masks.append(app._crop_to_roi(mask, roi))

    region_h, region_w = frames[0].shape[:2]
    print(f"Region   : {region_w}x{region_h} ({region_w * region_h / 1e6:.2f} Mpx)")

    ladder = _build_ladder(region_w, region_h, model_canvas,
                           max_px=region_w * region_h)
    print(f"Ladder   : {[s['label'] for s in ladder]}\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = load_videomama_pipeline(device=device)

    results = []
    for step in ladder:
        label = step['label']
        canvas = (step['width'], step['height'])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            outputs = videomama(
                pipeline, frames, masks,
                seed=42, mask_cond_mode='vae', fps=7, motion_bucket_id=127,
                noise_aug_strength=0.0, target_size=canvas,
                frame_indices=indices,
            )
        except BaseException as exc:  # noqa: BLE001 - classify, then stop the ladder
            kind = _classify_failure(exc)
            if kind is None:
                raise
            print(f"{label:32} {kind.upper()} -- ceiling for this GPU/chunk")
            print(f"{'':32} {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            results.append({**step, 'status': kind, 'error': f"{type(exc).__name__}: "
                                                             f"{str(exc).splitlines()[0][:200]}"})
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            break

        elapsed = time.time() - started
        peak = torch.cuda.max_memory_allocated()
        metrics = _alpha_metrics(outputs)
        canvas_px = canvas[0] * canvas[1]
        record = {
            **step,
            'status': 'ok',
            'canvas_px': canvas_px,
            'seconds': round(elapsed, 2),
            'peak_vram_bytes': int(peak),
            'peak_vram_gib': round(peak / 2 ** 30, 2),
            'bytes_per_pixel_frame': round(peak / float(canvas_px * len(frames)), 1),
            **metrics,
        }
        results.append(record)
        print(f"{label:32} {record['peak_vram_gib']:>5.2f} GiB  "
              f"{elapsed:>6.1f}s  band={metrics['band_width_px']:.3f}px  "
              f"detail={metrics['detail_energy']:.3f}  crawl={metrics['temporal_crawl']:.5f}")

        # Save the first frame's alpha so the numbers can be eyeballed.
        from PIL import Image
        alpha = np.asarray(outputs[0], dtype=np.float32)
        if alpha.ndim == 3:
            alpha = alpha.mean(axis=2)
        safe = label.replace(' ', '_').replace('(', '').replace(')', '')
        Image.fromarray((np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)).save(
            out_dir / f"alpha_{safe}.png"
        )
        torch.cuda.empty_cache()

    fitted = [r['bytes_per_pixel_frame'] for r in results if r.get('status') == 'ok']
    summary = {
        'sequence_dir': str(sequence_dir),
        'frames': indices,
        'region': [region_w, region_h],
        'roi': roi,
        'frames_in_chunk': len(frames),
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'free_vram_bytes_at_start': model_canvas.free_vram_bytes(),
        'results': results,
        'fitted_bytes_per_pixel_frame': round(max(fitted), 1) if fitted else None,
    }
    out_path = out_dir / 'resolution_ladder.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Alpha previews in {out_dir}")
    if fitted:
        print(f"Fitted bytes/pixel/frame (worst case): {max(fitted):.1f}")


if __name__ == '__main__':
    main()
