"""Tests for the model canvas sizing rules.

These encode the three properties the old fixed-canvas path violated:
aspect preservation, no pointless upscaling, and staying inside a pixel budget.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / 'demo'
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import model_canvas  # noqa: E402


class CanvasForTests(unittest.TestCase):
    def test_canvas_is_64_aligned(self):
        width, height = model_canvas.canvas_for(2325, 1641, budget_px=2_400_000)
        self.assertEqual(width % 64, 0)
        self.assertEqual(height % 64, 0)

    def test_canvas_preserves_region_aspect(self):
        # The regression this module exists for: 2325x1641 (aspect 1.417) used
        # to be stretched onto a fixed 2048x1152 (aspect 1.778), a 25.5%
        # horizontal distortion of the subject.
        region_w, region_h = 2325, 1641
        width, height = model_canvas.canvas_for(region_w, region_h, budget_px=2_400_000)
        region_aspect = region_w / region_h
        canvas_aspect = width / height
        self.assertLess(abs(canvas_aspect / region_aspect - 1.0), 0.05)

    def test_canvas_never_exceeds_the_pixel_budget(self):
        for budget in (500_000, 1_000_000, 2_400_000, 8_000_000):
            width, height = model_canvas.canvas_for(3840, 2160, budget_px=budget)
            self.assertLessEqual(width * height, budget, f"budget {budget}")

    def test_canvas_does_not_upscale_a_small_region(self):
        # A 640x480 ROI given a huge budget should stay at or below its own
        # size. Upscaling into the model invents no detail and costs VRAM.
        width, height = model_canvas.canvas_for(640, 480, budget_px=50_000_000)
        self.assertLessEqual(width, 640)
        self.assertLessEqual(height, 480)

    def test_canvas_may_upscale_when_allowed(self):
        width, height = model_canvas.canvas_for(
            640, 480, budget_px=50_000_000, allow_upscale=True
        )
        self.assertGreater(width, 640)

    def test_region_smaller_than_alignment_still_yields_a_runnable_canvas(self):
        # The UNet cannot run below one alignment block, so this is the one case
        # where upscaling is unavoidable rather than wasteful.
        width, height = model_canvas.canvas_for(40, 30, budget_px=2_400_000)
        self.assertGreaterEqual(width, 64)
        self.assertGreaterEqual(height, 64)

    def test_square_region_gets_a_square_canvas(self):
        width, height = model_canvas.canvas_for(1600, 1600, budget_px=2_400_000)
        self.assertEqual(width, height)

    def test_tall_region_gets_a_tall_canvas(self):
        width, height = model_canvas.canvas_for(800, 1600, budget_px=2_400_000)
        self.assertLess(width, height)

    def test_larger_budget_never_yields_a_smaller_canvas(self):
        previous = 0
        for budget in (500_000, 1_000_000, 2_000_000, 4_000_000):
            width, height = model_canvas.canvas_for(3840, 2160, budget_px=budget)
            self.assertGreaterEqual(width * height, previous)
            previous = width * height

    def test_rejects_degenerate_regions(self):
        for bad in ((0, 100), (100, 0), (-10, 10)):
            with self.assertRaises(ValueError):
                model_canvas.canvas_for(bad[0], bad[1], budget_px=2_400_000)

    def test_rejects_nonpositive_budget(self):
        with self.assertRaises(ValueError):
            model_canvas.canvas_for(1920, 1080, budget_px=0)


class PixelBudgetTests(unittest.TestCase):
    def test_budget_scales_with_available_vram(self):
        small = model_canvas.pixel_budget(
            available_vram_bytes=8 << 30, frames_in_chunk=2, ceiling_px=10_000_000
        )
        large = model_canvas.pixel_budget(
            available_vram_bytes=20 << 30, frames_in_chunk=2, ceiling_px=10_000_000
        )
        self.assertGreater(large, small)

    def test_budget_shrinks_as_the_chunk_grows(self):
        two = model_canvas.pixel_budget(
            available_vram_bytes=20 << 30, frames_in_chunk=2, ceiling_px=10_000_000
        )
        eight = model_canvas.pixel_budget(
            available_vram_bytes=20 << 30, frames_in_chunk=8, ceiling_px=10_000_000
        )
        self.assertLess(eight, two)

    def test_ceiling_caps_the_budget(self):
        budget = model_canvas.pixel_budget(
            available_vram_bytes=80 << 30, frames_in_chunk=1, ceiling_px=2_359_296
        )
        self.assertEqual(budget, 2_359_296)

    def test_vram_beats_the_minimum_budget_floor(self):
        """MIN_PIXEL_BUDGET is a preference, not a guarantee.

        Applying it with max() overrode the VRAM limit: at chunk_size 16 on a
        17 GiB budget it proposed 589,824 px/frame, a predicted peak of 22.5
        GiB -- a guaranteed OOM. A small canvas is bad; an OOM mid-run is worse.
        """
        available = 17 << 30
        for chunk in (8, 12, 16, 20, 32):
            budget = model_canvas.pixel_budget(
                available_vram_bytes=available, frames_in_chunk=chunk,
                ceiling_px=model_canvas.HARD_CANVAS_PX_LIMIT,
            )
            peak = (model_canvas.MEASURED_FIXED_OVERHEAD_BYTES
                    + model_canvas.MEASURED_BYTES_PER_PIXEL_FRAME * budget * chunk)
            self.assertLessEqual(
                peak, available,
                f"chunk {chunk}: budget {budget} predicts {peak / 2**30:.1f} GiB "
                f"vs {available / 2**30:.1f} available",
            )

    def test_rejects_nonpositive_chunk(self):
        with self.assertRaises(ValueError):
            model_canvas.pixel_budget(
                available_vram_bytes=20 << 30, frames_in_chunk=0, ceiling_px=10_000_000
            )

    def test_budget_subtracts_the_resident_model_overhead(self):
        """Calibration measured peak = 4.24 GiB fixed + ~2073 B/px/frame.

        A purely proportional model ignores that fixed term and over-estimates
        the budget by more than 2x on a 23GB card, which OOMs instead of
        degrading. Doubling available VRAM must therefore MORE than double the
        budget -- the signature of the fixed term being accounted for.
        """
        low = model_canvas.pixel_budget(
            available_vram_bytes=10 << 30, frames_in_chunk=2, ceiling_px=100_000_000
        )
        high = model_canvas.pixel_budget(
            available_vram_bytes=20 << 30, frames_in_chunk=2, ceiling_px=100_000_000
        )
        self.assertGreater(high, 2 * low)

    def test_budget_collapses_when_vram_cannot_even_hold_the_weights(self):
        """Returns the smallest runnable area, not the training-size floor.

        This assertion previously expected MIN_PIXEL_BUDGET, which encoded the
        bug: nothing this function returns can rescue a card that cannot hold
        the 4.24 GiB of weights, and returning the floor pretends it fits.
        """
        budget = model_canvas.pixel_budget(
            available_vram_bytes=3 << 30, frames_in_chunk=2, ceiling_px=100_000_000
        )
        self.assertLess(budget, model_canvas.MIN_PIXEL_BUDGET)
        self.assertGreaterEqual(budget, model_canvas.DEFAULT_ALIGNMENT ** 2)

    def test_measured_constants_reproduce_the_calibration(self):
        """The fit must predict the measured peaks it was derived from."""
        for canvas_px, frames, measured_gib in (
            (896 * 640, 2, 6.46),
            (1664 * 1152, 2, 11.64),
            (1984 * 1408, 2, 15.03),
        ):
            predicted = (model_canvas.MEASURED_FIXED_OVERHEAD_BYTES
                         + model_canvas.MEASURED_BYTES_PER_PIXEL_FRAME * canvas_px * frames)
            self.assertAlmostEqual(predicted / 2 ** 30, measured_gib, delta=0.05)


class CanvasForRegionTests(unittest.TestCase):
    """The single call the app makes: VRAM -> budget -> aspect-matched canvas."""

    L4_AVAILABLE = 20 << 30

    def test_returns_concrete_aligned_dimensions(self):
        width, height, report = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=2, available_vram_bytes=self.L4_AVAILABLE
        )
        self.assertEqual(width % 64, 0)
        self.assertEqual(height % 64, 0)
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)

    def test_report_explains_the_choice(self):
        _w, _h, report = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=2, available_vram_bytes=self.L4_AVAILABLE
        )
        for key in ('budget_px', 'canvas_px', 'limited_by', 'predicted_peak_bytes'):
            self.assertIn(key, report)

    def test_says_when_the_quality_ceiling_is_what_bound_it(self):
        _w, _h, report = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=1,
            available_vram_bytes=80 << 30, ceiling_px=1_000_000,
        )
        self.assertEqual(report['limited_by'], 'quality-ceiling')

    def test_report_flags_a_chunk_too_large_for_this_gpu(self):
        """Actionable signal: the fix is a smaller chunk, not a smaller canvas."""
        _w, _h, report = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=32,
            available_vram_bytes=12 << 30, ceiling_px=100_000_000,
        )
        self.assertTrue(report['below_min_budget'])
        _w, _h, ok = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=2,
            available_vram_bytes=20 << 30, ceiling_px=100_000_000,
        )
        self.assertFalse(ok['below_min_budget'])

    def test_says_when_vram_is_what_bound_it(self):
        _w, _h, report = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=8,
            available_vram_bytes=12 << 30, ceiling_px=100_000_000,
        )
        self.assertEqual(report['limited_by'], 'vram')

    def test_says_when_the_region_itself_is_what_bound_it(self):
        _w, _h, report = model_canvas.canvas_for_region(
            640, 480, frames_in_chunk=1,
            available_vram_bytes=80 << 30, ceiling_px=100_000_000,
        )
        self.assertEqual(report['limited_by'], 'region-size')

    def test_predicted_peak_stays_inside_available_vram(self):
        for frames in (1, 2, 4, 8, 12, 16, 24, 32):
            _w, _h, report = model_canvas.canvas_for_region(
                4096, 2160, frames_in_chunk=frames,
                available_vram_bytes=self.L4_AVAILABLE, ceiling_px=100_000_000,
            )
            self.assertLessEqual(
                report['predicted_peak_bytes'], self.L4_AVAILABLE,
                f"predicted peak exceeds available VRAM at {frames} frames",
            )

    def test_a_bigger_chunk_yields_a_smaller_canvas(self):
        small_chunk = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=2,
            available_vram_bytes=self.L4_AVAILABLE, ceiling_px=100_000_000)[:2]
        big_chunk = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=8,
            available_vram_bytes=self.L4_AVAILABLE, ceiling_px=100_000_000)[:2]
        self.assertLess(big_chunk[0] * big_chunk[1], small_chunk[0] * small_chunk[1])

    def test_vram_is_bucketed_so_minor_drift_does_not_move_the_canvas(self):
        """Auto sizing becomes part of the run cache key.

        Bucketing means any two VRAM readings in the same bucket give the same
        canvas, so another process nibbling a little memory does not silently
        change the model input. It does NOT make the canvas invariant across
        bucket boundaries -- run-to-run stickiness comes from the resolved
        dimensions being written into session.json and reused on resume.
        """
        bucket = model_canvas.VRAM_BUCKET_BYTES
        base = (self.L4_AVAILABLE // bucket) * bucket
        baseline = model_canvas.canvas_for_region(
            4096, 2160, frames_in_chunk=2,
            available_vram_bytes=base, ceiling_px=100_000_000)[:2]
        for offset in (1, bucket // 4, bucket // 2, bucket - 1):
            same_bucket = model_canvas.canvas_for_region(
                4096, 2160, frames_in_chunk=2,
                available_vram_bytes=base + offset, ceiling_px=100_000_000)[:2]
            self.assertEqual(same_bucket, baseline, f"canvas moved within bucket (+{offset}B)")

    def test_never_exceeds_the_cuda_grid_limit(self):
        """The limit that is not about memory.

        Calibration: 2752x1408 (60,544 latent tokens) ran; 3136x1664 (81,536)
        died with `CUDA error: invalid configuration argument` inside
        scaled_dot_product_attention. The temporal attention block launches one
        grid slot per latent spatial token and CUDA caps that at 65,535. A
        bigger GPU does not lift this, and the failure poisons the CUDA context
        rather than raising something recoverable.
        """
        for frames in (1, 2, 4):
            width, height, report = model_canvas.canvas_for_region(
                4096, 2160, frames_in_chunk=frames,
                available_vram_bytes=200 << 30, ceiling_px=1_000_000_000,
            )
            latent_tokens = (width // 8) * (height // 8)
            self.assertLessEqual(latent_tokens, 65535,
                                 f"{width}x{height} would fail the kernel launch")
            self.assertLessEqual(width * height, model_canvas.HARD_CANVAS_PX_LIMIT)
            self.assertEqual(report['limited_by'], 'cuda-limit')

    def test_known_good_and_known_bad_calibration_points(self):
        self.assertLessEqual(2752 * 1408, model_canvas.HARD_CANVAS_PX_LIMIT)   # measured OK
        self.assertGreater(3136 * 1664, model_canvas.HARD_CANVAS_PX_LIMIT)     # measured CUDA-LIMIT


class ShrinkCanvasTests(unittest.TestCase):
    """Recovery path for when the memory prediction is wrong.

    A torch OOM is recoverable (unlike the CUDA grid-limit error, which poisons
    the context), so the pass can retry at a smaller canvas instead of losing
    the run. The fit is measured on one machine at one chunk size; it WILL be
    wrong somewhere, and the system has to survive that.
    """

    def test_shrink_returns_a_strictly_smaller_canvas(self):
        width, height = model_canvas.shrink_canvas(1088, 1216)
        self.assertLess(width * height, 1088 * 1216)

    def test_shrink_stays_aligned(self):
        width, height = model_canvas.shrink_canvas(1088, 1216)
        self.assertEqual(width % 64, 0)
        self.assertEqual(height % 64, 0)

    def test_shrink_preserves_aspect(self):
        width, height = model_canvas.shrink_canvas(1088, 1216)
        self.assertLess(abs((width / height) / (1088 / 1216) - 1.0), 0.08)

    def test_repeated_shrink_always_terminates(self):
        width, height = 2048, 1152
        seen = set()
        for _ in range(50):
            width, height = model_canvas.shrink_canvas(width, height)
            self.assertGreaterEqual(width, 64)
            self.assertGreaterEqual(height, 64)
            if (width, height) in seen:
                break
            seen.add((width, height))
        self.assertEqual((width, height), (64, 64))

    def test_shrink_never_goes_below_one_block(self):
        self.assertEqual(model_canvas.shrink_canvas(64, 64), (64, 64))


if __name__ == '__main__':
    unittest.main()
