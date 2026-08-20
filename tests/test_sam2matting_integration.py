"""
Regression tests for the SAM3.1 + SAM2Matting matting pipeline.

These are all CPU-only. The SAM2Matting worker is exercised through a fake
subprocess so the crop/letterbox mapping, conditioning selection, manifest
invalidation, alpha precision, frame ordering and error reporting are all
covered without a GPU or a 900 MB checkpoint.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import production_frame_app as app  # noqa: E402
import matting_backends as mb  # noqa: E402
import matte_qc  # noqa: E402
import sam2matting_client as client  # noqa: E402
import subject_roi  # noqa: E402
import alpha_transforms  # noqa: E402


# --------------------------------------------------------------------------
# Backend registry and identity
# --------------------------------------------------------------------------


class BackendRegistryTests(unittest.TestCase):
    def test_base_plus_is_the_default_and_resolves_by_id_and_label(self):
        self.assertEqual(mb.DEFAULT_BACKEND_ID, "sam2matting-sam2.1base+")
        spec = mb.resolve_backend(mb.DEFAULT_BACKEND_ID)
        self.assertIs(mb.resolve_backend(spec.label), spec)
        self.assertIs(mb.resolve_backend(spec), spec)
        self.assertEqual(spec.checkpoint_name, "SAM2Matting-SAM2.1Base+.pt")
        self.assertFalse(spec.experimental)

    def test_videomama_is_still_available_and_keeps_its_output_directories(self):
        spec = mb.resolve_backend("videomama")
        self.assertEqual(spec.output_dirnames(), ("videomama_frames", "alpha_frames"))

    def test_backends_do_not_share_output_directories(self):
        names = [mb.resolve_backend(label).output_dirnames() for label in mb.backend_labels()]
        self.assertEqual(len(names), len(set(names)))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            mb.resolve_backend("definitely-not-a-backend")

    def test_identity_separates_variants_checkpoints_and_revisions(self):
        params = mb.SAM2MattingParams()
        runtime = {"commit": "a" * 40, "torch_version": "2.8.0"}
        base = mb.backend_identity(mb.SAM2MATTING_BASE_PLUS, params, runtime, checkpoint_digest="d1")
        tiny = mb.backend_identity(mb.SAM2MATTING_TINY, params, runtime, checkpoint_digest="d1")
        other_weights = mb.backend_identity(
            mb.SAM2MATTING_BASE_PLUS, params, runtime, checkpoint_digest="d2"
        )
        other_commit = mb.backend_identity(
            mb.SAM2MATTING_BASE_PLUS, params, {"commit": "b" * 40}, checkpoint_digest="d1"
        )
        hashes = {mb.stable_hash(x) for x in (base, tiny, other_weights, other_commit)}
        self.assertEqual(len(hashes), 4)

    def test_identity_changes_when_a_matting_parameter_changes(self):
        runtime = {"commit": "a" * 40}
        first = mb.backend_identity(
            mb.SAM2MATTING_BASE_PLUS, mb.SAM2MattingParams(guidance_interval=12), runtime
        )
        second = mb.backend_identity(
            mb.SAM2MATTING_BASE_PLUS, mb.SAM2MattingParams(guidance_interval=4), runtime
        )
        self.assertNotEqual(mb.stable_hash(first), mb.stable_hash(second))


# --------------------------------------------------------------------------
# Conditioning strategy
# --------------------------------------------------------------------------


class ConditioningTests(unittest.TestCase):
    def test_artist_keyframes_always_condition(self):
        frames = mb.conditioning_frames("keyframes", 0, 40, [7, 23])
        self.assertIn(7, frames)
        self.assertIn(23, frames)

    def test_range_start_always_conditions_so_propagation_has_an_anchor(self):
        # SAM2Matting propagates outward from the earliest conditioned frame.
        # A mid-shot range with no keyframe inside it would otherwise have none.
        frames = mb.conditioning_frames("keyframes", 100, 120, [3])
        self.assertEqual(frames, [100])

    def test_periodic_adds_guidance_between_keyframes_and_anchors_the_tail(self):
        frames = mb.conditioning_frames("periodic", 0, 30, [5], guidance_interval=10)
        self.assertEqual(frames, [0, 5, 10, 20, 30])

    def test_dense_conditions_on_every_frame_in_range(self):
        self.assertEqual(mb.conditioning_frames("dense", 4, 8, []), [4, 5, 6, 7, 8])

    def test_keyframes_outside_the_range_are_ignored(self):
        frames = mb.conditioning_frames("keyframes", 10, 20, [2, 15, 99])
        self.assertEqual(frames, [10, 15])

    def test_labels_and_ids_both_resolve(self):
        self.assertEqual(mb.resolve_conditioning(mb.DEFAULT_CONDITIONING_LABEL), "periodic")
        self.assertEqual(mb.resolve_conditioning("dense"), "dense")
        with self.assertRaises(ValueError):
            mb.resolve_conditioning("sometimes")


# --------------------------------------------------------------------------
# Automatic subject ROI
# --------------------------------------------------------------------------


class SubjectROITests(unittest.TestCase):
    def test_roi_covers_every_frame_of_a_moving_subject_with_padding(self):
        boxes = [(1800, 700, 1900, 1100), (1500, 720, 1620, 1150), (2100, 690, 2210, 1080)]
        roi = subject_roi.compute_stable_roi(boxes, 3840, 2160, padding_ratio=0.1, min_padding_px=20)
        self.assertTrue(roi["enabled"])
        for x0, y0, x1, y1 in boxes:
            self.assertLessEqual(roi["x"], x0)
            self.assertLessEqual(roi["y"], y0)
            self.assertGreaterEqual(roi["x"] + roi["width"], x1)
            self.assertGreaterEqual(roi["y"] + roi["height"], y1)

    def test_roi_is_identical_for_every_frame_so_the_matte_cannot_breathe(self):
        boxes = [(100, 100, 200, 400), (150, 90, 260, 420)]
        first = subject_roi.compute_stable_roi(boxes, 1920, 1080)
        second = subject_roi.compute_stable_roi(list(reversed(boxes)), 1920, 1080)
        self.assertEqual(first, second)

    def test_full_frame_fallback_when_the_crop_saves_little(self):
        roi = subject_roi.compute_stable_roi([(5, 5, 1915, 1075)], 1920, 1080, min_area_gain=1.35)
        self.assertFalse(roi["enabled"])
        self.assertEqual((roi["width"], roi["height"]), (1920, 1080))
        self.assertIn("full frame", roi["reason"])

    def test_full_frame_fallback_when_no_mask_has_pixels(self):
        roi = subject_roi.compute_stable_roi([None, None], 1920, 1080)
        self.assertFalse(roi["enabled"])

    def test_extreme_aspect_is_relaxed_because_the_model_squashes_to_a_square(self):
        roi = subject_roi.compute_stable_roi(
            [(0, 1000, 3800, 1060)], 3840, 2160, padding_ratio=0.0,
            min_padding_px=0, aspect_limit=2.0, min_area_gain=1.0,
        )
        self.assertLessEqual(roi["width"] / roi["height"], 2.05)

    def test_roi_never_leaves_the_frame(self):
        roi = subject_roi.compute_stable_roi(
            [(0, 0, 40, 40)], 640, 480, padding_ratio=0.5, min_padding_px=200
        )
        self.assertGreaterEqual(roi["x"], 0)
        self.assertGreaterEqual(roi["y"], 0)
        self.assertLessEqual(roi["x"] + roi["width"], 640)
        self.assertLessEqual(roi["y"] + roi["height"], 480)

    def test_mask_bbox_returns_none_for_an_empty_mask(self):
        self.assertIsNone(subject_roi.mask_bbox(np.zeros((8, 8), dtype=np.uint8)))

    def test_manual_roi_clamping_rejects_a_degenerate_crop(self):
        with self.assertRaises(ValueError):
            subject_roi.clamp_roi(
                {"enabled": True, "x": 630, "y": 470, "width": 100, "height": 100}, 640, 480
            )


# --------------------------------------------------------------------------
# Crop / letterbox -> matte -> full source resolution mapping
# --------------------------------------------------------------------------


class AlphaMappingTests(unittest.TestCase):
    def test_roi_alpha_lands_in_the_right_source_pixels(self):
        roi = {"enabled": True, "x": 40, "y": 24, "width": 32, "height": 16}
        alpha = np.ones((16, 32), dtype=np.float32)
        full = app._place_alpha_from_roi(alpha, roi, 200, 120)
        self.assertEqual(full.shape, (120, 200))
        np.testing.assert_array_equal(full[24:40, 40:72], 1.0)
        self.assertEqual(full.sum(), 32 * 16)

    def test_staged_alpha_is_upsampled_back_to_the_roi_before_placement(self):
        # The worker returns alpha at the staged (capped) size, not the ROI size.
        roi = {"enabled": True, "x": 10, "y": 10, "width": 64, "height": 64}
        staged = np.ones((32, 32), dtype=np.float32)
        full = app._place_alpha_from_roi(staged, roi, 128, 128)
        np.testing.assert_allclose(full[10:74, 10:74], 1.0, atol=1e-5)
        self.assertEqual(full[:10].sum(), 0.0)
        self.assertEqual(full[74:].sum(), 0.0)

    def test_full_frame_alpha_is_resized_not_padded(self):
        alpha = np.full((10, 20), 0.5, dtype=np.float32)
        full = app._place_alpha_from_roi(alpha, {"enabled": False}, 40, 20)
        self.assertEqual(full.shape, (20, 40))
        np.testing.assert_allclose(full, 0.5, atol=1e-5)

    def test_bicubic_resampling_never_leaves_the_zero_one_range(self):
        alpha = np.zeros((16, 16), dtype=np.float32)
        alpha[6:10, 6:10] = 1.0  # a hard edge is where bicubic overshoots
        resized = app._resize_alpha(alpha, (64, 64))
        self.assertGreaterEqual(resized.min(), 0.0)
        self.assertLessEqual(resized.max(), 1.0)

    def test_soft_alpha_survives_the_round_trip_without_binarizing(self):
        roi = {"enabled": True, "x": 0, "y": 0, "width": 8, "height": 8}
        alpha = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        full = app._place_alpha_from_roi(alpha, roi, 8, 8)
        soft = np.count_nonzero((full > 0.02) & (full < 0.98))
        self.assertGreater(soft, 40)


class OverlapBlendTests(unittest.TestCase):
    def test_the_crossfade_never_lands_on_either_endpoint(self):
        previous = np.zeros((2, 2), dtype=np.float32)
        current = np.ones((2, 2), dtype=np.float32)
        weights = [
            float(alpha_transforms.blend_overlap_alpha(previous, current, i, 3)[0, 0])
            for i in range(3)
        ]
        self.assertEqual(weights, sorted(weights))
        self.assertGreater(weights[0], 0.0)
        self.assertLess(weights[-1], 1.0)

    def test_blending_mismatched_shapes_is_rejected(self):
        with self.assertRaises(ValueError):
            alpha_transforms.blend_overlap_alpha(
                np.zeros((2, 2), np.float32), np.zeros((3, 3), np.float32), 0, 1
            )

    def test_the_blend_result_stays_a_valid_matte(self):
        blended = alpha_transforms.blend_overlap_alpha(
            np.full((2, 2), 1.0, np.float32), np.full((2, 2), 1.0, np.float32), 0, 1
        )
        self.assertLessEqual(float(blended.max()), 1.0)
        self.assertGreaterEqual(float(blended.min()), 0.0)


class QCDownsampleTests(unittest.TestCase):
    def test_a_mask_stays_binary_through_downsampling(self):
        mask = np.zeros((2160, 3840), dtype=np.uint8)
        mask[500:1500, 1000:2000] = 255
        small = alpha_transforms.downsample_for_metrics(mask, long_edge=512, nearest=True)
        self.assertEqual(max(small.shape), 512)
        self.assertEqual(set(np.unique(small).tolist()), {0, 255})

    def test_a_small_array_is_returned_untouched(self):
        alpha = np.full((100, 100), 0.5, dtype=np.float32)
        self.assertIs(alpha_transforms.downsample_for_metrics(alpha, long_edge=512), alpha)

    def test_soft_alpha_stays_soft_through_downsampling(self):
        alpha = np.linspace(0.0, 1.0, 1024 * 1024, dtype=np.float32).reshape(1024, 1024)
        small = alpha_transforms.downsample_for_metrics(alpha, long_edge=256)
        self.assertEqual(small.shape, (256, 256))
        self.assertGreater(int(np.count_nonzero((small > 0.02) & (small < 0.98))), 1000)


class AlphaPrecisionTests(unittest.TestCase):
    def test_sixteen_bit_png_preserves_more_than_eight_bits_of_alpha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Values one 16-bit step apart collapse to the same 8-bit code.
            alpha = np.array([[0.25, 0.25 + 1.0 / 65535.0]], dtype=np.float32)
            app._save_matte_outputs(root / "preview", root / "alpha", "f.png", alpha, "16-bit PNG")
            written = np.array(Image.open(root / "alpha" / "f.png"))
            self.assertEqual(written.dtype, np.uint16)
            self.assertEqual(int(written[0, 1]) - int(written[0, 0]), 1)

    def test_half_exr_round_trips_a_soft_gradient(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alpha = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
            app._save_matte_outputs(root / "preview", root / "alpha", "f.png", alpha, "Half EXR")
            path = root / "alpha" / "f.exr"
            self.assertTrue(path.is_file())
            import Imath
            import OpenEXR

            handle = OpenEXR.InputFile(str(path))
            raw = handle.channel("A", Imath.PixelType(Imath.PixelType.HALF))
            handle.close()
            restored = np.frombuffer(raw, dtype=np.float16).reshape(16, 16).astype(np.float32)
            np.testing.assert_allclose(restored, alpha, atol=1e-3)

    def test_the_matte_path_does_not_quantize_before_writing(self):
        # A float alpha handed to the sink must reach the writer unrounded; only
        # the 8-bit preview is allowed to lose precision.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            captured = {}

            def capture(output_dir, alpha_dir, frame_name, alpha, fmt):
                captured["alpha"] = np.array(alpha)

            sink = app._MatteSink(root / "p", root / "a", ["f.png"], "16-bit PNG")
            with mock.patch.object(app, "_save_matte_outputs", side_effect=capture):
                sink.write(0, np.array([[0.123456789]], dtype=np.float32))
            self.assertAlmostEqual(float(captured["alpha"][0, 0]), 0.123456789, places=6)


# --------------------------------------------------------------------------
# Window planning and frame ordering
# --------------------------------------------------------------------------


class WindowPlanningTests(unittest.TestCase):
    def test_a_short_range_is_a_single_window(self):
        self.assertEqual(client.plan_windows(list(range(10)), 300, 8), [list(range(10))])
        self.assertEqual(client.plan_windows(list(range(10)), 0, 8), [list(range(10))])

    def test_windows_cover_every_frame_in_order_with_the_requested_overlap(self):
        indices = list(range(25))
        windows = client.plan_windows(indices, 10, 4)
        self.assertEqual(sorted(set(f for w in windows for f in w)), indices)
        for window in windows:
            self.assertEqual(window, sorted(window))
        for earlier, later in zip(windows, windows[1:]):
            shared = set(earlier) & set(later)
            self.assertEqual(len(shared), 4)

    def test_overlap_is_clamped_so_a_frame_is_in_at_most_two_windows(self):
        windows = client.plan_windows(list(range(30)), 10, 9)
        counts = {}
        for window in windows:
            for frame in window:
                counts[frame] = counts.get(frame, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_a_subrange_keeps_its_global_frame_indices(self):
        windows = client.plan_windows(list(range(100, 130)), 20, 4)
        self.assertEqual(windows[0][0], 100)
        self.assertEqual(windows[-1][-1], 129)


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------


class MatteQCTests(unittest.TestCase):
    def _mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        return mask

    def test_a_matching_alpha_is_not_flagged(self):
        mask = self._mask()
        alpha = (mask > 0).astype(np.float32)
        self.assertEqual(matte_qc.flag_frame(matte_qc.frame_metrics(mask, alpha)), [])

    def test_lost_tracking_is_flagged_as_empty_alpha(self):
        mask = self._mask()
        alpha = np.zeros((64, 64), dtype=np.float32)
        reasons = matte_qc.flag_frame(matte_qc.frame_metrics(mask, alpha))
        self.assertTrue(any("empty" in reason for reason in reasons))

    def test_alpha_bleeding_far_outside_the_mask_is_flagged(self):
        mask = self._mask()
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[16:48, 16:48] = 1.0
        alpha[0:8, :] = 1.0  # a second object picked up by mistake
        reasons = matte_qc.flag_frame(matte_qc.frame_metrics(mask, alpha))
        self.assertTrue(any("outside" in reason for reason in reasons))

    def test_soft_hair_just_outside_the_mask_is_not_flagged(self):
        mask = self._mask()
        alpha = (mask > 0).astype(np.float32)
        alpha[12:16, 16:48] = 0.4  # wisps within the tolerance band
        self.assertEqual(matte_qc.flag_frame(matte_qc.frame_metrics(mask, alpha)), [])

    def test_a_frame_with_no_subject_is_not_treated_as_a_failure(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.float32)
        self.assertEqual(matte_qc.flag_frame(matte_qc.frame_metrics(mask, alpha)), [])

    def test_summary_groups_flagged_frames_into_contiguous_ranges(self):
        mask = self._mask()
        good = matte_qc.frame_metrics(mask, (mask > 0).astype(np.float32))
        bad = matte_qc.frame_metrics(mask, np.zeros((64, 64), dtype=np.float32))
        summary = matte_qc.summarize({0: good, 1: bad, 2: bad, 5: bad, 6: good})
        self.assertEqual(summary["suspicious_ranges"], [[1, 2], [5, 5]])
        self.assertIn("review ranges 1-2, 5", matte_qc.status_line(summary))


# --------------------------------------------------------------------------
# Worker protocol: staging, streaming, failure reporting
# --------------------------------------------------------------------------


class _FakeProcess:
    """Stands in for the SAM2Matting worker subprocess.

    Replays a scripted NDJSON stream and writes the .npy files the client is
    told to read, so the client contract can be tested without a GPU.
    """

    def __init__(self, lines, returncode=0, stderr_lines=(), alpha_dir=None, alpha_shape=(4, 4)):
        self._alpha_dir = Path(alpha_dir) if alpha_dir else None
        self._alpha_shape = alpha_shape
        rendered = []
        for line in lines:
            if line.get("type") == "frame" and self._alpha_dir is not None:
                path = self._alpha_dir / f"{line['index']:06d}.npy"
                value = line.pop("_value", float(line["index"]) / 100.0)
                np.save(path, np.full(self._alpha_shape, value, dtype=np.float32))
                line["path"] = str(path)
                line.setdefault("min", value)
                line.setdefault("max", value)
                line.setdefault("mean", value)
            rendered.append(json.dumps(line) + "\n")
        self.stdout = iter(rendered)
        self.stderr = iter(f"{line}\n" for line in stderr_lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def _fake_popen(lines, returncode=0, stderr_lines=(), alpha_shape=(4, 4), recorder=None):
    def popen(command, **kwargs):
        if recorder is not None:
            recorder.append((command, kwargs))
        job_path = Path(command[command.index("--job") + 1])
        job = json.loads(job_path.read_text())
        return _FakeProcess(
            [dict(line) for line in lines],
            returncode=returncode,
            stderr_lines=stderr_lines,
            alpha_dir=job["alpha_dir"],
            alpha_shape=alpha_shape,
        )

    return popen


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.work = Path(self.tmpdir.name)
        (self.work / "alpha").mkdir(parents=True)
        self.job = {
            "alpha_dir": str(self.work / "alpha"),
            "sam2matting_home": str(self.work),
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_frames_are_yielded_in_the_order_the_worker_emits_them(self):
        lines = [{"type": "ready", "frames": 3}]
        lines += [{"type": "frame", "index": i} for i in (0, 1, 2)]
        lines += [{"type": "done", "written": 3, "peak_vram_bytes": 1}]
        received = list(
            client.run_window(self.work, self.job, "python", popen=_fake_popen(lines))
        )
        self.assertEqual([index for index, _, _ in received], [0, 1, 2])
        self.assertAlmostEqual(float(received[1][1][0, 0]), 0.01, places=5)

    def test_alpha_files_are_deleted_as_they_are_consumed(self):
        lines = (
            [{"type": "ready", "frames": 2}]
            + [{"type": "frame", "index": i} for i in (0, 1)]
            + [{"type": "done", "written": 2, "peak_vram_bytes": 1}]
        )
        for _ in client.run_window(self.work, self.job, "python", popen=_fake_popen(lines)):
            pass
        self.assertEqual(list((self.work / "alpha").glob("*.npy")), [])

    def test_alpha_arrives_as_float32_not_eight_bit(self):
        lines = (
            [{"type": "ready", "frames": 1}]
            + [{"type": "frame", "index": 0, "_value": 0.3372549}]
            + [{"type": "done", "written": 1, "peak_vram_bytes": 1}]
        )
        (_, alpha, _), = client.run_window(
            self.work, self.job, "python", popen=_fake_popen(lines)
        )
        self.assertEqual(alpha.dtype, np.float32)
        self.assertAlmostEqual(float(alpha[0, 0]), 0.3372549, places=6)

    def test_a_worker_error_line_is_reported_with_its_kind(self):
        lines = [
            {"type": "ready", "frames": 1},
            {"type": "error", "kind": "oom", "message": "CUDA out of memory", "traceback": "tb"},
        ]
        with self.assertRaises(client.SAM2MattingError) as caught:
            list(client.run_window(self.work, self.job, "python", popen=_fake_popen(lines)))
        self.assertEqual(caught.exception.kind, "oom")
        self.assertIn("out of memory", str(caught.exception))

    def test_a_crash_without_an_error_line_still_fails_with_stderr_context(self):
        lines = [{"type": "ready", "frames": 1}]
        popen = _fake_popen(lines, returncode=139, stderr_lines=["Segmentation fault"])
        with self.assertRaises(client.SAM2MattingError) as caught:
            list(client.run_window(self.work, self.job, "python", popen=popen))
        self.assertIn("139", str(caught.exception))
        self.assertIn("Segmentation fault", caught.exception.detail)

    def test_a_worker_that_stops_early_is_not_treated_as_success(self):
        lines = [{"type": "ready", "frames": 3}, {"type": "frame", "index": 0}]
        with self.assertRaises(client.SAM2MattingError) as caught:
            list(client.run_window(self.work, self.job, "python", popen=_fake_popen(lines)))
        self.assertIn("without reporting completion", str(caught.exception))

    def test_the_worker_does_not_inherit_the_ui_python_path(self):
        # Our repo root and demo/ are on the UI process's sys.path. Leaking that
        # into the worker would let our modules shadow SAM2Matting's own.
        env = client._worker_env(self.work)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_non_json_worker_chatter_is_logged_not_fatal(self):
        lines = [
            {"type": "ready", "frames": 1},
            {"type": "frame", "index": 0},
            {"type": "done", "written": 1, "peak_vram_bytes": 1},
        ]
        logged = []
        received = list(
            client.run_window(
                self.work, self.job, "python", log=logged.append, popen=_fake_popen(lines)
            )
        )
        self.assertEqual(len(received), 1)
        self.assertTrue(any("Launching SAM2Matting worker" in line for line in logged))


class StagingTests(unittest.TestCase):
    def test_frames_are_named_so_the_upstream_loader_can_sort_them(self):
        # load_video_frames_from_jpg_images sorts with int(splitext(name)[0]) and
        # raises on anything else, so production frame names cannot be reused.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged = client.stage_window(
                Path(tmpdir),
                [100, 101, 102],
                lambda idx: np.zeros((8, 8, 3), dtype=np.uint8),
                lambda idx: np.full((8, 8), 255, dtype=np.uint8),
                [100],
                {"enabled": False},
            )
            names = sorted(p.name for p in staged.frames_dir.glob("*.jpg"))
            self.assertEqual(names, ["00000.jpg", "00001.jpg", "00002.jpg"])
            for name in names:
                int(Path(name).stem)  # must not raise

    def test_only_conditioning_frames_get_a_staged_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            staged = client.stage_window(
                Path(tmpdir),
                [0, 1, 2, 3],
                lambda idx: np.zeros((8, 8, 3), dtype=np.uint8),
                lambda idx: np.full((8, 8), 255, dtype=np.uint8),
                [0, 2],
                {"enabled": False},
            )
            self.assertEqual(sorted(staged.mask_paths), [0, 2])

    def test_the_roi_is_applied_to_both_plate_and_mask(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frame[10:20, 15:25] = 200
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:20, 15:25] = 255
        roi = {"enabled": True, "x": 15, "y": 10, "width": 10, "height": 10}
        with tempfile.TemporaryDirectory() as tmpdir:
            staged = client.stage_window(
                Path(tmpdir), [0], lambda idx: frame, lambda idx: mask, [0], roi
            )
            self.assertEqual(staged.staged_size, (10, 10))
            staged_mask = np.array(Image.open(staged.mask_paths[0]))
            self.assertEqual(staged_mask.shape, (10, 10))
            self.assertTrue((staged_mask > 0).all())

    def test_large_rois_are_capped_but_small_ones_are_never_upscaled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            big = client.stage_window(
                Path(tmpdir) / "big", [0],
                lambda idx: np.zeros((2160, 3840, 3), dtype=np.uint8),
                lambda idx: np.full((2160, 3840), 255, dtype=np.uint8),
                [0], {"enabled": False}, long_edge_cap=1024,
            )
            self.assertEqual(big.staged_size, (1024, 576))
            small = client.stage_window(
                Path(tmpdir) / "small", [0],
                lambda idx: np.zeros((100, 200, 3), dtype=np.uint8),
                lambda idx: np.full((100, 200), 255, dtype=np.uint8),
                [0], {"enabled": False}, long_edge_cap=1024,
            )
            self.assertEqual(small.staged_size, (200, 100))

    def test_staging_without_a_conditioning_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(client.SAM2MattingError):
                client.stage_window(
                    Path(tmpdir), [0],
                    lambda idx: np.zeros((8, 8, 3), dtype=np.uint8),
                    lambda idx: np.zeros((8, 8), dtype=np.uint8),
                    [], {"enabled": False},
                )


# --------------------------------------------------------------------------
# run_sequence end-to-end with a faked worker
# --------------------------------------------------------------------------


def _build_state(root, frame_count=6, size=(400, 300), subject=(120, 90, 200, 180), sam_crop=None):
    """A sequence whose subject occupies a small, constant part of the frame."""
    sequence = root / "sequence"
    masks_dir = root / "run" / "sam3_masks"
    sequence.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    width, height = size
    x0, y0, x1, y1 = subject
    names, paths = [], []
    for index in range(frame_count):
        name = f"frame_{index:04d}.png"
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        frame[y0:y1, x0:x1] = 200
        Image.fromarray(frame).save(sequence / name)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        Image.fromarray(mask).save(masks_dir / name)
        names.append(name)
        paths.append(str(sequence / name))
    state = {
        "sequence_dir": str(sequence),
        "frame_paths": paths,
        "frame_names": names,
        "frame_sizes": [[width, height] for _ in names],
        "frame_transforms": [
            app._letterbox_transform(width, height, work_size=size) for _ in names
        ],
        "cache_frame_paths": paths,
        "cache_dir": str(sequence),
        "current_frame_idx": 0,
        "text_prompt_frame_idx": 0,
        "prompts_by_frame": {"0": {"points": [[30, 20]], "labels": [1]}},
        "preview_masks": {},
        "prompt_mode": "Point keyframes",
        "concept_prompt": "",
        "generated_masks_dir": str(masks_dir),
        "generated_masks_sam_output_prob_thresh": 0.5,
        "videomama_output_dir": None,
        "alpha_output_dir": None,
        "run_root": str(root / "run"),
        "exr_gamma": 1.0,
        "exr_exposure": 0.0,
        "work_size": list(size),
        "sam_output_prob_thresh": 0.5,
        "sam_refine_edges_against_plate": False,
        "sam_crop_settings": dict(sam_crop or app._sam_crop_settings()),
    }
    # Record the SAM 3 manifest the masks above correspond to, so run_sequence
    # treats them as current instead of trying to re-run SAM 3.
    app._update_run_manifest(root / "run", "sam3", app._prompt_manifest_payload(state, 0.5))
    return state


class _StubWorker:
    """Replaces the SAM2Matting subprocess with a deterministic alpha generator."""

    def __init__(self, alpha_value=0.5, fail=None):
        self.alpha_value = alpha_value
        self.fail = fail
        self.jobs = []

    def popen(self, command, **kwargs):
        job_path = Path(command[command.index("--job") + 1])
        job = json.loads(job_path.read_text())
        # Staging is cleaned up as soon as the window finishes, so record what
        # actually existed on disk at launch time rather than asserting later.
        job["_masks_on_disk"] = sorted(
            int(k) for k, path in job["masks"].items() if Path(path).is_file()
        )
        self.jobs.append(job)
        frames = sorted(Path(job["frames_dir"]).glob("*.jpg"))
        with Image.open(frames[0]) as image:
            width, height = image.size
        lines = [{"type": "ready", "frames": len(frames), "video_size": [width, height]}]
        if self.fail is not None:
            lines.append(self.fail)
        else:
            lines += [
                {"type": "frame", "index": i, "_value": self.alpha_value}
                for i in range(len(frames))
            ]
            lines.append({"type": "done", "written": len(frames), "peak_vram_bytes": 1})
        return _FakeProcess(
            lines,
            returncode=1 if self.fail else 0,
            alpha_dir=job["alpha_dir"],
            alpha_shape=(height, width),
        )


def _stub_runtime(root):
    return {
        "commit": "c" * 40,
        "torch_version": "2.8.0+cu128",
        "checkpoint_revision": "r" * 40,
        "checkpoint_sha256": {"SAM2Matting-SAM2.1Base+.pt": "s" * 64},
        "python": "/nonexistent/python",
        "home": str(root),
        "checkpoint_path": str(root / "ckpt.pt"),
    }


class RunSequenceSAM2MattingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.runtime = _stub_runtime(self.root)
        self.torch_patch = mock.patch.dict(
            sys.modules,
            {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))},
        )
        self.torch_patch.start()

    def tearDown(self):
        self.torch_patch.stop()
        self.tmpdir.cleanup()

    def _run(self, state, worker, **kwargs):
        with mock.patch.object(client, "preflight", return_value=self.runtime), \
                mock.patch.object(client.subprocess, "Popen", side_effect=worker.popen):
            defaults = dict(
                chunk_size=4, overlap=0, range_start=0, range_end=len(state["frame_names"]) - 1,
                matting_backend=mb.SAM2MATTING_BASE_PLUS.backend_id,
                alpha_output_format="16-bit PNG", progress=lambda *a, **k: None,
            )
            defaults.update(kwargs)
            return app.run_sequence(state, **defaults)

    def test_produces_full_source_resolution_alpha_from_a_cropped_roi(self):
        state = _build_state(self.root)
        worker = _StubWorker(alpha_value=0.5)
        payload = self._run(state, worker)
        out_state = payload[4]
        # The subject occupies well under 1/1.35 of the frame, so Auto crops.
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertTrue(manifest["matte_roi"]["enabled"])
        self.assertEqual(manifest["matte_roi_report"]["applied"], "auto")
        alpha_dir = Path(out_state["alpha_output_dir"])
        written = np.array(Image.open(alpha_dir / "frame_0000.png"))
        self.assertEqual(written.shape, (300, 400))  # full source resolution
        # Alpha exists inside the ROI and is exactly zero outside it.
        roi = manifest["matte_roi"]
        outside = written.copy()
        outside[roi["y"]:roi["y"] + roi["height"], roi["x"]:roi["x"] + roi["width"]] = 0
        self.assertEqual(int(outside.sum()), 0)
        self.assertGreater(int(written.sum()), 0)

    def test_the_roi_is_identical_for_every_frame_of_the_range(self):
        state = _build_state(self.root, frame_count=5)
        worker = _StubWorker()
        self._run(state, worker)
        # One window, one staged size, one ROI: the sampling grid cannot move.
        self.assertEqual(len(worker.jobs), 1)
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertTrue(manifest["matte_roi"]["enabled"])

    def test_conditioning_masks_are_supplied_for_more_than_the_first_frame(self):
        state = _build_state(self.root, frame_count=9)
        worker = _StubWorker()
        self._run(state, worker, s2m_conditioning="periodic", s2m_guidance_interval=4)
        job = worker.jobs[0]
        # Window-local indices: start, the artist keyframe, the periodic anchors
        # and the tail. Upstream's demo would have supplied only frame 0.
        self.assertEqual(sorted(int(k) for k in job["masks"]), [0, 4, 8])
        self.assertEqual(job["_masks_on_disk"], [0, 4, 8])

    def test_dense_conditioning_supplies_a_mask_for_every_frame(self):
        state = _build_state(self.root, frame_count=5)
        worker = _StubWorker()
        self._run(state, worker, s2m_conditioning="dense")
        self.assertEqual(sorted(int(k) for k in worker.jobs[0]["masks"]), [0, 1, 2, 3, 4])

    def test_every_window_conditions_on_its_own_first_frame(self):
        state = _build_state(self.root, frame_count=12)
        worker = _StubWorker()
        self._run(state, worker, s2m_window_size=6, s2m_window_overlap=2,
                  s2m_conditioning="keyframes")
        self.assertGreater(len(worker.jobs), 1)
        for job in worker.jobs:
            self.assertIn("0", job["masks"])

    def test_a_subrange_writes_only_that_range_and_keeps_frame_identity(self):
        state = _build_state(self.root, frame_count=8)
        worker = _StubWorker()
        payload = self._run(state, worker, range_start=2, range_end=5)
        alpha_dir = Path(payload[4]["alpha_output_dir"])
        written = sorted(p.stem for p in alpha_dir.glob("*.png"))
        self.assertEqual(written, ["frame_0002", "frame_0003", "frame_0004", "frame_0005"])
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertEqual(manifest["range"], [2, 5])
        self.assertEqual(manifest["status"], "partial")

    def test_windows_cover_a_long_range_without_gaps(self):
        state = _build_state(self.root, frame_count=14)
        worker = _StubWorker()
        payload = self._run(state, worker, s2m_window_size=6, s2m_window_overlap=2)
        alpha_dir = Path(payload[4]["alpha_output_dir"])
        self.assertEqual(len(list(alpha_dir.glob("*.png"))), 14)
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertEqual(manifest["completed_frame_indices"], list(range(14)))

    def test_a_worker_failure_surfaces_as_a_user_error_with_partial_progress_kept(self):
        state = _build_state(self.root, frame_count=4)
        worker = _StubWorker(fail={
            "type": "error", "kind": "oom",
            "message": "torch.OutOfMemoryError: CUDA out of memory",
            "traceback": "tb",
        })
        with self.assertRaises(Exception) as caught:
            self._run(state, worker)
        message = str(caught.exception)
        self.assertIn("out of memory", message)
        self.assertIn("CPU offload", message)  # actionable guidance
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertEqual(manifest["status"], "failed")

    def test_a_missing_runtime_is_reported_as_a_setup_problem(self):
        state = _build_state(self.root, frame_count=2)
        with mock.patch.object(
            client, "preflight",
            side_effect=client.SAM2MattingError("nope", kind="setup"),
        ):
            with self.assertRaises(Exception) as caught:
                app.run_sequence(
                    state, chunk_size=2, overlap=0,
                    matting_backend=mb.SAM2MATTING_BASE_PLUS.backend_id,
                    progress=lambda *a, **k: None,
                )
        self.assertIn("nope", str(caught.exception))

    def test_manual_roi_overrides_the_automatic_one(self):
        state = _build_state(
            self.root, frame_count=3,
            sam_crop=app._sam_crop_settings(True, 40, 30, 160, 120),
        )
        worker = _StubWorker()
        self._run(state, worker)
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertEqual(manifest["matte_roi_report"]["applied"], "manual")
        self.assertEqual(
            manifest["matte_roi"],
            {"enabled": True, "x": 40, "y": 30, "width": 160, "height": 120},
        )

    def test_full_frame_mode_disables_cropping_entirely(self):
        state = _build_state(self.root, frame_count=3)
        worker = _StubWorker()
        self._run(state, worker, matte_roi_mode="Full frame")
        manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertFalse(manifest["matte_roi"]["enabled"])
        self.assertEqual(worker.jobs[0]["masks"]["0"].endswith(".png"), True)

    def test_qc_records_metrics_and_flags_an_empty_alpha(self):
        state = _build_state(self.root, frame_count=3)
        worker = _StubWorker()
        with mock.patch.object(app, "_place_alpha_from_roi",
                               side_effect=lambda a, r, w, h: np.zeros((h, w), np.float32)):
            payload = self._run(state, worker)
        qc_path = self.root / "run" / "matte_qc_s2m_base_plus.json"
        self.assertTrue(qc_path.is_file())
        summary = json.loads(qc_path.read_text())["summary"]
        self.assertEqual(summary["suspicious_ranges"], [[0, 2]])
        self.assertIn("suspicious", payload[app.UI_OUTPUT_STATUS_INDEX])


class BackendIsolationTests(unittest.TestCase):
    """Results from different backends must never be reused for one another."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.runtime = _stub_runtime(self.root)
        self.torch_patch = mock.patch.dict(
            sys.modules,
            {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))},
        )
        self.torch_patch.start()

    def tearDown(self):
        self.torch_patch.stop()
        self.tmpdir.cleanup()

    def _run_sam2matting(self, state, **kwargs):
        worker = _StubWorker()
        with mock.patch.object(client, "preflight", return_value=self.runtime), \
                mock.patch.object(client.subprocess, "Popen", side_effect=worker.popen):
            defaults = dict(
                chunk_size=4, overlap=0,
                matting_backend=mb.SAM2MATTING_BASE_PLUS.backend_id,
                progress=lambda *a, **k: None,
            )
            defaults.update(kwargs)
            return app.run_sequence(state, **defaults), worker

    def _run_videomama(self, state, **kwargs):
        def fake_videomama(pipeline, frames, masks, **_):
            return [np.full((*f.shape[:2], 3), 0.25, dtype=np.float32) for f in frames]

        with mock.patch.object(app, "_ensure_videomama_pipeline", return_value=object()), \
                mock.patch.object(app, "videomama", side_effect=fake_videomama):
            defaults = dict(
                chunk_size=4, overlap=0, matting_backend="videomama",
                progress=lambda *a, **k: None,
            )
            defaults.update(kwargs)
            return app.run_sequence(state, **defaults)

    def test_two_backends_write_to_separate_directories(self):
        state = _build_state(self.root, frame_count=3)
        (payload, _worker) = self._run_sam2matting(state)
        s2m_dir = Path(payload[4]["matte_output_dir"])
        payload = self._run_videomama(state)
        vm_dir = Path(payload[4]["matte_output_dir"])
        self.assertNotEqual(s2m_dir, vm_dir)
        self.assertTrue(s2m_dir.is_dir())
        self.assertTrue(vm_dir.is_dir())
        self.assertEqual(len(list(s2m_dir.glob("*.png"))), 3)
        self.assertEqual(len(list(vm_dir.glob("*.png"))), 3)

    def test_switching_backends_invalidates_the_cached_frame_records(self):
        state = _build_state(self.root, frame_count=3)
        self._run_sam2matting(state)
        s2m_hash = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self._run_videomama(state)
        vm_manifest = app._read_run_manifest(self.root / "run")["matte"]
        self.assertNotEqual(vm_manifest["settings_hash"], s2m_hash)
        # None of the SAM2Matting frames may be carried over as VideoMaMa results.
        for record in vm_manifest["frame_records"].values():
            self.assertEqual(record["settings_hash"], vm_manifest["settings_hash"])
            self.assertEqual(record["backend_id"], "videomama")

    def test_changing_a_sam2matting_parameter_invalidates_the_cache(self):
        state = _build_state(self.root, frame_count=3)
        self._run_sam2matting(state, s2m_guidance_interval=4)
        first = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self._run_sam2matting(state, s2m_guidance_interval=2)
        second = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self.assertNotEqual(first, second)

    def test_changing_the_roi_invalidates_the_cache(self):
        state = _build_state(self.root, frame_count=3)
        self._run_sam2matting(state)
        auto = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self._run_sam2matting(state, matte_roi_mode="Full frame")
        full = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self.assertNotEqual(auto, full)

    def test_a_different_upstream_revision_invalidates_the_cache(self):
        state = _build_state(self.root, frame_count=3)
        self._run_sam2matting(state)
        first = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self.runtime = dict(self.runtime, commit="d" * 40)
        self._run_sam2matting(state)
        second = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self.assertNotEqual(first, second)

    def test_repeating_an_identical_run_keeps_the_same_identity(self):
        state = _build_state(self.root, frame_count=3)
        self._run_sam2matting(state)
        first = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self._run_sam2matting(state)
        second = app._read_run_manifest(self.root / "run")["matte"]["settings_hash"]
        self.assertEqual(first, second)

    def test_the_videomama_path_still_writes_its_historical_directories(self):
        state = _build_state(self.root, frame_count=3)
        payload = self._run_videomama(state)
        run_root = self.root / "run"
        self.assertEqual(Path(payload[4]["matte_output_dir"]), run_root / "videomama_frames")
        self.assertEqual(Path(payload[4]["alpha_output_dir"]), run_root / "alpha_frames")
        # And it still mirrors to the legacy manifest section older runs read.
        manifest = app._read_run_manifest(run_root)
        self.assertEqual(manifest["videomama"]["backend_id"], "videomama")
        self.assertEqual(manifest["videomama"]["status"], "complete")

    def test_videomama_output_values_are_unchanged_by_the_refactor(self):
        state = _build_state(self.root, frame_count=3)
        payload = self._run_videomama(state, matte_roi_mode="Full frame")
        alpha = np.array(Image.open(Path(payload[4]["alpha_output_dir"]) / "frame_0000.png"))
        np.testing.assert_allclose(alpha / 65535.0, 0.25, atol=2e-5)


class TrackingModelSelectionTests(unittest.TestCase):
    """SAM 3 stays the tracker; the UI only chooses which released weights."""

    def setUp(self):
        self.previous = os.environ.get("SAM3_MODEL_VERSION")

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("SAM3_MODEL_VERSION", None)
        else:
            os.environ["SAM3_MODEL_VERSION"] = self.previous

    def test_switching_to_sam31_drops_the_loaded_tracker(self):
        os.environ["SAM3_MODEL_VERSION"] = "sam3"
        with mock.patch.object(app, "_unload_sam3_tracker") as unload:
            message, _ = app.set_tracking_model("sam3.1")
        self.assertEqual(os.environ["SAM3_MODEL_VERSION"], "sam3.1")
        unload.assert_called_once()
        self.assertIn("regenerated", message)

    def test_reselecting_the_same_model_is_a_no_op(self):
        os.environ["SAM3_MODEL_VERSION"] = "sam3.1"
        with mock.patch.object(app, "_unload_sam3_tracker") as unload:
            message, _ = app.set_tracking_model("sam3.1")
        unload.assert_not_called()
        self.assertNotIn("regenerated", message)

    def test_an_unknown_tracking_model_is_rejected(self):
        with self.assertRaises(Exception):
            app.set_tracking_model("sam4")

    def test_the_tracking_model_is_part_of_the_sam_mask_identity(self):
        state = {
            "prompt_mode": "Point keyframes",
            "prompts_by_frame": {"0": {"points": [[1, 1]], "labels": [1]}},
            "current_frame_idx": 0,
            "frame_names": ["a.png"],
            "work_size": [64, 64],
            "sam_crop_settings": app._sam_crop_settings(),
            "sam_refine_edges_against_plate": False,
        }
        os.environ["SAM3_MODEL_VERSION"] = "sam3"
        first = app._prompt_manifest_payload(state, 0.5)
        os.environ["SAM3_MODEL_VERSION"] = "sam3.1"
        second = app._prompt_manifest_payload(state, 0.5)
        self.assertEqual(first["model_version"], "sam3")
        self.assertEqual(second["model_version"], "sam3.1")


class MatteViewerTests(unittest.TestCase):
    """The preview must say which backend produced what is on screen.

    The viewer was originally hard-labelled "VideoMaMa", so a SAM2Matting result
    rendered under a VideoMaMa heading and looked like it had never been made.
    """

    def test_the_tab_and_default_label_are_backend_agnostic(self):
        tabs = [child.label for child in app.demo.blocks.values() if isinstance(child, app.gr.Tab)]
        self.assertIn("Matte", tabs)
        self.assertNotIn("VideoMaMa", tabs)
        self.assertNotIn("VideoMaMa", app.output_img.label)

    def test_the_label_names_the_backend_that_produced_the_matte(self):
        state = {"matte_output_dir": "/x", "matte_backend_id": "sam2matting-sam2.1base+"}
        self.assertEqual(
            app._matte_viewer_label(state),
            f"Current Matte - {mb.SAM2MATTING_BASE_PLUS.label}",
        )
        state = {"matte_output_dir": "/x", "matte_backend_id": "videomama"}
        self.assertIn("VideoMaMa", app._matte_viewer_label(state))

    def test_the_label_says_so_when_nothing_has_been_generated(self):
        self.assertIn("none generated yet", app._matte_viewer_label({}))
        self.assertIn("none generated yet", app._matte_viewer_label(None))

    def test_an_unknown_backend_id_does_not_break_the_label(self):
        label = app._matte_viewer_label({"matte_output_dir": "/x", "matte_backend_id": "gone"})
        self.assertIn("gone", label)

    def test_the_preview_resolves_against_whichever_backend_wrote_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "s2m").mkdir()
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "s2m" / "f.png")
            state = {"matte_output_dir": str(root / "s2m"), "frame_names": ["f.exr"]}
            self.assertEqual(app._matte_output_path(state, 0), root / "s2m" / "f.png")
            # Runs from before backend selection only have the legacy key.
            legacy = {"videomama_output_dir": str(root / "s2m"), "frame_names": ["f.exr"]}
            self.assertEqual(app._matte_output_path(legacy, 0), root / "s2m" / "f.png")
            self.assertIsNone(app._matte_output_path({"frame_names": ["f.exr"]}, 0))

    def test_frame_info_reports_the_backend(self):
        state = {
            "current_frame_idx": 0,
            "frame_paths": ["a"],
            "frame_names": ["a.exr"],
            "frame_transforms": [{}],
            "matte_output_dir": "/x",
            "matte_backend_id": "sam2matting-sam2.1base+",
        }
        self.assertIn("SAM2Matting SAM2.1 Base+", app._frame_info(state))
        state.pop("matte_output_dir")
        self.assertNotIn("matte:", app._frame_info(state))

    def test_the_ui_payload_updates_the_viewer_label(self):
        state = {
            "current_frame_idx": 0,
            "frame_paths": ["a"],
            "frame_names": ["a.exr"],
            "frame_transforms": [{}],
            "cache_frame_paths": ["a"],
            "prompts_by_frame": {},
            "preview_masks": {},
            "matte_backend_id": "sam2matting-sam2.1base+",
            "matte_output_dir": "/x",
        }
        with mock.patch.object(app, "_render_frame_preview", return_value=(None, None, None)):
            payload = app._ui_state_payload(state, "done")
        self.assertEqual(len(payload), app.UI_OUTPUT_COUNT)
        self.assertIn("SAM2Matting SAM2.1 Base+", payload[3]["label"])
