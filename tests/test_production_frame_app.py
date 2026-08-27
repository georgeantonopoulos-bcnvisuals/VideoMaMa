import os
import sys
import tempfile
import unittest
import contextlib
import types
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import production_frame_app as app  # noqa: E402
import sam3_wrapper_hf  # noqa: E402
import videomama_wrapper  # noqa: E402


def _write_sequence(sequence_dir: Path, count=2):
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = idx * 40
        frame[:, :, 1] = 120
        Image.fromarray(frame).save(sequence_dir / f"frame_{idx:04d}.png")


class ClearSequenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.app_tmp = self.tmp_path / "production_sequence_app"
        self.app_tmp.mkdir()
        self.patches = [
            mock.patch.object(app, "APP_TMP_ROOT", self.app_tmp),
            mock.patch.object(app, "SETTINGS_PATH", self.app_tmp / "ui_settings.json"),
        ]
        for patcher in self.patches:
            patcher.start()
        with app._DEBUG_LOCK:
            app._DEBUG_LINES.clear()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tmpdir.cleanup()

    def test_deletes_current_run_and_returns_full_ui_payload(self):
        sequence_dir = self.tmp_path / "sequence"
        _write_sequence(sequence_dir)
        stale_run = self.app_tmp / "stale_sequence_run"
        stale_run.mkdir()
        (stale_run / "marker.txt").write_text("stale", encoding="utf-8")

        payload = app.clear_sequence_cache_and_reload(
            {"run_root": str(stale_run)},
            str(sequence_dir),
            1.0,
            0.0,
            "Point keyframes",
            "",
        )

        self.assertFalse(stale_run.exists())
        self.assertEqual(len(payload), app.UI_OUTPUT_COUNT)
        self.assertIn("Cleared cache run(s): stale_sequence_run", payload[app.UI_OUTPUT_STATUS_INDEX])
        self.assertIn("Cleared cache run(s): stale_sequence_run", payload[app.UI_OUTPUT_DEBUG_INDEX])
        self.assertEqual(payload[4]["sequence_dir"], str(sequence_dir.resolve()))
        self.assertTrue(Path(payload[4]["run_root"]).exists())

    def test_does_not_delete_outside_app_tmp(self):
        sequence_dir = self.tmp_path / "sequence"
        _write_sequence(sequence_dir)
        outside_run = self.tmp_path / "outside_run"
        outside_run.mkdir()
        (outside_run / "marker.txt").write_text("keep", encoding="utf-8")

        payload = app.clear_sequence_cache_and_reload(
            {"run_root": str(outside_run)},
            str(sequence_dir),
            1.0,
            0.0,
            "Point keyframes",
            "",
        )

        self.assertTrue(outside_run.exists())
        self.assertTrue((outside_run / "marker.txt").exists())
        self.assertEqual(len(payload), app.UI_OUTPUT_COUNT)
        self.assertIn("No prior cache run was found; loaded fresh.", payload[app.UI_OUTPUT_STATUS_INDEX])

    def test_directory_not_empty_cleanup_error_does_not_block_reload(self):
        sequence_dir = self.tmp_path / "sequence"
        _write_sequence(sequence_dir)
        stale_run = self.app_tmp / "shot_040_foot"
        stale_run.mkdir()
        marker = stale_run / "marker.txt"
        marker.write_text("stale", encoding="utf-8")

        with mock.patch.object(app.shutil, "rmtree", side_effect=OSError(39, "Directory not empty")):
            payload = app.clear_sequence_cache_and_reload(
                {"run_root": str(stale_run)},
                str(sequence_dir),
                1.0,
                0.0,
                "Point keyframes",
                "",
            )

        self.assertFalse(stale_run.exists())
        self.assertEqual(len(payload), app.UI_OUTPUT_COUNT)
        self.assertIn("Cleared cache run(s): shot_040_foot", payload[app.UI_OUTPUT_STATUS_INDEX])
        self.assertIn("Deferred cleanup warning", payload[app.UI_OUTPUT_STATUS_INDEX])
        self.assertTrue(any(self.app_tmp.glob(f"{app.CLEARED_RUN_PREFIX}_*_shot_040_foot")))
        self.assertFalse(marker.exists())
        self.assertTrue(Path(payload[4]["run_root"]).exists())

    def test_rename_permission_error_falls_back_to_clearing_contents_in_place(self):
        sequence_dir = self.tmp_path / "sequence"
        _write_sequence(sequence_dir)
        stale_run = self.app_tmp / "shot_040_foot"
        stale_run.mkdir()
        nested = stale_run / "sam3_frames"
        nested.mkdir()
        marker = nested / "marker.txt"
        marker.write_text("stale", encoding="utf-8")

        with mock.patch("pathlib.Path.rename", side_effect=PermissionError(13, "Permission denied")):
            payload = app.clear_sequence_cache_and_reload(
                {"run_root": str(stale_run)},
                str(sequence_dir),
                1.0,
                0.0,
                "Point keyframes",
                "",
            )

        self.assertFalse(stale_run.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(len(payload), app.UI_OUTPUT_COUNT)
        self.assertIn("Cleared cache run(s): shot_040_foot", payload[app.UI_OUTPUT_STATUS_INDEX])
        self.assertIn("cleared contents in place", payload[app.UI_OUTPUT_STATUS_INDEX])


class ProductionUtilityTests(unittest.TestCase):
    def test_videomama_guide_source_resolves_sam3_and_sam2matting(self):
        self.assertIsNone(app._videomama_guide_spec(app.VIDEOMAMA_GUIDE_SAM3))
        self.assertIs(
            app._videomama_guide_spec(app.mb.SAM2MATTING_BASE_PLUS.label),
            app.mb.SAM2MATTING_BASE_PLUS,
        )
        with self.assertRaises(ValueError):
            app._videomama_guide_spec('not a guide source')

    def test_model_specific_run_buttons_select_the_expected_backend(self):
        backend_update, backend_info = app._select_sam2matting_backend(
            app.mb.VIDEOMAMA_BACKEND.label
        )
        self.assertEqual(backend_update['value'], app.mb.SAM2MATTING_BASE_PLUS.label)
        self.assertIn(app.mb.SAM2MATTING_BASE_PLUS.label, backend_info)

        backend_update, backend_info = app._select_sam2matting_backend(
            app.mb.SAM2MATTING_TINY.label
        )
        self.assertEqual(backend_update['value'], app.mb.SAM2MATTING_TINY.label)
        self.assertIn(app.mb.SAM2MATTING_TINY.label, backend_info)

        backend_update, backend_info = app._select_videomama_backend()
        self.assertEqual(backend_update['value'], app.mb.VIDEOMAMA_BACKEND.label)
        self.assertIn(app.mb.VIDEOMAMA_BACKEND.label, backend_info)

    def test_model_specific_run_buttons_are_exposed(self):
        self.assertEqual(
            app.generate_sam2matting_btn.value,
            'Generate SAM2Matting Matte from SAM 3 Masks',
        )
        self.assertEqual(
            app.generate_videomama_btn.value,
            'Generate VideoMaMa Matte (Selected Range)',
        )
        self.assertEqual(
            app.run_btn.value,
            'Generate Matte Using Backend Above',
        )
        self.assertIn(
            app.mb.SAM2MATTING_BASE_PLUS.label,
            [choice[0] for choice in app.videomama_guide_source.choices],
        )

    def test_sam_preview_controls_are_bounded(self):
        self.assertEqual(app._sam_preview_height(100), 360)
        self.assertEqual(app._sam_preview_height(2000), 1200)
        self.assertEqual(app._sam_overlay_opacity(-1), 0.0)
        self.assertEqual(app._sam_overlay_opacity(2), 1.0)
        updates = app.resize_sam_previews(840)
        self.assertEqual(len(updates), 5)
        self.assertTrue(all(update["height"] == 840 for update in updates))

    def test_maximum_detail_preset_controls_sam_and_videomama_quality(self):
        updates = app.apply_quality_preset('Maximum Detail')
        self.assertEqual(len(updates), 9)
        self.assertEqual(updates[0]['value'], 0.35)
        self.assertEqual(updates[1]['value'], 'vae')
        # These two presets now select Auto rather than pinning a fixed 16:9
        # canvas: picking a max-quality preset must not drop the artist off
        # aspect-matched, VRAM-aware sizing back onto a stretched canvas.
        self.assertEqual(updates[5]['value'], app.AUTO_PROCESSING_RESOLUTION)
        self.assertTrue(updates[6]['value'])
        self.assertEqual(updates[7]['value'], 0)
        self.assertTrue(updates[8]['value'])
        self.assertEqual(
            app.quality_preset.label,
            'Combined SAM 3 + VideoMaMa Quality Preset',
        )

    def test_hair_detail_preset_preserves_thin_boundaries(self):
        updates = app.apply_quality_preset('Hair Detail')

        self.assertEqual(len(updates), 9)
        self.assertEqual(updates[0]['value'], 0.2)
        # These two presets now select Auto rather than pinning a fixed 16:9
        # canvas: picking a max-quality preset must not drop the artist off
        # aspect-matched, VRAM-aware sizing back onto a stretched canvas.
        self.assertEqual(updates[5]['value'], app.AUTO_PROCESSING_RESOLUTION)
        self.assertFalse(updates[6]['value'])
        self.assertEqual(updates[7]['value'], 8)
        self.assertFalse(updates[8]['value'])

    def test_videomama_guide_margin_adds_context_without_mutating_sam_mask(self):
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[7, 7] = 255

        unchanged = app._expand_videomama_guide(mask, 0)
        expanded = app._expand_videomama_guide(mask, 2)

        np.testing.assert_array_equal(unchanged, mask)
        self.assertIsNot(unchanged, mask)
        self.assertGreater(int((expanded > 0).sum()), 1)
        self.assertEqual(expanded[7, 7], 255)
        self.assertEqual(int((mask > 0).sum()), 1)

    def test_sam_mask_inspector_has_real_zoom_controls(self):
        self.assertEqual(app.mask_img.elem_id, "sam_mask_zoom")
        self.assertEqual(app.preview_large.elem_id, "sam_overlay_zoom")
        self.assertIn('rootIds = ["sam_mask_zoom", "sam_overlay_zoom"]', app.APP_JS)
        self.assertIn("sam-zoom-toolbar", app.APP_JS)
        self.assertIn('addEventListener("wheel"', app.APP_JS)
        self.assertIn('addEventListener("pointermove"', app.APP_JS)
        self.assertIn("maxZoom = 12.0", app.APP_JS)
        self.assertIn("imageWasAdded", app.APP_JS)
        self.assertIn("readout.textContent !== label", app.APP_JS)

    def test_zoomed_overlay_maps_clicks_to_sam_prompt_coordinates(self):
        self.assertEqual(app.overlay_point_x.elem_id, "sam_overlay_point_x")
        self.assertEqual(app.overlay_point_y.elem_id, "sam_overlay_point_y")
        self.assertEqual(app.overlay_point_submit.elem_id, "sam_overlay_point_submit")
        self.assertIn('rootId !== "sam_overlay_zoom"', app.APP_JS)
        self.assertIn("img.naturalWidth", app.APP_JS)
        self.assertIn("contentLeft", app.APP_JS)
        self.assertIn('setBridgeNumber("sam_overlay_point_x", x)', app.APP_JS)
        self.assertIn('querySelector("#sam_overlay_point_submit")?.click()', app.APP_JS)
        self.assertNotIn('#sam_overlay_point_submit button', app.APP_JS)

    def test_mask_inspector_uses_source_resolution_generated_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_dir = Path(tmpdir)
            source_mask = np.zeros((4, 8), dtype=np.uint8)
            source_mask[:, 2:6] = 255
            Image.fromarray(source_mask).save(mask_dir / "frame_0001.png")
            state = {
                "generated_masks_dir": str(mask_dir),
                "frame_names": ["frame_0001.exr"],
                "frame_transforms": [app._letterbox_transform(8, 4, work_size=(4, 4))],
                "preview_masks": {},
            }

            work_mask = app._load_current_mask(state, 0)
            inspection_mask = app._load_current_mask_for_inspection(state, 0)
            self.assertEqual(work_mask.shape, (4, 4))
            self.assertEqual(inspection_mask.shape, source_mask.shape)

    def test_sam_plate_refinement_keeps_binary_mask_and_moves_boundary(self):
        plate = np.zeros((64, 64, 3), dtype=np.uint8)
        plate[:, 32:] = 255
        raw_mask = np.zeros((64, 64), dtype=np.uint8)
        raw_mask[:, 27:] = 255

        refined = app._refine_sam_mask_against_plate(plate, raw_mask)

        self.assertEqual(refined.shape, raw_mask.shape)
        self.assertEqual(set(np.unique(refined)), {0, 255})
        self.assertLess(int((refined > 0).sum()), int((raw_mask > 0).sum()))
        self.assertTrue(np.all(refined[:, 32:] == 255))

    def test_sam_plate_refinement_keeps_artist_point_constraints(self):
        plate = np.zeros((40, 60, 3), dtype=np.uint8)
        raw_mask = np.zeros((40, 60), dtype=np.uint8)
        raw_mask[10:30, 15:45] = 255

        refined = app._refine_sam_mask_against_plate(
            plate,
            raw_mask,
            points=[[5, 5], [30, 20]],
            point_labels=[1, 0],
        )

        self.assertEqual(refined[5, 5], 255)
        self.assertEqual(refined[20, 30], 0)

    def test_natural_sequence_sort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(root / name)
            self.assertEqual(
                [path.name for path in app._discover_sequence_files(str(root))],
                ["frame_1.png", "frame_2.png", "frame_10.png"],
            )

    def test_duplicate_output_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(root / "frame_0001.png")
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(root / "frame_0001.jpg")
            with self.assertRaises(Exception):
                app._discover_sequence_files(str(root))

    def test_letterbox_mask_round_trip_preserves_source_shape(self):
        transform = app._letterbox_transform(13, 7, work_size=(32, 18))
        source = np.zeros((7, 13), dtype=np.uint8)
        source[2:5, 4:9] = 255
        work = app._source_mask_to_work(source, transform)
        restored = app._work_mask_to_source(work, transform)
        self.assertEqual(work.shape, (18, 32))
        self.assertEqual(restored.shape, source.shape)
        self.assertGreater(int(restored.sum()), 0)

    def test_sam_source_crop_expands_roi_and_restores_full_frame_mask(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[20:60, 30:90] = (25, 100, 225)
        crop_settings = app._sam_crop_settings(True, 30, 20, 60, 40)

        working, transform = app._letterbox_rgb_frame(
            frame, work_size=(90, 60), crop_settings=crop_settings
        )

        self.assertEqual(working.shape, (60, 90, 3))
        self.assertEqual(transform['crop_x'], 30)
        self.assertEqual(transform['crop_y'], 20)
        self.assertEqual(transform['crop_width'], 60)
        self.assertEqual(transform['crop_height'], 40)
        self.assertTrue(np.all(working == (25, 100, 225)))

        source_mask = app._work_mask_to_source(np.full((60, 90), 255, dtype=np.uint8), transform)
        self.assertEqual(source_mask.shape, (80, 120))
        self.assertTrue(np.all(source_mask[20:60, 30:90] == 255))
        self.assertEqual(int(source_mask[:20].sum()), 0)
        self.assertEqual(int(source_mask[:, :30].sum()), 0)
        self.assertEqual(int(source_mask[60:].sum()), 0)
        self.assertEqual(int(source_mask[:, 90:].sum()), 0)

        round_trip = app._source_mask_to_work(source_mask, transform)
        self.assertEqual(round_trip.shape, (60, 90))
        self.assertTrue(np.all(round_trip == 255))

    def test_artist_can_select_sam_crop_with_two_full_resolution_clicks(self):
        self.assertEqual(app.crop_selector_btn.value, '1. Load Full-Resolution Frame')
        self.assertEqual(app.apply_crop_btn.value, '2. Apply Selected Crop + Load Sequence')
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = np.zeros((80, 120, 3), dtype=np.uint8)
            Image.fromarray(source).save(root / 'frame_0001.png')
            preview, selector_state, message = app.load_sam_crop_selector(
                str(root), 1.0, 0.0, 'Gamma / Exposure', 'scene_linear', '', ''
            )
            self.assertEqual(preview.shape, source.shape)
            self.assertIn('120x80', message)

            first = app.select_sam_crop_roi(selector_state, mock.Mock(index=(20, 15)))
            self.assertIn('First corner', first[-1])
            second = app.select_sam_crop_roi(first[1], mock.Mock(index=(100, 70)))
            self.assertEqual(second[2:7], (True, 20, 15, 80, 55))
            self.assertIn('SAM source ROI selected', second[-1])

    def test_point_prompt_is_recorded_on_cropped_sam_canvas(self):
        crop_settings = app._sam_crop_settings(True, 960, 540, 1920, 1080)
        transform = app._sam_crop_transform(
            3840, 2160, work_size=(1536, 864), crop_settings=crop_settings
        )
        state = {
            'current_frame_idx': 0,
            'prompts_by_frame': {},
            'frame_transforms': [transform],
            'work_size': [1536, 864],
        }
        with mock.patch.object(app, 'sam3_tracker', object()), \
                mock.patch.object(app, '_compute_preview_mask'), \
                mock.patch.object(app, '_invalidate_generated_results'), \
                mock.patch.object(app, '_ui_state_payload', return_value=('updated',)):
            result = app._add_point_at_coordinates(
                state, 'Point keyframes', 'Positive', 0.35, 768, 432
            )

        self.assertEqual(result, ('updated',))
        self.assertEqual(state['prompts_by_frame']['0']['points'], [[768, 432]])
        self.assertEqual(state['prompts_by_frame']['0']['labels'], [1])

    def test_overlap_blend_uses_smooth_crossfade(self):
        previous = np.zeros((2, 2), dtype=np.float32)
        current = np.ones((2, 2), dtype=np.float32)
        first = app._blend_overlap_alpha(previous, current, 0, 2)
        second = app._blend_overlap_alpha(previous, current, 1, 2)
        np.testing.assert_allclose(first, 1.0 / 3.0)
        np.testing.assert_allclose(second, 2.0 / 3.0)

    def test_viewer_defaults_read_exrs_as_aces_scene_linear(self):
        self.assertEqual(app.DEFAULT_EXR_COLOR_MODE, "OCIO Display")
        self.assertEqual(app.DEFAULT_OCIO_INPUT_COLORSPACE, "scene_linear")
        self.assertEqual(app.DEFAULT_SETTINGS["exr_color_mode"], "OCIO Display")
        self.assertEqual(app.DEFAULT_SETTINGS["ocio_input_colorspace"], "scene_linear")

    def test_scene_linear_resolves_to_an_aces_colorspace(self):
        config, source = app._resolve_ocio_config("scene_linear")
        resolved = config.getColorSpace("scene_linear")
        self.assertIsNotNone(resolved, f"scene_linear unresolved in config from {source}")
        self.assertIn("ACEScg", resolved.getName())

    def test_ocio_display_applies_the_aces_view_transform(self):
        # 0.18 scene-linear grey lands at ~0.3565 through the ACES sRGB output
        # transform. A passthrough config would leave it at 0.18, so this is the
        # assertion that fails if OCIO silently degrades to "color management
        # disabled" the way it did when $OCIO was unset.
        grey = np.full((1, 1, 3), 0.18, dtype=np.float32)
        transformed = app._apply_ocio_display(grey)
        np.testing.assert_allclose(transformed, 0.3565, atol=2e-3)

    def test_ocio_resolver_rejects_a_config_without_the_input_colorspace(self):
        with self.assertRaises(RuntimeError) as caught:
            app._resolve_ocio_config("definitely_not_a_colorspace")
        self.assertIn("definitely_not_a_colorspace", str(caught.exception))

    def test_ocio_resolver_falls_back_to_the_builtin_aces_config(self):
        # Machines without the production mount must still get ACES, not raw.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIDEOMAMA_OCIO_CONFIG", None)
            os.environ.pop("OCIO", None)
            with mock.patch.object(app, "ACES_OCIO_CONFIG_CANDIDATES", ("/nonexistent/config.ocio",)), \
                    mock.patch.dict(app._OCIO_CONFIG_CACHE, {}, clear=True):
                config, source = app._resolve_ocio_config("scene_linear")
        self.assertEqual(source, app.ACES_OCIO_BUILTIN_CONFIG)
        self.assertIsNotNone(config.getColorSpace("scene_linear"))

    def test_exr_read_uses_aces_and_rolls_off_highlights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plate.exr"
            header = app.OpenEXR.Header(3, 1)
            float_channel = app.Imath.Channel(app.Imath.PixelType(app.Imath.PixelType.FLOAT))
            header["channels"] = {name: float_channel for name in ("R", "G", "B")}
            values = np.array([0.18, 1.0, 4.0], dtype=np.float32)
            output = app.OpenEXR.OutputFile(str(path), header)
            output.writePixels({name: values.tobytes() for name in ("R", "G", "B")})
            output.close()

            aces = app._read_exr_rgb(str(path))[0, :, 0]
            legacy = app._read_exr_rgb(
                str(path), exr_gamma=1.0, exr_color_mode="Gamma / Exposure"
            )[0, :, 0]

        # Mid grey is lifted by the tone curve rather than passed through.
        self.assertEqual(int(aces[0]), 91)
        # The legacy raw-linear path clips everything at and above 1.0 to white;
        # the ACES view keeps 1.0 and 4.0 distinguishable.
        self.assertEqual(list(legacy[1:]), [255, 255])
        self.assertLess(int(aces[1]), int(aces[2]))
        self.assertLess(int(aces[2]), 255)

    def test_legacy_color_settings_do_not_match_the_aces_defaults(self):
        # Otherwise a pre-ACES run cache would be reused with the old colour
        # baked into its JPEGs instead of being rebuilt.
        current = {
            "mode": app.DEFAULT_EXR_COLOR_MODE,
            "input_colorspace": app.DEFAULT_OCIO_INPUT_COLORSPACE,
            "display": app.DEFAULT_OCIO_DISPLAY,
            "view": app.DEFAULT_OCIO_VIEW,
        }
        self.assertNotEqual(dict(app.LEGACY_EXR_COLOR_SETTINGS), current)

    def test_exr_exposure_and_gamma_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plate.exr"
            header = app.OpenEXR.Header(2, 1)
            float_channel = app.Imath.Channel(app.Imath.PixelType(app.Imath.PixelType.FLOAT))
            header["channels"] = {name: float_channel for name in ("R", "G", "B")}
            values = np.array([0.25, 0.5], dtype=np.float32)
            output = app.OpenEXR.OutputFile(str(path), header)
            output.writePixels({name: values.tobytes() for name in ("R", "G", "B")})
            output.close()

            converted = app._read_exr_rgb(
                str(path),
                exr_gamma=1.0,
                exr_exposure=1.0,
                exr_color_mode="Gamma / Exposure",
            )
            np.testing.assert_array_equal(converted[0, :, 0], np.array([128, 255], dtype=np.uint8))

    def test_atomic_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app._update_run_manifest(root, "sam3", {"status": "complete", "frame_count": 2})
            manifest = app._read_run_manifest(root)
            self.assertEqual(manifest["schema_version"], app.RUN_MANIFEST_SCHEMA)
            self.assertEqual(manifest["sam3"]["frame_count"], 2)

    def test_sequence_completeness_checks_every_frame_and_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            names = ["a.exr", "b.exr"]
            sizes = [(4, 3), (4, 3)]
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(root / "a.png")
            self.assertFalse(app._sequence_outputs_complete(root, names, sizes))
            Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(root / "b.png")
            self.assertTrue(app._sequence_outputs_complete(root, names, sizes))

    def test_matte_writer_preserves_sixteen_bit_alpha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alpha = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
            app._save_matte_outputs(root / "preview", root / "alpha", "frame.exr", alpha, "16-bit PNG")
            saved = np.array(Image.open(root / "alpha" / "frame.png"))
            self.assertGreater(int(saved.max()), 255)
            self.assertAlmostEqual(int(saved[0, 1]), 32768, delta=1)

    def test_half_exr_alpha_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alpha = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
            app._save_matte_outputs(root / "preview", root / "alpha", "frame.exr", alpha, "Half EXR")
            exr = app.OpenEXR.InputFile(str(root / "alpha" / "frame.exr"))
            try:
                raw = exr.channel("A", app.Imath.PixelType(app.Imath.PixelType.FLOAT))
            finally:
                exr.close()
            recovered = np.frombuffer(raw, dtype=np.float32).reshape(alpha.shape)
            np.testing.assert_allclose(recovered, alpha, atol=1e-3)


class WrapperContractTests(unittest.TestCase):
    def test_sam_preview_uses_exact_cached_jpeg_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / 'cached.jpg'
            Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(source_path, quality=91)
            source_bytes = source_path.read_bytes()

            tracker = sam3_wrapper_hf.SAM3VideoTracker.__new__(sam3_wrapper_hf.SAM3VideoTracker)

            def fake_start_session(frames_dir):
                self.assertEqual((Path(frames_dir) / '00000.jpg').read_bytes(), source_bytes)
                return 'session'

            tracker._start_session = fake_start_session
            tracker._close_session = lambda session_id: None
            tracker._add_prompt = lambda *args, **kwargs: {
                'outputs': {
                    'out_obj_ids': [1],
                    'out_binary_masks': [np.ones((4, 6), dtype=bool)],
                }
            }

            mask = tracker.get_frame_mask_from_path(
                str(source_path), [[1, 1]], [1], output_prob_thresh=0.35
            )
            self.assertTrue(np.all(mask == 255))

    def test_sam31_builder_does_not_receive_legacy_gpu_ids(self):
        calls = []
        model_builder = types.ModuleType("sam3.model_builder")
        model_builder.build_sam3_predictor = lambda **kwargs: calls.append(kwargs) or object()
        sam3_module = types.ModuleType("sam3")
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 0,
            )
        )
        with mock.patch.dict(
            sys.modules,
            {"sam3": sam3_module, "sam3.model_builder": model_builder, "torch": fake_torch},
        ), mock.patch.object(sam3_wrapper_hf, "_assert_hf_sam3_access"):
            sam3_wrapper_hf.SAM3VideoTracker(model_version="sam3.1")

        self.assertEqual(calls, [{"use_fa3": False, "version": "sam3.1"}])

    def test_sam31_session_shim_omits_unsupported_state_offload(self):
        init_calls = []

        class Model:
            def init_state(self, **kwargs):
                init_calls.append(kwargs)
                return {"ready": True}

        tracker = sam3_wrapper_hf.SAM3VideoTracker.__new__(sam3_wrapper_hf.SAM3VideoTracker)
        tracker.model_version = "sam3.1"
        tracker.predictor = types.SimpleNamespace(
            model=Model(),
            async_loading_frames=True,
            _all_inference_states={},
        )
        session_id = tracker._start_session("/tmp/frames")

        self.assertIn(session_id, tracker.predictor._all_inference_states)
        self.assertEqual(
            init_calls,
            [{
                "resource_path": "/tmp/frames",
                "offload_video_to_cpu": False,
                "async_loading_frames": True,
            }],
        )

    def test_videomama_pads_1536x864_for_unet_and_crops_output(self):
        class RecordingPipeline:
            def __init__(self):
                self.cond_sizes = None
                self.mask_sizes = None

            def run(self, **kwargs):
                self.cond_sizes = [frame.size for frame in kwargs["cond_frames"]]
                self.mask_sizes = [frame.size for frame in kwargs["mask_frames"]]
                height = kwargs["cond_frames"][0].height
                width = kwargs["cond_frames"][0].width
                # A vertical gradient makes the centered crop observable.
                gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
                return [np.broadcast_to(gradient, (height, width, 3)).copy()]

        pipeline = RecordingPipeline()
        frames = [np.zeros((9, 16, 3), dtype=np.uint8)]
        masks = [np.full((9, 16), 255, dtype=np.uint8)]
        result = videomama_wrapper.videomama(
            pipeline,
            frames,
            masks,
            target_size=(1536, 864),
        )

        self.assertEqual(pipeline.cond_sizes, [(1536, 896)])
        self.assertEqual(pipeline.mask_sizes, [(1536, 896)])
        self.assertEqual(result[0].shape, (9, 16, 3))
        # Cropping 16 pixels from each side removes the padded extremes.
        self.assertGreater(float(result[0][0].mean()), 0.0)
        self.assertLess(float(result[0][-1].mean()), 1.0)

    def test_model_padding_leaves_native_resolution_unchanged(self):
        self.assertEqual(videomama_wrapper._model_padding((1024, 576)), (0, 0, 0, 0))

    def test_videomama_rejects_partial_pipeline_output(self):
        class PartialPipeline:
            def run(self, **kwargs):
                return [np.zeros((4, 6, 3), dtype=np.float32)]

        frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(2)]
        masks = [np.zeros((4, 6), dtype=np.uint8) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "returned 1 frames for 2 inputs"):
            videomama_wrapper.videomama(PartialPipeline(), frames, masks, target_size=(6, 4))

    def test_sam_streaming_callback_avoids_collecting_frame_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index in range(3):
                Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(root / f"{index:05d}.jpg")

            tracker = sam3_wrapper_hf.SAM3VideoTracker.__new__(sam3_wrapper_hf.SAM3VideoTracker)
            tracker._start_session = lambda frames_dir: "session"
            tracker._close_session = lambda session_id: None
            tracker._add_prompt = lambda *args, **kwargs: {
                "outputs": {"out_obj_ids": [1], "out_binary_masks": [np.ones((4, 6), dtype=bool)]}
            }

            class Predictor:
                def handle_stream_request(self, request):
                    for frame_index in range(3):
                        yield {
                            "frame_index": frame_index,
                            "outputs": {
                                "out_obj_ids": [1],
                                "out_binary_masks": [np.ones((4, 6), dtype=bool)],
                            },
                        }

            tracker.predictor = Predictor()
            fake_torch = types.SimpleNamespace(
                autocast=lambda **kwargs: contextlib.nullcontext(),
                bfloat16=object(),
            )
            callbacks = {}
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                result = tracker.track_video_from_dir(
                    str(root),
                    {0: {"points": [[1, 1]], "labels": [1]}},
                    mask_callback=lambda index, mask: callbacks.__setitem__(index, mask.copy()),
                    collect_masks=False,
                )
            self.assertEqual(set(callbacks), {0, 1, 2})
            self.assertEqual(result, [None, None, None])

    def test_sam_streaming_never_overwrites_exact_keyframe_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index in range(4):
                Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(root / f"{index:05d}.jpg")

            tracker = sam3_wrapper_hf.SAM3VideoTracker.__new__(sam3_wrapper_hf.SAM3VideoTracker)
            tracker._start_session = lambda frames_dir: "session"
            tracker._close_session = lambda session_id: None

            exact_masks = {
                1: np.pad(np.ones((2, 2), dtype=bool), ((0, 2), (0, 4))),
                3: np.pad(np.ones((2, 2), dtype=bool), ((2, 0), (4, 0))),
            }
            shared_output_buffer = np.zeros((4, 6), dtype=bool)

            def fake_add_prompt(session_id, frame_idx, *args, **kwargs):
                # Match SAM 3's real behavior: later prompt calls can reuse and
                # mutate the same output storage returned for earlier frames.
                shared_output_buffer[:] = exact_masks[frame_idx]
                return {
                    "outputs": {
                        "out_obj_ids": [1],
                        "out_binary_masks": [shared_output_buffer],
                    }
                }

            tracker._add_prompt = fake_add_prompt

            class Predictor:
                def handle_stream_request(self, request):
                    for frame_index in range(4):
                        yield {
                            "frame_index": frame_index,
                            "outputs": {
                                "out_obj_ids": [1],
                                "out_binary_masks": [np.zeros((4, 6), dtype=bool)],
                            },
                        }

            tracker.predictor = Predictor()
            fake_torch = types.SimpleNamespace(
                autocast=lambda **kwargs: contextlib.nullcontext(),
                bfloat16=object(),
            )
            callbacks = {}
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                tracker.track_video_from_dir(
                    str(root),
                    {
                        1: {"points": [[1, 1]], "labels": [1]},
                        3: {"points": [[5, 3]], "labels": [1]},
                    },
                    mask_callback=lambda index, mask: callbacks.__setitem__(index, mask.copy()),
                    collect_masks=False,
                )

            np.testing.assert_array_equal(callbacks[1] > 0, exact_masks[1])
            np.testing.assert_array_equal(callbacks[3] > 0, exact_masks[3])
            drift = tracker.last_tracking_stats['propagated_keyframe_drift_pixels']
            self.assertGreater(drift[1], 0)
            self.assertGreater(drift[3], 0)

    def test_run_sequence_crossfades_overlap_and_records_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence = root / "sequence"
            masks_dir = root / "run" / "sam3_masks"
            sequence.mkdir()
            masks_dir.mkdir(parents=True)
            frame_paths = []
            frame_names = []
            for index in range(6):
                name = f"frame_{index:04d}.png"
                path = sequence / name
                Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(path)
                Image.fromarray(np.full((4, 6), 255, dtype=np.uint8)).save(masks_dir / name)
                frame_paths.append(str(path))
                frame_names.append(name)

            state = {
                "sequence_dir": str(sequence),
                "frame_paths": frame_paths,
                "frame_names": frame_names,
                "frame_sizes": [[6, 4] for _ in frame_names],
                "frame_transforms": [app._letterbox_transform(6, 4, work_size=(6, 4)) for _ in frame_names],
                "cache_frame_paths": frame_paths,
                "cache_dir": str(sequence),
                "current_frame_idx": 0,
                "text_prompt_frame_idx": 0,
                "prompts_by_frame": {"0": {"points": [[1, 1]], "labels": [1]}},
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
                "work_size": [6, 4],
                "sam_output_prob_thresh": 0.5,
            }
            app._update_run_manifest(root / "run", "sam3", app._prompt_manifest_payload(state, 0.5))

            call_count = 0

            def fake_videomama(pipeline, frames, masks, **kwargs):
                nonlocal call_count
                value = float(call_count)
                call_count += 1
                return [np.full((*frame.shape[:2], 3), value, dtype=np.float32) for frame in frames]

            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
            )
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                    mock.patch.object(app, "_ensure_videomama_pipeline", return_value=object()), \
                    mock.patch.object(app, "videomama", side_effect=fake_videomama):
                payload = app.run_sequence(
                    state,
                    chunk_size=4,
                    overlap=2,
                    range_start=0,
                    range_end=5,
                    alpha_output_format="16-bit PNG",
                    matting_backend="videomama",
                )

            output_state = payload[4]
            alpha_dir = Path(output_state["alpha_output_dir"])
            frame_2 = np.array(Image.open(alpha_dir / "frame_0002.png"), dtype=np.float32) / 65535.0
            frame_3 = np.array(Image.open(alpha_dir / "frame_0003.png"), dtype=np.float32) / 65535.0
            np.testing.assert_allclose(frame_2, 1.0 / 3.0, atol=2e-5)
            np.testing.assert_allclose(frame_3, 2.0 / 3.0, atol=2e-5)
            manifest = app._read_run_manifest(root / "run")["videomama"]
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_frame_indices"], list(range(6)))

    def test_explicit_processing_resolutions_are_unchanged(self):
        """Regression guard: adding Auto must not disturb the pinned sizes."""
        for label, expected in (
            ('1024x576', (1024, 576)),
            ('1280x720 (experimental)', (1280, 720)),
            ('1536x864 (experimental)', (1536, 864)),
            ('2048x1152 (experimental)', (2048, 1152)),
        ):
            self.assertEqual(app._processing_work_size(label), expected, label)

    def test_unknown_processing_resolution_still_falls_back(self):
        self.assertEqual(
            app._processing_work_size('not-a-resolution'),
            app._processing_work_size(app.DEFAULT_PROCESSING_RESOLUTION),
        )

    def test_auto_resolution_resolves_to_concrete_even_dimensions(self):
        width, height = app._resolve_work_size(
            app.AUTO_PROCESSING_RESOLUTION, 4096, 2160, frames_in_chunk=2
        )
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertEqual(width % 64, 0)
        self.assertEqual(height % 64, 0)
        self.assertGreater(width * height, 0)

    def test_auto_resolution_never_leaks_the_label_into_the_cache_key(self):
        """session.json's work_size is a cache key.

        If Auto reached it as a string, a resumed run would compare equal
        regardless of the size its frames were actually built at.
        """
        width, height = app._resolve_work_size(
            app.AUTO_PROCESSING_RESOLUTION, 4096, 2160, frames_in_chunk=2
        )
        self.assertNotIsInstance(width, str)
        self.assertNotIn(app.AUTO_PROCESSING_RESOLUTION, (width, height))

    def test_auto_resolution_respects_the_cuda_grid_limit(self):
        import model_canvas
        width, height = app._resolve_work_size(
            app.AUTO_PROCESSING_RESOLUTION, 8192, 4320, frames_in_chunk=1,
            available_vram_bytes=200 << 30,
        )
        self.assertLessEqual((width // 8) * (height // 8), 65535)
        self.assertLessEqual(width * height, model_canvas.HARD_CANVAS_PX_LIMIT)

    def test_auto_resolution_does_not_block_reload_on_vram_drift(self):
        """Auto is sticky per run.

        Re-resolving it against different free VRAM must not raise the
        "cache is a different size" error, or a machine under memory pressure
        could never resume a run.
        """
        state = {'work_size': [1792, 1280]}
        app._ensure_loaded_processing_resolution(state, app.AUTO_PROCESSING_RESOLUTION)

    def test_explicit_resolution_mismatch_still_raises(self):
        state = {'work_size': [1024, 576]}
        with self.assertRaises(app.gr.Error):
            app._ensure_loaded_processing_resolution(state, '2048x1152 (experimental)')

    def test_matte_roi_defaults_to_full_frame(self):
        self.assertEqual(app.DEFAULT_SETTINGS['matte_roi_mode'], 'Full frame')

    def test_videomama_canvas_matches_the_roi_aspect_not_the_fixed_canvas(self):
        """The stretch bug, pinned.

        A 1200x2000 ROI (aspect 0.60) used to be resized onto the fixed 16:9
        work canvas, entering the model ~3x wider than reality. The canvas must
        now carry the ROI's own aspect.
        """
        roi = {'enabled': True, 'x': 0, 'y': 0, 'width': 1200, 'height': 2000}
        width, height, report = app._videomama_canvas(
            roi, source_size=(4096, 2160), work_size=(2048, 1152), frames_in_chunk=2,
            available_vram_bytes=20 << 30,
        )
        self.assertLess(width, height, 'a portrait ROI must get a portrait canvas')
        # The achievable aspect error is bounded by the 64px alignment grid: one
        # step on the short axis moves the aspect by alignment/min(w, h). Assert
        # against that rather than a magic number, and against the 196% error
        # the old fixed 2048x1152 canvas produced for this ROI.
        error = abs((width / height) / (1200 / 2000) - 1.0)
        self.assertLessEqual(error, 64 / min(width, height))
        self.assertLess(error, 0.15)
        self.assertEqual(width % 64, 0)
        self.assertEqual(height % 64, 0)
        self.assertIn('limited_by', report)

    def test_videomama_canvas_uses_full_frame_when_no_roi(self):
        roi = {'enabled': False, 'x': 0, 'y': 0, 'width': 4096, 'height': 2160}
        width, height, _report = app._videomama_canvas(
            roi, source_size=(4096, 2160), work_size=(2048, 1152), frames_in_chunk=2,
            available_vram_bytes=20 << 30,
        )
        self.assertLessEqual(abs((width / height) / (4096 / 2160) - 1.0),
                             64 / min(width, height))

    def test_videomama_canvas_never_exceeds_the_cuda_grid_limit(self):
        import model_canvas
        roi = {'enabled': False, 'x': 0, 'y': 0, 'width': 8192, 'height': 4320}
        width, height, _report = app._videomama_canvas(
            roi, source_size=(8192, 4320), work_size=(8192, 4320), frames_in_chunk=1,
            available_vram_bytes=200 << 30,
        )
        self.assertLessEqual((width // 8) * (height // 8), 65535)
        self.assertLessEqual(width * height, model_canvas.HARD_CANVAS_PX_LIMIT)

    def test_pinned_resolution_becomes_a_pixel_budget_not_a_stretch(self):
        """An explicit resolution keeps its pixel budget but stops distorting.

        2048x1152 is 2.36 Mpx; the canvas it produces for a 4:3 ROI should be
        near that area while carrying 4:3, rather than literally 2048x1152.
        """
        roi = {'enabled': True, 'x': 0, 'y': 0, 'width': 2400, 'height': 1800}
        width, height, _report = app._videomama_canvas(
            roi, source_size=(4096, 2160), work_size=(2048, 1152), frames_in_chunk=2,
            available_vram_bytes=20 << 30,
        )
        self.assertLessEqual(width * height, 2048 * 1152)
        self.assertLessEqual(abs((width / height) / (2400 / 1800) - 1.0),
                             64 / min(width, height))

    def test_videomama_receives_the_aspect_matched_canvas(self):
        """End to end: whatever _videomama_canvas decides is what the model gets."""
        roi = {'enabled': True, 'x': 0, 'y': 0, 'width': 1200, 'height': 2000}
        expected = app._videomama_canvas(
            roi, source_size=(4096, 2160), work_size=(2048, 1152), frames_in_chunk=2,
            available_vram_bytes=20 << 30)[:2]
        self.assertEqual(expected[0] % 64, 0)
        self.assertNotEqual(expected, (2048, 1152))

    def test_oom_classifier_accepts_allocator_oom(self):
        class FakeOOM(Exception):
            pass
        FakeOOM.__name__ = 'OutOfMemoryError'
        self.assertTrue(app._is_recoverable_oom(FakeOOM('CUDA out of memory')))
        self.assertTrue(app._is_recoverable_oom(
            RuntimeError('CUDA out of memory. Tried to allocate 5.05 GiB')))

    def test_oom_classifier_rejects_the_cuda_grid_limit(self):
        """That failure poisons the CUDA context; retrying is not recoverable."""
        self.assertFalse(app._is_recoverable_oom(
            RuntimeError('CUDA error: invalid configuration argument')))
        self.assertFalse(app._is_recoverable_oom(ValueError('some other bug')))

    def test_videomama_processes_sam_roi_but_writes_full_frame_alpha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence = root / 'sequence'
            masks_dir = root / 'run' / 'sam3_masks'
            sequence.mkdir()
            masks_dir.mkdir(parents=True)
            frame_path = sequence / 'frame_0000.png'
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(frame_path)
            cache_path = root / 'crop_cache.jpg'
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(cache_path)
            source_mask = np.zeros((32, 32), dtype=np.uint8)
            source_mask[12:20, 12:20] = 255
            Image.fromarray(source_mask).save(masks_dir / frame_path.name)

            crop_settings = app._sam_crop_settings(True, 8, 8, 16, 16)
            transform = app._sam_crop_transform(
                32, 32, work_size=(16, 16), crop_settings=crop_settings
            )
            state = {
                'sequence_dir': str(sequence),
                'frame_paths': [str(frame_path)],
                'frame_names': [frame_path.name],
                'frame_sizes': [[32, 32]],
                'frame_transforms': [transform],
                'cache_frame_paths': [str(cache_path)],
                'cache_dir': str(sequence),
                'current_frame_idx': 0,
                'text_prompt_frame_idx': 0,
                'prompts_by_frame': {'0': {'points': [[2, 2]], 'labels': [1]}},
                'preview_masks': {},
                'prompt_mode': 'Point keyframes',
                'concept_prompt': '',
                'generated_masks_dir': str(masks_dir),
                'generated_masks_sam_output_prob_thresh': 0.5,
                'videomama_output_dir': None,
                'alpha_output_dir': None,
                'run_root': str(root / 'run'),
                'exr_gamma': 1.0,
                'exr_exposure': 0.0,
                'work_size': [16, 16],
                'sam_crop_settings': crop_settings,
                'sam_output_prob_thresh': 0.5,
            }
            app._update_run_manifest(root / 'run', 'sam3', app._prompt_manifest_payload(state, 0.5))
            received_shapes = []
            received_masks = []

            def fake_videomama(pipeline, frames, masks, **kwargs):
                received_shapes.append((frames[0].shape, masks[0].shape))
                received_masks.append(masks[0].copy())
                return [np.ones((16, 16, 3), dtype=np.float32)]

            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with mock.patch.dict(sys.modules, {'torch': fake_torch}), \
                    mock.patch.object(app, '_ensure_videomama_pipeline', return_value=object()), \
                    mock.patch.object(app, 'videomama', side_effect=fake_videomama):
                payload = app.run_sequence(
                    state, chunk_size=1, overlap=0, videomama_guide_expand_px=2,
                    matting_backend="videomama",
                )

            self.assertEqual(received_shapes, [((16, 16, 3), (16, 16))])
            self.assertGreater(int((received_masks[0] > 0).sum()), int((source_mask[8:24, 8:24] > 0).sum()))
            alpha_path = Path(payload[4]['alpha_output_dir']) / frame_path.name
            alpha = np.array(Image.open(alpha_path))
            self.assertEqual(alpha.shape, (32, 32))
            self.assertTrue(np.all(alpha[8:24, 8:24] == 65535))
            self.assertEqual(int(alpha[:8].sum()), 0)
            self.assertEqual(int(alpha[:, :8].sum()), 0)
            self.assertEqual(int(alpha[24:].sum()), 0)
            self.assertEqual(int(alpha[:, 24:].sum()), 0)
            manifest = app._read_run_manifest(root / 'run')['videomama']
            self.assertEqual(manifest['backend_id'], 'videomama')
            self.assertEqual(manifest['identity']['params']['guide_expand_px'], 2)
            # The manual SAM ROI crop still overrides the automatic matte ROI.
            self.assertEqual(
                manifest['matte_roi'],
                {'enabled': True, 'x': 8, 'y': 8, 'width': 16, 'height': 16},
            )


if __name__ == "__main__":
    unittest.main()
