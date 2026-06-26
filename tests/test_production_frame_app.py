import sys
import tempfile
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
                "Point keyframes",
                "",
            )

        self.assertFalse(stale_run.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(len(payload), app.UI_OUTPUT_COUNT)
        self.assertIn("Cleared cache run(s): shot_040_foot", payload[app.UI_OUTPUT_STATUS_INDEX])
        self.assertIn("cleared contents in place", payload[app.UI_OUTPUT_STATUS_INDEX])


if __name__ == "__main__":
    unittest.main()
