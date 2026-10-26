"""CPU-only acceptance of the synthetic PlateLock backward-map visual example."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import synthetic_registration_demo as demo  # noqa: E402


class SyntheticRegistrationDemoTests(unittest.TestCase):
    def test_known_translation_is_exactly_removed_in_interior(self):
        images = demo.register_synthetic()
        self.assertGreater(images["unaligned_mae"], 20)
        self.assertEqual(images["registered_mae"], 0)
        self.assertEqual(images["original"].shape, (demo.H, demo.W, 3))
        self.assertEqual(images["registered_clean"].dtype, np.uint8)

    def test_result_is_reproducible(self):
        a = demo.register_synthetic()
        b = demo.register_synthetic()
        for name in ("original", "unaligned_clean", "registered_clean"):
            self.assertTrue(np.array_equal(a[name], b[name]))

    def test_contact_sheet_is_full_three_panel_image(self):
        image = demo.make_contact_sheet(demo.register_synthetic())
        self.assertEqual(image.shape, (demo.H + 65, 3 * demo.W + 24, 3))

    def test_png_is_real_decodable_output_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "synthetic.png"
            metadata = demo.render_png(file)
            self.assertTrue(metadata["synthetic_only"])
            self.assertGreater(metadata["unaligned_mae"], 20)
            self.assertEqual(metadata["registered_mae"], 0)
            loaded = cv2.imread(str(file), cv2.IMREAD_COLOR)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (demo.H + 65, 3 * demo.W + 24, 3))
            with self.assertRaises(FileExistsError):
                demo.render_png(file)

    def test_cli_does_not_require_project_venv_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.png"
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "examples" /
                                         "synthetic_registration_demo.py"),
                 "--out", str(path)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(path.exists())
            result = json.loads(proc.stdout)
            self.assertEqual(result["known_translation_px"], [16, -10])
            self.assertEqual(result["registered_mae"], 0)


if __name__ == "__main__":
    unittest.main()
