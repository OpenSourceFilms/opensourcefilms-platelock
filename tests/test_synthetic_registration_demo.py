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

    def test_stmap_identity_endpoints_both_origin_modes(self):
        h, w = demo.H, demo.W
        map_x, map_y = demo.geometry.identity_map(h, w)
        for origin in ("bottom-left", "top-left"):
            with self.subTest(origin=origin):
                st = demo.geometry.backward_map_to_stmap(
                    map_x, map_y, w, h, origin=origin
                )
                self.assertEqual(st.shape, (h, w, 2))
                self.assertEqual(st.dtype, np.float32)
                self.assertAlmostEqual(float(st[0, 0, 0]), 0.0)
                self.assertAlmostEqual(float(st[0, w - 1, 0]), 1.0)
                self.assertAlmostEqual(float(st[0, w // 2, 0]),
                                       (w // 2) / (w - 1), places=6)
                if origin == "bottom-left":
                    self.assertAlmostEqual(float(st[0, 0, 1]), 1.0)
                    self.assertAlmostEqual(float(st[h - 1, 0, 1]), 0.0)
                else:
                    self.assertAlmostEqual(float(st[0, 0, 1]), 0.0)
                    self.assertAlmostEqual(float(st[h - 1, 0, 1]), 1.0)
                restored_x, restored_y = demo.geometry.stmap_to_backward_map(
                    st, w, h, origin=origin
                )
                self.assertTrue(np.allclose(map_x, restored_x, atol=2e-5))
                self.assertTrue(np.allclose(map_y, restored_y, atol=2e-5))

    def test_stmap_does_not_silently_clamp_out_of_bounds_coordinates(self):
        mx, my = demo.geometry.identity_map(demo.H, demo.W)
        mx[0, 0] = -3.0
        my[0, 0] = -7.0
        mx[-1, -1] = float(demo.W + 5)
        my[-1, -1] = float(demo.H + 4)
        st = demo.geometry.backward_map_to_stmap(mx, my, demo.W, demo.H)
        self.assertLess(float(st[0, 0, 0]), 0.0)
        self.assertGreater(float(st[0, 0, 1]), 1.0)
        self.assertGreater(float(st[-1, -1, 0]), 1.0)
        self.assertLess(float(st[-1, -1, 1]), 0.0)

    def test_readme_stmap_formula_matches_geometry_contract(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("map_x / (W - 1)", text)
        self.assertIn("1 - map_y / (H - 1)", text)
        self.assertIn("Nuke node-level visual", text)

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
