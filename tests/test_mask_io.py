"""Offline regression tests for PlateLock's bool/float mask PNG boundary.

Only synthetic pixels in temporary directories. No GPU, models, footage,
network connection, installed production PlateLock or ComfyUI required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

import cv2
import numpy as np

from platelock.masks import load_mask, save_mask


class MaskPngIoContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="platelock_mask_io_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic_mask.png"

    def decoded(self):
        return cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)

    def test_float_mask_roundtrips_without_erasing_foreground(self):
        source = np.array([[0.0, 0.49, 0.5, 0.8, 1.0]], dtype=np.float32)
        save_mask(str(self.path), source)
        self.assertEqual(self.decoded().dtype, np.uint8)
        np.testing.assert_array_equal(
            self.decoded(), [[0, 125, 128, 204, 255]]
        )
        np.testing.assert_array_equal(
            load_mask(str(self.path)), [[False, False, True, True, True]]
        )

    def test_boolean_mask_preserves_existing_binary_output_contract(self):
        source = np.array([[False, True, False, True]], dtype=bool)
        save_mask(str(self.path), source)
        np.testing.assert_array_equal(self.decoded(), [[0, 255, 0, 255]])
        np.testing.assert_array_equal(load_mask(str(self.path)), source)

    def test_uint8_0_1_binary_mask_is_not_lost(self):
        source = np.array([[0, 1, 1, 0]], dtype=np.uint8)
        save_mask(str(self.path), source)
        np.testing.assert_array_equal(self.decoded(), [[0, 255, 255, 0]])
        np.testing.assert_array_equal(
            load_mask(str(self.path)), [[False, True, True, False]]
        )

    def test_full_range_uint8_grayscale_is_backwards_compatible(self):
        source = np.array([[0, 100, 127, 128, 200, 255]], dtype=np.uint8)
        save_mask(str(self.path), source)
        np.testing.assert_array_equal(self.decoded(), source)
        np.testing.assert_array_equal(
            load_mask(str(self.path)),
            [[False, False, False, True, True, True]],
        )

    def test_out_of_range_or_nonfinite_float_fails_before_existing_write(self):
        self.path.write_bytes(b"previous accepted mask")
        for bad in (
            np.array([[float("nan")]], dtype=np.float32),
            np.array([[float("inf")]], dtype=np.float32),
            np.array([[-0.001]], dtype=np.float32),
            np.array([[1.001]], dtype=np.float32),
        ):
            with self.subTest(value=bad.item()):
                with self.assertRaisesRegex(ValueError, "finite values"):
                    save_mask(str(self.path), bad)
                self.assertEqual(self.path.read_bytes(), b"previous accepted mask")

    def test_invalid_shape_or_dtype_fails_before_existing_write(self):
        self.path.write_bytes(b"previous accepted mask")
        invalid = (
            np.ones((1, 2, 3), dtype=np.float32),
            np.array([[300]], dtype=np.uint16),
            np.array([["x"]], dtype=object),
        )
        for mask in invalid:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    save_mask(str(self.path), mask)
                self.assertEqual(self.path.read_bytes(), b"previous accepted mask")

    def test_failed_png_writer_cannot_replace_existing_file(self):
        self.path.write_bytes(b"previous accepted mask")
        with mock.patch("platelock.masks.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(OSError, "Failed to write mask PNG"):
                save_mask(str(self.path), np.array([[True]], dtype=bool))
        self.assertEqual(self.path.read_bytes(), b"previous accepted mask")


if __name__ == "__main__":
    unittest.main()
