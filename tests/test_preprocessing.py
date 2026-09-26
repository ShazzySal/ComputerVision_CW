"""Unit tests for the fundus preprocessing pipeline (core.preprocessing)."""

import os
import tempfile
import unittest
import cv2
import numpy as np

from core.config import AppConfig
from core.preprocessing import (
    apply_clahe,
    ben_graham_enhance,
    crop_image_from_gray,
    denoise_fundus,
    enhance_edges,
    preprocess_image,
)


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite covering optical standardisation and preprocessing functions."""

    def setUp(self):
        # Create a synthetic circular fundus-like image: 300x300 RGB
        self.raw_size = (300, 300)
        self.sample_img = np.zeros((*self.raw_size, 3), dtype=np.uint8)
        # Draw a central orange/red fundus circle
        cv2.circle(self.sample_img, (150, 150), 120, (180, 80, 20), -1)
        # Add a simulated optic disc and a small blood vessel
        cv2.circle(self.sample_img, (110, 150), 18, (240, 210, 80), -1)
        cv2.line(self.sample_img, (110, 150), (220, 120), (120, 20, 10), 3)

    def test_crop_image_from_gray_removes_black_border(self):
        """Verify that crop_image_from_gray removes outer black margins."""
        cropped = crop_image_from_gray(self.sample_img, threshold=7)
        self.assertLess(cropped.shape[0], self.raw_size[0])
        self.assertLess(cropped.shape[1], self.raw_size[1])
        self.assertEqual(cropped.shape[2], 3)
        self.assertGreater(cropped.size, 0)

    def test_crop_image_all_black_returns_original(self):
        """An entirely black image should be returned unchanged without raising."""
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_image_from_gray(black)
        self.assertEqual(out.shape, black.shape)

    def test_ben_graham_enhance_preserves_dimensions(self):
        """Ben Graham enhancement should return uint8 with identical dimensions."""
        enhanced = ben_graham_enhance(self.sample_img, sigma=10)
        self.assertEqual(enhanced.shape, self.sample_img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_preprocess_image_from_array_standard_spec(self):
        """preprocess_image on array must output (224, 224, 3) float32 in [0, 1]."""
        out = preprocess_image(self.sample_img)
        self.assertEqual(out.shape, (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_preprocess_image_custom_size_and_ablation(self):
        """Verify custom target resolution and ablation flags."""
        custom_size = 128
        out = preprocess_image(self.sample_img, img_size=custom_size, apply_ben_graham=False)
        self.assertEqual(out.shape, (custom_size, custom_size, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_preprocess_image_from_filepath(self):
        """preprocess_image should load and process from a valid image path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, cv2.cvtColor(self.sample_img, cv2.COLOR_RGB2BGR))
            out = preprocess_image(tmp_path)
            self.assertEqual(out.shape, (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3))
            self.assertEqual(out.dtype, np.float32)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_preprocess_image_nonexistent_file_raises_value_error(self):
        """Passing a non-existent file path must raise ValueError."""
        with self.assertRaises(ValueError):
            preprocess_image("non_existent_fundus_photo_12345.jpg")

    def test_preprocess_image_empty_array_raises_value_error(self):
        """Passing an empty array must raise ValueError."""
        with self.assertRaises(ValueError):
            preprocess_image(np.array([], dtype=np.uint8))

    def test_optional_enhancements(self):
        """Verify that optional enhancement filters (CLAHE, denoise, sharpen) execute."""
        clahe_out = apply_clahe(self.sample_img)
        self.assertEqual(clahe_out.shape, self.sample_img.shape)

        denoise_out = denoise_fundus(self.sample_img)
        self.assertEqual(denoise_out.shape, self.sample_img.shape)

        edge_out = enhance_edges(self.sample_img)
        self.assertEqual(edge_out.shape, self.sample_img.shape)


if __name__ == "__main__":
    unittest.main()
