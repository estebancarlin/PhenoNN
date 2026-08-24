import unittest

import numpy as np

from scripts.build_selected_pixels import GRID_SHAPE, coarse_valid, era_coordinates


class TestBuildSelectedPixels(unittest.TestCase):
    def test_coarse_valid_requires_two_native_pixels(self):
        raw = np.full((3600, 7200), 255, dtype=np.uint8)
        raw[0, 0] = 30
        raw[1, 1] = 60
        raw[2, 2] = 30

        valid = coarse_valid(raw, min_valid_native=2)

        self.assertEqual(valid.shape, GRID_SHAPE)
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[1, 1])
        self.assertEqual(int(valid.sum()), 1)

    def test_era_coordinates_match_historical_index_mapping(self):
        lat_idx = np.array([0, 1, 1799], dtype=np.int32)
        lon_idx = np.array([0, 1, 1799, 1800, 3599], dtype=np.int32)

        latitude, _ = era_coordinates(lat_idx, lon_idx[:3])
        _, longitude = era_coordinates(np.zeros(5, dtype=np.int32), lon_idx)

        np.testing.assert_allclose(latitude, [90.0, 89.9, -89.9], atol=1e-5)
        np.testing.assert_allclose(
            longitude, [180.0, -179.9, -0.1, 0.0, 179.9], atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
