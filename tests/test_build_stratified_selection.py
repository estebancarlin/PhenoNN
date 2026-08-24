import unittest

import numpy as np

from scripts.build_stratified_selection import (
    assign_cluster_splits,
    choose_pixels_in_cells,
    systematic_pps_cells,
)


class TestBuildStratifiedSelection(unittest.TestCase):
    def test_pps_selects_fixed_number_of_distinct_cells(self):
        cell_ids = np.arange(100, dtype=np.int32)
        counts = np.arange(1, 101, dtype=np.int64)
        rng = np.random.default_rng(42)

        selected = systematic_pps_cells(cell_ids, counts, 10, rng, draw=0)

        self.assertEqual(selected.size, 10)
        self.assertEqual(np.unique(selected).size, 10)

    def test_one_pixel_is_selected_per_cell(self):
        sorted_cells = np.repeat(np.arange(5), [1, 2, 3, 4, 5])
        candidates = np.arange(sorted_cells.size)
        rng = np.random.default_rng(42)

        selected = choose_pixels_in_cells(np.arange(5), sorted_cells, candidates, rng)

        self.assertEqual(selected.size, 5)
        np.testing.assert_array_equal(sorted_cells[selected], np.arange(5))

    def test_cluster_splits_approximate_requested_ratios(self):
        labels = np.repeat(np.arange(20), 50)
        dominant = np.tile(np.arange(1, 6), 200)
        lai_bin = np.tile(np.arange(5), 200)

        splits = assign_cluster_splits(
            labels, dominant, lai_bin, np.array([0.7, 0.15, 0.15]), seed=42
        )

        counts = np.bincount(splits, minlength=3)
        np.testing.assert_allclose(counts / counts.sum(), [0.7, 0.15, 0.15])
        for label in np.unique(labels):
            self.assertEqual(np.unique(splits[labels == label]).size, 1)


if __name__ == "__main__":
    unittest.main()
