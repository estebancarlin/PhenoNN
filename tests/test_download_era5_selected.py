import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from netCDF4 import Dataset

from scripts.download_era5_selected import (
    aggregate_accum,
    aggregate_instant,
    copy_selection_metadata,
    era5_land_site_indices,
    selected_to_arco_indices,
    spatial_batches,
)


class TestDownloadEra5Selected(unittest.TestCase):
    def test_selected_coordinates_map_to_arco_indices(self):
        latitude = np.ma.array([90.0, 89.9, -89.9], dtype=np.float32)
        longitude = np.ma.array([180.0, -179.9, 179.9], dtype=np.float32)

        lat_idx, lon_idx = selected_to_arco_indices(latitude, longitude)

        np.testing.assert_array_equal(lat_idx, [1800, 1799, 1])
        np.testing.assert_array_equal(lon_idx, [3599, 0, 3598])

    def test_spatial_batches_keep_each_site_once(self):
        lat_idx = np.array([0, 1, 4, 8], dtype=np.int32)
        lon_idx = np.array([0, 1, 8, 16], dtype=np.int32)

        batches, unique_count = spatial_batches(lat_idx, lon_idx, 2)

        self.assertEqual(unique_count, 3)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(4))

    def test_instantaneous_daily_aggregation(self):
        times = np.arange(
            np.datetime64("2000-01-01T00"),
            np.datetime64("2000-01-03T00"),
            dtype="datetime64[h]",
        )
        values = np.arange(48, dtype=np.float32)[:, None]

        days, minimum, maximum, mean = aggregate_instant(times, values)

        np.testing.assert_array_equal(
            days, np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]")
        )
        np.testing.assert_allclose(minimum[:, 0], [0, 24])
        np.testing.assert_allclose(maximum[:, 0], [23, 47])
        np.testing.assert_allclose(mean[:, 0], [11.5, 35.5])

    def test_instantaneous_ignores_next_midnight_boundary(self):
        times = np.arange(
            np.datetime64("2000-01-01T00"),
            np.datetime64("2000-01-02T01"),
            dtype="datetime64[h]",
        )
        values = np.arange(25, dtype=np.float32)[:, None]

        days, _, maximum, _ = aggregate_instant(times, values)

        np.testing.assert_array_equal(
            days, np.array(["2000-01-01"], dtype="datetime64[D]")
        )
        np.testing.assert_allclose(maximum[:, 0], [23])

    def test_accumulation_shifts_midnight_to_previous_day(self):
        times = np.arange(
            np.datetime64("2000-01-01T00"),
            np.datetime64("2000-01-02T02"),
            dtype="datetime64[h]",
        )
        values = np.ones((times.size, 1), dtype=np.float32)

        days, totals = aggregate_accum(times, values)

        np.testing.assert_array_equal(
            days,
            np.array(
                ["1999-12-31", "2000-01-01", "2000-01-02"],
                dtype="datetime64[D]",
            ),
        )
        np.testing.assert_allclose(totals[:, 0], [1, 24, 1])

    def test_sampling_metadata_is_copied(self):
        with TemporaryDirectory() as directory:
            selected_path = Path(directory) / "selected.nc"
            output_path = Path(directory) / "output.nc"
            with Dataset(selected_path, "w") as selected:
                selected.createDimension("site", 2)
                selected.createDimension("pft", 2)
                selected.createVariable("is_primary", "i1", ("site",))[:] = [1, 0]
                selected.createVariable("pft_fraction", "f4", ("site", "pft"))[:] = [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ]
                selected.seed = 42
            with Dataset(selected_path) as selected, Dataset(
                output_path, "w"
            ) as output:
                output.createDimension("site", 2)
                copy_selection_metadata(selected, output)
            with Dataset(output_path) as output:
                np.testing.assert_array_equal(output.variables["is_primary"][:], [1, 0])
                np.testing.assert_allclose(
                    output.variables["pft_fraction"][:],
                    [[0.8, 0.2], [0.3, 0.7]],
                )
                self.assertEqual(output.selection_seed, 42)

    def test_sampling_metadata_can_be_subset(self):
        with TemporaryDirectory() as directory:
            selected_path = Path(directory) / "selected.nc"
            output_path = Path(directory) / "output.nc"
            with Dataset(selected_path, "w") as selected:
                selected.createDimension("site", 3)
                selected.createDimension("pft", 2)
                selected.createVariable("split", "i1", ("site",))[:] = [0, 1, 2]
                selected.createVariable("pft_fraction", "f4", ("site", "pft"))[:] = [
                    [0.8, 0.2],
                    [0.3, 0.7],
                    [0.6, 0.4],
                ]
            with Dataset(selected_path) as selected, Dataset(
                output_path, "w"
            ) as output:
                output.createDimension("site", 2)
                copy_selection_metadata(selected, output, np.array([0, 2]))
            with Dataset(output_path) as output:
                np.testing.assert_array_equal(output.variables["split"][:], [0, 2])
                np.testing.assert_allclose(
                    output.variables["pft_fraction"][:], [[0.8, 0.2], [0.6, 0.4]]
                )

    def test_era5_land_mask_requires_consistent_variables(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "staging.nc"
            with Dataset(path, "w") as staging:
                staging.createDimension("time", 2)
                staging.createDimension("site", 3)
                for name in (
                    "Tmin",
                    "Tmax",
                    "Tmean",
                    "Tdew_mean",
                    "sp_mean",
                    "ssrd_sum",
                    "strd_sum",
                    "tp_sum",
                ):
                    variable = staging.createVariable(
                        name, "f4", ("time", "site"), fill_value=np.nan
                    )
                    variable[:] = [[1.0, np.nan, 2.0], [1.0, np.nan, 2.0]]
            with Dataset(path) as staging:
                np.testing.assert_array_equal(
                    era5_land_site_indices(staging), np.array([0, 2])
                )


if __name__ == "__main__":
    unittest.main()
