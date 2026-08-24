import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from netCDF4 import Dataset

from phenonn.data.dataset_netcdf import (
    GlobalLAIDataset,
    METEO_FEATURES,
    _read_selected_sites,
)
from phenonn.utils.wrappers import Every10DaysWrapper, _OBS_POSITIONS


class IdentitySequence(torch.nn.Module):
    def forward(self, values):
        return values[:, :1, :]


class TestGlobalLAIDataset(unittest.TestCase):
    def write_selection(self, path):
        with Dataset(path, "w") as output:
            output.createDimension("site", 2)
            site_id = output.createVariable("site_id", str, ("site",))
            site_id[:] = np.asarray(["primary", "auxiliary"], dtype=object)
            values = {
                "split": ("i1", [0, 0]),
                "is_primary": ("i1", [1, 0]),
                "arco_chunk_id": ("i8", [10, 10]),
                "sample_weight": ("f4", [1.0, 0.5]),
            }
            for name, (dtype, data) in values.items():
                output.createVariable(name, dtype, ("site",))[:] = data

    def write_era(self, path, year):
        leap = year % 4 == 0
        days = 366 if leap else 365
        with Dataset(path, "w") as output:
            output.createDimension("time", days)
            output.createDimension("site", 2)
            site_id = output.createVariable("site_id", str, ("site",))
            site_id[:] = np.asarray(["primary", "auxiliary"], dtype=object)
            for feature_index, name in enumerate(METEO_FEATURES):
                variable = output.createVariable(name, "f4", ("time", "site"))
                values = (
                    np.arange(days, dtype=np.float32)[:, None] + feature_index * 1000
                )
                variable[:] = np.repeat(values, 2, axis=1)

    def write_target(self, root, year):
        lai_path = root / "lai" / f"LAI_dekadal_{year}.nc"
        pft_path = root / "pft" / f"PFTmap_{year}.nc"
        lai_path.parent.mkdir(parents=True, exist_ok=True)
        pft_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(lai_path, "w") as output:
            output.createDimension("dekad", 36)
            output.createDimension("site", 2)
            site_id = output.createVariable("site_id", str, ("site",))
            site_id[:] = np.asarray(["primary", "auxiliary"], dtype=object)
            output.createVariable("LAI", "f4", ("dekad", "site"))[:] = 2.0
            output.createVariable("target_valid", "i1", ("dekad", "site"))[:] = 1
        with Dataset(pft_path, "w") as output:
            output.createDimension("pft", 15)
            output.createDimension("site", 2)
            site_id = output.createVariable("site_id", str, ("site",))
            site_id[:] = np.asarray(["primary", "auxiliary"], dtype=object)
            fractions = np.zeros((15, 2), dtype=np.float32)
            fractions[0] = 1.0
            output.createVariable("pft_frac", "f4", ("pft", "site"))[:] = fractions
            output.createVariable("pft_valid", "i1", ("site",))[:] = 1

    def write_co2(self, path):
        with Dataset(path, "w") as output:
            output.createDimension("year", 2)
            output.createVariable("year", "i2", ("year",))[:] = [1999, 2000]
            output.createVariable("co2", "f4", ("year",))[:] = [367.8, 368.97]

    def test_repo_contract_and_leap_alignment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            era = root / "era"
            target = root / "target"
            era.mkdir()
            target.mkdir()
            selection = root / "selection.nc"
            self.write_selection(selection)
            self.write_era(era / "ERA5_daily_pixelset_1999.nc", 1999)
            self.write_era(era / "ERA5_daily_pixelset_2000.nc", 2000)
            self.write_target(target, 2000)
            self.write_co2(target / "CO2_annual.nc")

            dataset = GlobalLAIDataset(
                str(era),
                str(target),
                str(selection),
                years=[2000],
                split="train",
                seq_length=720,
            )
            features, targets = dataset[0]

            self.assertEqual(features.shape, (27, 720))
            self.assertEqual(targets.shape, (1, 36))
            self.assertEqual(dataset.feature_channels, 27)
            self.assertEqual(
                dataset.feature_names[-15:], [f"pft{i}_frac" for i in range(1, 16)]
            )
            np.testing.assert_allclose(features[-16].numpy(), 368.97)
            np.testing.assert_allclose(features[-15].numpy(), 1.0)

            wrapper = Every10DaysWrapper(IdentitySequence())
            predictions = wrapper(features.unsqueeze(0))[0, 0]
            current_year = np.delete(np.arange(366, dtype=np.float32), 59)
            np.testing.assert_allclose(
                predictions.numpy(), current_year[_OBS_POSITIONS]
            )

    def test_primary_only_filter(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            era = root / "era"
            target = root / "target"
            era.mkdir()
            target.mkdir()
            selection = root / "selection.nc"
            self.write_selection(selection)
            self.write_era(era / "ERA5_daily_pixelset_1999.nc", 1999)
            self.write_era(era / "ERA5_daily_pixelset_2000.nc", 2000)
            self.write_target(target, 2000)
            self.write_co2(target / "CO2_annual.nc")

            dataset = GlobalLAIDataset(
                str(era),
                str(target),
                str(selection),
                years=[2000],
                split="train",
                include_auxiliary=False,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.get_site_info(0)["site_id"], "primary")

    def test_normalization_is_explicit_opt_in(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            era = root / "era"
            target = root / "target"
            era.mkdir()
            target.mkdir()
            selection = root / "selection.nc"
            self.write_selection(selection)
            self.write_era(era / "ERA5_daily_pixelset_1999.nc", 1999)
            self.write_era(era / "ERA5_daily_pixelset_2000.nc", 2000)
            self.write_target(target, 2000)
            self.write_co2(target / "CO2_annual.nc")
            statistics = {
                feature: {"mean": 1.0, "std": 2.0}
                for feature in GlobalLAIDataset(
                    str(era),
                    str(target),
                    str(selection),
                    years=[2000],
                    site_ids=["primary"],
                ).feature_names
            }

            normalized = GlobalLAIDataset(
                str(era),
                str(target),
                str(selection),
                years=[2000],
                site_ids=["primary"],
                normalize=True,
                norm_stats=statistics,
            )
            features, _ = normalized[0]

            self.assertAlmostEqual(float(features[-15, 0]), 0.0)
            self.assertAlmostEqual(
                float(features[-16, 0]), (368.97 - 1.0) / 2.0, places=4
            )

    def test_contiguous_reader_preserves_sparse_site_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "values.nc"
            with Dataset(path, "w") as output:
                output.createDimension("time", 3)
                output.createDimension("site", 7)
                variable = output.createVariable("value", "f4", ("time", "site"))
                variable[:] = np.arange(21, dtype=np.float32).reshape(3, 7)
            with Dataset(path) as source:
                values = _read_selected_sites(
                    source.variables["value"],
                    np.array([0, 2], dtype=np.int32),
                    np.array([1, 5], dtype=np.int32),
                    site_batch=3,
                )
            np.testing.assert_array_equal(values, [[1, 5], [15, 19]])


if __name__ == "__main__":
    unittest.main()
