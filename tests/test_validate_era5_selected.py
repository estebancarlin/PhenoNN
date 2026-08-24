import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from netCDF4 import Dataset

from scripts.download_era5_selected import FINAL_VARIABLES
from scripts.validate_era5_selected import audit


class TestValidateEra5Selected(unittest.TestCase):
    def write_selection(self, path):
        with Dataset(path, "w") as output:
            output.createDimension("site", 3)
            output.createDimension("pft", 2)
            variables = {
                "lat_idx": ("i4", [1, 2, 3]),
                "lon_idx": ("i4", [4, 5, 6]),
                "latitude": ("f4", [10.0, 20.0, 30.0]),
                "longitude": ("f4", [40.0, 50.0, 60.0]),
                "era_latitude": ("f4", [10.0, 20.0, 30.0]),
                "era_longitude": ("f4", [40.0, 50.0, 60.0]),
                "arco_chunk_id": ("i8", [1, 1, 2]),
                "is_primary": ("i1", [1, 0, 1]),
                "split": ("i1", [0, 0, 1]),
                "spatial_group": ("i2", [1, 1, 2]),
                "sample_weight": ("f4", [1.0, 1.0, 1.0]),
                "source_site_index": ("i4", [0, 2, 3]),
            }
            for name, (dtype, values) in variables.items():
                output.createVariable(name, dtype, ("site",))[:] = values
            site_id = output.createVariable("site_id", str, ("site",))
            site_id[:] = np.asarray(["a", "b", "c"], dtype=object)
            output.createVariable("pft_fraction", "f4", ("site", "pft"))[:] = [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
            ]

    def write_daily(self, path, selection_path, year):
        days = 366 if year % 4 == 0 else 365
        with Dataset(selection_path) as selection, Dataset(path, "w") as output:
            output.createDimension("time", days)
            output.createDimension("site", 3)
            output.createDimension("pft", 2)
            time = output.createVariable("time", "i4", ("time",))
            time.units = f"days since {year}-01-01 00:00:00"
            time.calendar = "proleptic_gregorian"
            time[:] = np.arange(days)
            for name, source in selection.variables.items():
                variable = output.createVariable(
                    name, source.datatype, source.dimensions
                )
                variable[:] = source[:]
            for name, (units, _) in FINAL_VARIABLES.items():
                variable = output.createVariable(name, "f4", ("time", "site"))
                variable.units = units
                variable[:] = 1.0
            output.variables["Tmin"][:] = 0.0
            output.variables["Tmean"][:] = 1.0
            output.variables["Tmax"][:] = 2.0
            output.variables["VPD_mean"][:] = 0.5
            output.variables["VPD_max"][:] = 1.0

    def test_complete_artifact_passes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.nc"
            daily_dir = root / "daily"
            daily_dir.mkdir()
            manifest = root / "manifest.json"
            self.write_selection(selection)
            self.write_daily(daily_dir / "ERA5_daily_pixelset_2000.nc", selection, 2000)
            args = argparse.Namespace(
                selection=selection,
                daily_dir=daily_dir,
                output=manifest,
                start_year=2000,
                end_year=2000,
                site_batch=2,
                skip_sha256=True,
            )

            result = audit(args)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["selection_summary"]["n_site"], 3)
            self.assertEqual(json.loads(manifest.read_text())["errors"], [])

    def test_spatial_group_cannot_cross_active_splits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "selection.nc"
            self.write_selection(path)
            with Dataset(path, "a") as selection:
                selection.variables["spatial_group"][:] = [1, 1, 1]

            daily_dir = Path(directory) / "daily"
            daily_dir.mkdir()
            self.write_daily(daily_dir / "ERA5_daily_pixelset_2001.nc", path, 2001)
            args = argparse.Namespace(
                selection=path,
                daily_dir=daily_dir,
                output=Path(directory) / "manifest.json",
                start_year=2001,
                end_year=2001,
                site_batch=2,
                skip_sha256=True,
            )

            with self.assertRaisesRegex(RuntimeError, "ERA5 audit failed"):
                audit(args)


if __name__ == "__main__":
    unittest.main()
