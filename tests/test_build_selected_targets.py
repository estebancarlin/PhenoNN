import gzip
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from netCDF4 import Dataset

from scripts.build_selected_targets import extract_lai, write_pft_year
from scripts.validate_selected_targets import validate_lai_arrays, validate_pft_arrays


class TestBuildSelectedTargets(unittest.TestCase):
    def test_extract_lai_uses_two_by_two_valid_mean(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lai.h5.gz"
            uncompressed = Path(directory) / "lai.h5"
            raw = np.array(
                [
                    [30, 60, 90, 120],
                    [255, 30, 255, 60],
                    [0, 30, 60, 90],
                    [60, 90, 120, 150],
                ],
                dtype=np.uint8,
            )
            with h5py.File(uncompressed, "w") as output:
                output.create_dataset("LAI", data=raw)
            with uncompressed.open("rb") as source, gzip.open(path, "wb") as compressed:
                shutil.copyfileobj(source, compressed)

            values = extract_lai(
                path,
                np.array([0, 1], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                min_valid_native=2,
            )

            np.testing.assert_allclose(values, [4 / 3, 3.5])

    def test_pft_output_preserves_site_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "PFTmap_2000.nc"
            output_path = root / "selected.nc"
            with Dataset(source_path, "w") as source:
                source.createDimension("time_counter", 1)
                source.createDimension("veget", 15)
                source.createDimension("lat", 2)
                source.createDimension("lon", 2)
                values = source.createVariable(
                    "maxvegetfrac",
                    "f4",
                    ("time_counter", "veget", "lat", "lon"),
                )
                values[:] = 0.0
                values[0, 0, :, :] = [[0.6, 0.7], [0.8, 0.9]]
            metadata = {}
            site_values = {
                "site_id": np.array(["second", "first"]),
                "lat_idx": np.array([1, 0], dtype=np.int32),
                "lon_idx": np.array([1, 0], dtype=np.int32),
            }
            for name in (
                "latitude",
                "longitude",
                "era_latitude",
                "era_longitude",
                "lai_mean",
                "lai_amplitude",
                "sample_weight",
                "inclusion_probability",
            ):
                site_values[name] = np.ones(2, dtype=np.float32)
            for name in ("arco_chunk_id", "source_site_index"):
                site_values[name] = np.arange(2, dtype=np.int64)
            for name in (
                "is_primary",
                "split",
                "spatial_group",
                "dominant_pft",
                "mixed_pft",
                "lai_bin",
            ):
                site_values[name] = np.zeros(2, dtype=np.int8)
            metadata.update(site_values)

            summary = write_pft_year(
                output_path, 2000, source_path, metadata, "selection-hash"
            )

            with Dataset(output_path) as output:
                np.testing.assert_allclose(output.variables["pft_frac"][0], [0.9, 0.6])
                np.testing.assert_array_equal(
                    np.asarray(output.variables["site_id"][:]).astype(str),
                    ["second", "first"],
                )
            self.assertEqual(summary["valid_sites"], 2)

    def test_target_array_validation(self):
        values = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        valid = np.isfinite(values)
        errors, summary = validate_lai_arrays(
            values, valid, valid.sum(axis=0), min_valid_dekads=2
        )
        self.assertEqual(errors, [])
        self.assertEqual(summary["valid_site_years"], 1)

    def test_pft_array_validation(self):
        fractions = np.zeros((15, 2), dtype=np.float32)
        fractions[0] = [0.6, 0.4]
        fractions[1] = [0.4, 0.6]
        errors, summary = validate_pft_arrays(fractions, np.ones(2, dtype=np.int8))
        self.assertEqual(errors, [])
        self.assertEqual(summary["valid_sites"], 2)


if __name__ == "__main__":
    unittest.main()
