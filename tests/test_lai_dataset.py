# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Kept-surface tests for phenonn.data.lai_dataset helpers.

Only pure helpers with tiny synthetic NetCDF fixtures in temporary
directories — never `/data` or `runs/`. Full LAIDataset/RamLAIDataset
assembly is covered by scripts/smoke_train.sh on real pixelsets.
"""

import os
import tempfile
import unittest

import numpy as np

from phenonn.data.lai_dataset import (
    compute_climatology_lookup,
    decode_site_id,
    encode_site_id,
    generate_site_ids_from_range,
    list_valid_sites,
    load_co2_lut,
    load_selected_pixels,
)
from phenonn.utils.config import TARGETS_FNAME, VALID_FNAME

try:
    import netCDF4  # noqa: F401
    import xarray as xr

    _HAS_NC = True
except ImportError:  # pragma: no cover
    xr = None
    _HAS_NC = False


class TestSiteIds(unittest.TestCase):
    """Unit tests for site-id helpers."""

    def test_encode_decode_roundtrip(self):
        """encode then decode returns the indices."""
        sid = encode_site_id(351, 1105)
        self.assertEqual(sid, "pix_0351_01105")
        self.assertEqual(decode_site_id(sid), (351, 1105))

    def test_decode_malformed_raises(self):
        """Malformed ids raise ValueError."""
        for bad in ("E351_1105", "pix_0351", "pix_0351_01105_extra", ""):
            with self.assertRaises(ValueError):
                decode_site_id(bad)

    def test_generate_range_box(self):
        """Inclusive box generates every id exactly once."""
        sids = generate_site_ids_from_range((0, 1), (0, 2))
        self.assertEqual(len(sids), 6)
        self.assertEqual(len(set(sids)), 6)
        self.assertIn("pix_0000_00000", sids)
        self.assertIn("pix_0001_00002", sids)


class TestCo2Lut(unittest.TestCase):
    """Unit tests for load_co2_lut."""

    def test_parses_annee_lines(self):
        """Comments/blanks skipped, Annee_YYYY=VALUE parsed."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("# comment\n\nAnnee_2000=370.25\nAnnee_2001 = 371.50\n")
            path = f.name
        try:
            lut = load_co2_lut(path)
        finally:
            os.unlink(path)
        self.assertEqual(lut, {2000: 370.25, 2001: 371.50})

    def test_empty_file_raises(self):
        """A file with no entries raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("# nothing here\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_co2_lut(path)
        finally:
            os.unlink(path)


@unittest.skipUnless(_HAS_NC, "netCDF4/xarray required for fixtures")
class TestClimatologyLookup(unittest.TestCase):
    """Unit tests for compute_climatology_lookup."""

    def _write_targets(self, directory, year, ids, vals):
        ds = xr.Dataset(
            {"LAI": (("site", "dekad"), np.asarray(vals, dtype=np.float32))},
            coords={"site_id": ("site", np.asarray(ids))},
        )
        ds.to_netcdf(os.path.join(directory, TARGETS_FNAME.format(year=year)))
        ds.close()

    def test_mean_across_years(self):
        """Per-(site, dekad) mean over the requested years."""
        ids = ["pix_0000_00000", "pix_0000_00001"]
        with tempfile.TemporaryDirectory() as d:
            self._write_targets(d, 2000, ids, [[1.0] * 36, [2.0] * 36])
            self._write_targets(d, 2001, ids, [[3.0] * 36, [4.0] * 36])
            lut = compute_climatology_lookup(d, [2000, 2001], ids, verbose=False)
        np.testing.assert_allclose(lut[ids[0]], np.full(36, 2.0, dtype=np.float32))
        np.testing.assert_allclose(lut[ids[1]], np.full(36, 3.0, dtype=np.float32))

    def test_nan_propagates_and_missing_year_skipped(self):
        """All-NaN windows stay NaN; absent year files are skipped."""
        ids = ["pix_0000_00000", "pix_0000_00001"]
        with tempfile.TemporaryDirectory() as d:
            vals = np.full((2, 36), 5.0, dtype=np.float32)
            vals[1, :] = np.nan
            self._write_targets(d, 2000, ids, vals)
            # 2001 file absent on purpose.
            lut = compute_climatology_lookup(d, [2000, 2001], ids, verbose=False)
        np.testing.assert_allclose(lut[ids[0]], np.full(36, 5.0, dtype=np.float32))
        self.assertTrue(np.isnan(lut[ids[1]]).all())

    def test_empty_wanted_returns_empty(self):
        """No sites requested returns {} without touching disk."""
        self.assertEqual(
            compute_climatology_lookup("/nonexistent", [2000], [], verbose=False),
            {},
        )


@unittest.skipUnless(_HAS_NC, "netCDF4/xarray required for fixtures")
class TestValidSites(unittest.TestCase):
    """Unit tests for list_valid_sites."""

    def test_true_cells_become_ids(self):
        """True mask cells map to pix_LLLL_OOOOO ids."""
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 1] = True
        mask[2, 0] = True
        with tempfile.TemporaryDirectory() as d:
            ds = xr.Dataset({"valid": (("latitude", "longitude"), mask)})
            ds.to_netcdf(os.path.join(d, VALID_FNAME.format(year=2000)))
            ds.close()
            out = list_valid_sites(d, [2000])
        self.assertEqual(sorted(out[2000]), ["pix_0000_00001", "pix_0002_00000"])

    def test_missing_year_gives_empty(self):
        """Absent mask file yields an empty pool for that year."""
        with tempfile.TemporaryDirectory() as d:
            out = list_valid_sites(d, [1999])
        self.assertEqual(out[1999], [])


@unittest.skipUnless(_HAS_NC, "netCDF4/xarray required for fixtures")
class TestSelectedPixels(unittest.TestCase):
    """Unit tests for load_selected_pixels."""

    def test_roundtrip(self):
        """Written site_id values read back in order."""
        ids = ["pix_0000_00000", "pix_0001_00002"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "selected_pixels.nc")
            ds = xr.Dataset({"site_id": ("site", np.asarray(ids))})
            ds.to_netcdf(path)
            ds.close()
            self.assertEqual(load_selected_pixels(path), ids)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSiteIds))
    suite.addTests(loader.loadTestsFromTestCase(TestCo2Lut))
    suite.addTests(loader.loadTestsFromTestCase(TestClimatologyLookup))
    suite.addTests(loader.loadTestsFromTestCase(TestValidSites))
    suite.addTests(loader.loadTestsFromTestCase(TestSelectedPixels))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
