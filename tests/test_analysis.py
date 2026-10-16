# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Smoke tests for the hand-run analyses under ``analysis/``.

No DATA, checkpoints, or GPUs: every analysis module must import cleanly,
and every CLI exposing ``parse_args`` must answer ``--help`` with exit 0.
The real runs (documented in ``analysis/README.md``) stay manual.
"""

import importlib
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANALYSIS_PACKAGES = [
    "analysis",
    "analysis.diagnostics",
]

ANALYSIS_SCRIPTS = [
    "analysis.diagnostics.annual_trend_study",
    "analysis.diagnostics.climatology_baseline",
    "analysis.diagnostics.climatology_skill",
    "analysis.diagnostics.diagnose_grid_alignment",
    "analysis.diagnostics.diagnose_selected_pixels",
    "analysis.diagnostics.inspect_era5_pixelset",
    "analysis.diagnostics.pft_amplitude_correlation",
    "analysis.diagnostics.pft_curves",
    "analysis.diagnostics.pft_diagnostics",
    "analysis.diagnostics.print_lai_pft_sites",
    "analysis.diagnostics.spatial_plot_selected_pixels",
]

# Only modules exposing parse_args; climatology_skill,
# diagnose_grid_alignment and spatial_plot_selected_pixels build their
# argparse inline in main() and are covered by the import test above.
HELP_MODULES = [
    "analysis.diagnostics.annual_trend_study",
    "analysis.diagnostics.climatology_baseline",
    "analysis.diagnostics.diagnose_selected_pixels",
    "analysis.diagnostics.inspect_era5_pixelset",
    "analysis.diagnostics.pft_amplitude_correlation",
    "analysis.diagnostics.pft_curves",
    "analysis.diagnostics.pft_diagnostics",
    "analysis.diagnostics.print_lai_pft_sites",
]


class TestAnalysisImports(unittest.TestCase):
    """Every analysis module imports without DATA or checkpoints."""

    def test_all_modules_import(self):
        for name in ANALYSIS_PACKAGES + ANALYSIS_SCRIPTS:
            with self.subTest(module=name):
                importlib.import_module(name)


class TestAnalysisHelp(unittest.TestCase):
    """Every CLI exposing parse_args answers --help with exit 0."""

    def test_help_exits_zero(self):
        for name in HELP_MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(
                    hasattr(module, "parse_args"), f"{name} has no parse_args"
                )
                # Scripts define parse_args() with no argv parameter
                # (argparse reads sys.argv); patch sys.argv for --help.
                argv = sys.argv
                sys.argv = [name, "--help"]
                try:
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as ctx:
                            module.parse_args()
                finally:
                    sys.argv = argv
                self.assertEqual(ctx.exception.code, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisImports))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisHelp))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
