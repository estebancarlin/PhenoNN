# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
phenonn.datagen
===============

Data-generation pipeline for the ``pixelset1`` dataset: site selection,
rebalancing, ERA5 daily pixelsets, LAI/PFT pixelsets, daily LAI interpolation,
optional derived features, and normalisation statistics.

Submodules are CLI scripts and are meant to be run as modules, e.g.::

    python -m phenonn.datagen.select_pixels --help
    python -m phenonn.datagen.build_daily_dataset_pixelset --help
    python -m phenonn.datagen.create_stats_file --help

They are NOT imported here on purpose: importing them would drag in
``xarray`` / ``matplotlib`` / ``dask`` at package-import time. Import the
shared constants from :mod:`phenonn.datagen.config` instead.

Pipeline order
--------------
1. ``compute_valid_pixels`` → ``valid_pixels_{YYYY}.nc``
2. ``select_pixels`` → ``all_pixels_selected.nc``,
   ``selected_pixels005_1.nc``, ``selected_pixels01_1.nc``
3. ``rebalance_pixels`` (PFT route) / ``rebalance_spatial`` (spatial route) →
   ``selected_pixels005_balanced.nc``
4. ``make_selected_pixels_01`` → ``selected_pixels_01.nc`` (deduplicated 0.1°)
5. ``build_daily_dataset_pixelset`` (0.05° route) /
   ``build_daily_dataset_era5idx`` (0.1° route) →
   ``era5_pixelset[_01]/ERA5_daily_pixelset_{Y}.nc``
6. ``build_pixelset_targets`` → ``LAI_pixelset/``, ``PFT_pixelset/``
7. ``interpolate_lai_daily_linear`` → ``LAI_daily_linear/``
8. ``add_daylength``, ``add_pheno_thermal`` (optional augmentations)
9. ``create_stats_file`` → ``norm_stats_{Y0}_{Y1}.json``

Diagnostics (not part of the data build):
``diagnose_selected_pixels``, ``diagnose_selected_pixels_bis``,
``pft_diagnostics``.
"""

from phenonn.datagen.config import (
    N_PFT,
    N_DEKAD_YEAR,
    METEO_BASE,
    DYNAMIC_FEATURES,
    PFT_COLS,
    PFT_NAMES,
    FEATURES_FNAME,
    TARGETS_FNAME,
    PFT_FNAME,
    VALID_FNAME,
    SELECTED_PIXELS_FNAME,
)

__all__ = [
    "N_PFT",
    "N_DEKAD_YEAR",
    "METEO_BASE",
    "DYNAMIC_FEATURES",
    "PFT_COLS",
    "PFT_NAMES",
    "FEATURES_FNAME",
    "TARGETS_FNAME",
    "PFT_FNAME",
    "VALID_FNAME",
    "SELECTED_PIXELS_FNAME",
]
