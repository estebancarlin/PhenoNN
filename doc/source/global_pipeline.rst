Global Data Pipeline
====================

This page gives the canonical execution order. Run scripts from the repository
root with ``python scripts/NAME.py``. Keep large products outside Git.

1. Download Source Products
---------------------------

GEOV2 R01 and R02:

.. code-block:: bash

   export GEODES_API_KEY=YOUR_KEY
   python scripts/download_geodes_lai.py --output-dir data/geov2

Annual ORCHIDEE PFT maps and TRENDY CO2:

.. code-block:: bash

   python scripts/download_ipsl_forcings.py \
       --output-dir data/ipsl_forcings \
       --start-year 1992 --end-year 2019

The GEODES API reference is https://geodes.cnes.fr/support/api/. ERA5-Land
documentation is maintained by ECMWF at
https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5.

2. Build Validity Masks
-----------------------

``build_selected_pixels.py`` produces reusable annual validity masks. Its simple
random selection is not the canonical training selection.

.. code-block:: bash

   python scripts/build_selected_pixels.py \
       --geov2-dir data/geov2/R01 \
       --pft-dir data/ipsl_forcings/pft \
       --output-dir data/selection_work \
       --start-year 1992 --end-year 2018 \
       --fraction 0.005

3. Build the Balanced Spatial Selection
---------------------------------------

.. code-block:: bash

   python scripts/build_stratified_selection.py \
       --mask-dir data/selection_work/valid_masks \
       --geov2-dir data/geov2/R01 \
       --pft-map data/ipsl_forcings/pft/PFTmap_2000.nc \
       --output-dir data/stratified_selection \
       --fraction 0.005 \
       --buffer-degrees 2.0

Use ``selected_primary_auxiliary.nc`` for ERA extraction. Auxiliaries improve
training density but never enter validation or test metrics.

4. Extract and Derive ERA5-Land
-------------------------------

.. code-block:: bash

   python scripts/download_era5_selected.py \
       --selected-pixels data/stratified_selection/selected_primary_auxiliary.nc \
       --output-dir data/era5_selected \
       --start-year 1991 --end-year 2018

This writes ARCO staging state, filters sites masked for all ERA variables, and
creates the canonical ``selected_pixels_era5_valid.nc`` plus daily yearly files.
Use ``--derive-only`` to restart local daily derivation without another download.

The standalone ``download_era5_land.py`` downloads monthly hourly CDS files for
bounded regions. Those files are not direct ``train-global`` inputs.

5. Audit and Freeze ERA
-----------------------

.. code-block:: bash

   python scripts/validate_era5_selected.py \
       --selection data/era5_selected/selected_pixels_era5_valid.nc \
       --daily-dir data/era5_selected/daily \
       --output data/era5_selected/era5_audit_manifest.json \
       --start-year 1991 --end-year 2018

The audit checks calendars, metadata order, units, finite values, physical
consistency, split integrity, and SHA-256 hashes.

6. Build Selected Targets
-------------------------

Always use the ERA-filtered canonical selection:

.. code-block:: bash

   python scripts/build_selected_targets.py \
       --selection data/era5_selected/selected_pixels_era5_valid.nc \
       --geov2-dir data/geov2/R01 \
       --pft-dir data/ipsl_forcings/pft \
       --co2-file data/ipsl_forcings/co2/CO2_1700_2023_TRENDYv2024.txt \
       --output-dir data/selected_targets \
       --start-year 1992 --end-year 2018 \
       --co2-start-year 1991

7. Audit and Freeze Targets
---------------------------

.. code-block:: bash

   python scripts/validate_selected_targets.py \
       --selection data/era5_selected/selected_pixels_era5_valid.nc \
       --target-dir data/selected_targets \
       --output data/selected_targets/target_audit_manifest.json \
       --start-year 1992 --end-year 2018

8. Optional Normalization Ablation
----------------------------------

Raw physical inputs are the default. Compute normalization statistics from
spatial-training primaries only when running an explicit ablation:

.. code-block:: bash

   python scripts/compute_global_norm_stats.py \
       --selection data/era5_selected/selected_pixels_era5_valid.nc \
       --era-dir data/era5_selected/daily \
       --target-dir data/selected_targets \
       --train-years 1993-2014 \
       --output data/selected_targets/norm_stats_train_primary_1993_2014.json

Then add ``--normalize --norm-stats FILE`` to ``train-global``.

9. Train and Validate
---------------------

Use :doc:`training` for architecture options and :doc:`prediction` for spatial,
temporal, and final locked-test evaluation. Do not use 2000 as a temporal test
year because its LAI informed selection diagnostics.
