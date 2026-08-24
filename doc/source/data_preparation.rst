Data Formats
============

Per-Site CSV
------------

Files are named ``{PFT}_{site}.csv``. Daily rows must include:

.. code-block:: text

   year doy tmin tmax daylength vpd prcp srad swe mat map LAI

Rows are sorted by year and day of year. GDD, CDD, NCD, and Botta forcing
features are derived internally. A target year normally requires at least one
preceding year of daily history.

Flat CSV
--------

``features.csv`` contains one daily row per site:

.. code-block:: text

   site_id date year month day
   tmin tmax daylength prcp srad vpd swe
   pft1_frac ... pft15_frac

``date`` uses ``YYYYMMDD``. Seven phenology features are added internally,
giving 29 model channels: 7 meteorological, 7 derived, and 15 PFT fractions.

``targets.csv`` contains:

.. code-block:: text

   site_id date year month day LAI

Each ``(site_id, year)`` requires 36 LAI rows corresponding to days 5, 15, and
25 of each month.

Streaming Yearly CSV
--------------------

``train-big`` expects:

.. code-block:: text

   FEATURES_DIR/feature_{year}.csv
   TARGET_DIR/target_{year}.csv

For target year ``Y``, both ``feature_{Y-1}.csv`` and ``feature_Y.csv`` are
required. Rows must be contiguous by ``site_id``. IDs are expected to encode
grid positions, for example ``pix_0900_01500``. The loader creates adjacent
``.pixidx.npz`` indexes, so source directories must be writable.

Global Selected-Site NetCDF
---------------------------

The global dataset expects:

.. code-block:: text

   ERA_DIR/ERA5_daily_pixelset_{year}.nc
   TARGET_DIR/lai/LAI_dekadal_{year}.nc
   TARGET_DIR/pft/PFTmap_{year}.nc
   TARGET_DIR/CO2_annual.nc
   selected_pixels_era5_valid.nc

ERA files contain daily ``Tmin``, ``Tmax``, ``Tmean``, ``ssrd_sum``,
``strd_sum``, ``tp_sum``, ``VPD_max``, ``VPD_mean``, ``Rn_tot``, ``PET``, and
``SMI``. The target directory contains 36-date LAI, annual 15-PFT fractions,
and annual CO2.

Selection metadata includes ``site_id``, ``split``, ``is_primary``,
``arco_chunk_id``, ``sample_weight``, and source-index traceability. Every
yearly file must preserve canonical site order. Leap day is removed before
720-day windows are formed.

Splits
------

Global split codes are:

.. code-block:: text

   0 train
   1 validation
   2 test
   3 buffer

Auxiliaries are training-only. Validation and test metrics always use primary
sites. Site splits test spatial transfer; year splits test temporal transfer.
