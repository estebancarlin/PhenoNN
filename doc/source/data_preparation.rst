Data preparation
================

PhenoNN reads pixelset NetCDF files. Target variables (LAI and PFT) are
stored in per-year files with a flat ``site`` dimension and a ``site_id``
coordinate on the native 0.05° grid. ERA5 features also use a flat ``site``
dimension, but may be keyed either by the same 0.05° IDs or by deduplicated
0.1° cell IDs joined through a parent map.

File naming
-----------

The names are defined in :mod:`phenonn.utils.config` and must be
respected. For each year ``YYYY``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - File
     - Variables
     - Dims
   * - ``ERA5_daily_pixelset_{YYYY}.nc``
     - meteorology, engineered
     - ``(time, site)``
   * - ``LAI_dekadal_{YYYY}.nc``
     - ``LAI``
     - ``(dekad, site)``
   * - ``PFTmap_{YYYY}.nc``
     - ``pft_frac``
     - ``(pft, site)``

In addition, a small set of **non-yearly** auxiliary files:

- ``selected_pixels005_balanced.nc`` — the 0.05° training site pool, with
  ``site_id``, ``latitude`` and ``longitude``.
- ``selected_pixels01_1.nc`` — optional 0.05°→0.1° parent map, with
  ``site_id05`` and ``parent_site``.
- ``norm_stats.json`` — per-feature ``{"mean": ..., "std": ...}``.
- a CO2 lookup file, one ``Annee_YYYY=VALUE`` per line.

Site identifiers
----------------

A target site is identified by ``pix_LLLL_OOOOO`` where ``LLLL`` and
``OOOOO`` are indices on the native 0.05° LAI/PFT grid, both zero-padded.
LAI, PFT, daily LAI and target statistics use these IDs directly. On the
0.1° ERA5 route, features use ``E{lat_idx}_{lon_idx}`` IDs; the parent map
links each ``pix_...`` site to its ``E...`` parent. On the legacy identity
route, features use the same ``pix_...`` IDs as the targets.

The functions :func:`~phenonn.data.lai_dataset.encode_site_id` and
:func:`~phenonn.data.lai_dataset.decode_site_id` convert between
``(lat_idx, lon_idx)`` and the string form.

Meteorology file
----------------

``ERA5_daily_pixelset_{YYYY}.nc`` is opened with
``xarray.open_dataset`` and read by variable name. The variables
expected are listed in :data:`phenonn.utils.config.METEO_BASE` (10
entries) plus the engineered ``daylength`` and ``SMI`` (see
:data:`phenonn.utils.config.AUGMENTED_FEATURES`)::

    Tmin, Tmax, Tmean
    ssrd_sum, strd_sum, tp_sum
    VPD_max, VPD_mean
    Rn_tot, PET
    daylength, SMI

If ``add_pheno_features = True`` in :mod:`phenonn.utils.config`, the ten
thermal phenology variables in ``DERIVED_FEATURES`` are also read
(GDD, NCD, NGD, Botta threshold and forcing, rising and falling
temperature trends). Those are computed by
:mod:`phenonn.datagen.add_pheno_thermal` when the optional phenology step is
enabled.

Target file
-----------

``LAI_dekadal_{YYYY}.nc`` has a ``dekad`` dimension of length 36. The
36 dekads correspond to days 5, 15, 25 of every month in a non-leap
year. The variable is ``LAI`` and can contain NaNs — the loss masks
them out and the predictor reports only the finite targets.

For the daily-interpolated target (used with ``--daily_lai``), an
additional file per year is required:

- ``LAI_daily_{YYYY}.nc`` — dims ``(day, site)`` with ``day = 365``.

The daily curve is produced offline by linear interpolation between the
36 dekadal anchors. See the ``interpolate_lai_daily_linear.py`` script
in the preprocessing module.

PFT file
--------

``PFTmap_{YYYY}.nc`` has a ``pft`` dimension of length 15, in the
ORCHIDEE order (:data:`phenonn.utils.config.PFT_NAMES`). The variable is
``pft_frac(pft, site)``. The loader accepts two file layouts:

- a single 3-D variable named ``pft_frac`` on ``(pft, site)``;
- fifteen separate 2-D variables ``pft1_frac … pft15_frac`` on ``(site,)``.

If the file is in the native ORCHIDEE output layout
(``maxvegetfrac(time_counter, veget, lat, lon)``) the loader squeezes
``time_counter`` and renames ``veget`` to ``pft`` automatically. See
:func:`~phenonn.data.lai_dataset._open_pft_array` and
:func:`~phenonn.data.lai_dataset._load_pft_vector_array` for the exact
dispatch.

Selected pixels
---------------

``selected_pixels005_balanced.nc`` defines the target site pool used by the
training and prediction jobs. It carries three coordinates:

- ``site_id`` — string, ``pix_LLLL_OOOOO``
- ``latitude`` — float, degrees
- ``longitude`` — float, degrees

The file is read by
:func:`~phenonn.data.lai_dataset.load_selected_pixels`. When you also
provide ``--parent_map`` (a ``selected_pixels01_1.nc`` produced by the
0.05°→0.1° deduplication step), the training and prediction jobs read
ERA5 features from the deduplicated 0.1° file via each 0.05° site's
containing cell. A parent map with no overlap raises a clear error; a
partial map is reported and the uncovered sites are dropped.

Normalisation statistics
------------------------

``norm_stats.json`` is a flat JSON dictionary keyed by feature name::

    {
      "Tmin":      {"mean": ..., "std": ...},
      "Tmax":      {"mean": ..., "std": ...},
      ...,
      "co2":       {"mean": ..., "std": ...},
      "pft1_frac": {"mean": ..., "std": ...},
      ...,
      "LAI":       {"mean": ..., "std": ...}
    }

The entries are used in two places:

- **Features** — heavy-tailed variables (see
  :data:`phenonn.utils.config.LOG_TRANSFORM_FEATURES`) are passed
  through ``log1p`` before being z-scored; the others are z-scored
  directly.
- **Target** — the LAI target is z-scored with the ``"LAI"`` entry
  unless ``normalize_lai=False`` (or anomaly mode is on).

If ``norm_stats.json`` is absent, features and targets are passed to the
model on their native scales. That is supported but not recommended: it
requires a correspondingly larger learning-rate tuning.

CO2 lookup
----------

One ``Annee_YYYY=VALUE`` per line, with ``VALUE`` in ppm. Comments start
with ``#``. Read by :func:`~phenonn.data.lai_dataset.load_co2_lut`.

When ``add_co2_features = True`` in :mod:`phenonn.utils.config`, the CO2
value for the target year is broadcast as a constant channel over the
whole input window.

Window reconstruction
---------------------

The dataset does not materialise a separate "two-year" file. For a
target year ``Y``, it opens ``ERA5_daily_pixelset_{Y-1}.nc`` and
``ERA5_daily_pixelset_{Y}.nc``, concatenates them along ``time``, and
keeps the last ``seq_length`` days. With the default ``seq_length = 720``,
that is almost exactly two years ending on 31 December of ``Y``.

Consequence: **for every target year ``Y`` in a training or prediction
run, the file for ``Y-1`` must also be present in ``features_dir``.**
Samples whose ``Y-1`` file is missing are silently dropped by the
dataset, and the count of dropped samples is printed at load time.

Data requirements
-----------------

- **Time span** — at least ``seq_length`` days of history per site,
  ending on 31 December of the target year. With the default
  ``seq_length``, at least two consecutive years of feature files.
- **Targets** — the 36 dekadal ``LAI`` values of the target year. NaNs
  are allowed; they are masked out of the loss and skipped in the
  metrics.
- **NaN in features** — not allowed. Any sample whose feature window
  contains a NaN is dropped at dataset construction (the drop count is
  printed).
- **PFT fractions** — all 15 values per site. Missing or all-zero rows
  cause the sample to be dropped.

Directory layout
----------------

A working setup looks like this::

    pixelset1/
    ├── era5_pixelset/
    │   ├── ERA5_daily_pixelset_1992.nc
    │   ├── ERA5_daily_pixelset_1993.nc
    │   └── ...
    ├── LAI_pixelset/
    │   ├── LAI_dekadal_1992.nc
    │   └── ...
    ├── PFT_pixelset/
    │   ├── PFTmap_1992.nc
    │   └── ...
    ├── selected_pixels005_1.nc
    ├── selected_pixels005_balanced.nc
    ├── selected_pixels01_1.nc         # 0.05°→0.1° parent map
    ├── norm_stats.json
    └── CO2_1700_2023_TRENDYv2024.txt

The training and prediction jobs take three directory flags
(``--features_dir``, ``--target_dir``, ``--pft_dir``) plus the auxiliary
paths (``--selected_pixels``, ``--parent_map``, ``--stats_path``,
``--co2_path``), all of which point into this tree. ``--parent_map`` is
required for the 0.1° ERA5 route and omitted for the 0.05° identity route.

Daily LAI mode
--------------

When ``--daily_lai`` is passed to the trainer, the model outputs 365
daily values and the loss is computed on all 365 of them. Two extra
requirements apply:

- ``--daily_target_dir`` must point at a folder of ``LAI_daily_{Y}.nc``
  files, produced offline by linear interpolation.
- Validation still scores only the real dekad observations, taken from
  the regular ``--target_dir`` files, so that the val metric stays
  comparable with non-daily runs.

The helper :func:`~phenonn.data.lai_dataset._clim36_to_365` performs the
same interpolation when a daily climatology is needed for anomaly mode.

Building these files
--------------------

The scripts that produce this layout (ERA5 extraction, satellite LAI
resampling, PFT mosaicking, site selection, normalisation statistics) live in
``phenonn.datagen``; see ``phenonn/datagen/README.md`` for the full pipeline
and run order. They depend on the raw ERA5-Land, ESA-CCI and ORCHIDEE
archives.

If you already have the files, skip straight to :doc:`quickstart`. If
you are starting from scratch, the raw archives and the extraction
scripts are the prerequisite.

Next steps
----------

- :doc:`quickstart` — an end-to-end run in a few minutes.
- :doc:`training` — every training flag and model type.
- :doc:`prediction` — inference flags and output format.
- :doc:`api/modules` — the full API reference.
