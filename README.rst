PhenoNN: Phenology Neural Networks
==================================

PhenoNN is a Python/PyTorch package for daily vegetation phenology and Leaf
Area Index (LAI) modelling. The primary workflow trains sequence models on
audited ERA5-Land forcing, GEOV2 LAI targets, annual ORCHIDEE PFT fractions,
and atmospheric CO2. Earlier per-site and flat-CSV workflows remain available.

The package convention is ``(batch, features, sequence)`` for inputs and
``(batch, output, sequence)`` for model outputs. Global models use 720 daily
steps and predict 36 LAI observations on days 5, 15, and 25 of each month.

Supported Workflows
-------------------

``train-global``
   Primary selected-site NetCDF workflow. Tested architectures are ``lstm``,
   ``gru``, ``transformer``, ``bitransformer``, and ``fcn``.

``train`` / ``predict``
   Per-site CSV training and prediction.

``train-flat`` / ``predict-flat``
   Daily feature CSV plus sparse target CSV workflow.

``train-big``
   Streaming yearly-CSV training. There is no matching prediction command.

``evaluate-global`` / ``compare-global``
   Explicit global split evaluation and cross-run comparison.

``evaluate-phenocam``
   Association, seasonal-shape, and timing comparison against prepared
   Phenocam GCC observations. GCC is not treated as physical LAI.

Installation
------------

PhenoNN requires Python 3.8 or newer. PyTorch is constrained to
``>=1.10,<2.5`` by the project metadata.

.. code-block:: bash

   git clone https://github.com/estebancarlin/PhenoNN.git
   cd PhenoNN
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[ci,dev]"
   python tests/test_phenonn_installation.py

On Windows with CUDA, install the appropriate CUDA PyTorch wheel first. The
verified local setup uses the ``phenonn`` conda environment, Python 3.11, and
PyTorch 2.4.1 with CUDA 12.4.

.. code-block:: powershell

   conda activate phenonn
   python -m pip install -e ".[ci,dev]"
   python -m pip install pytest
   python tests/test_phenonn_installation.py

See ``doc/source/installation.rst`` for CI and documentation dependencies.

Global Quick Start
------------------

The global workflow expects this layout:

.. code-block:: text

   ERA_DIR/daily/ERA5_daily_pixelset_1991.nc ...
   TARGET_DIR/lai/LAI_dekadal_1992.nc ...
   TARGET_DIR/pft/PFTmap_1992.nc ...
   TARGET_DIR/CO2_annual.nc
   selected_pixels_era5_valid.nc

Train an unchanged LSTM baseline in raw physical units:

.. code-block:: bash

   phenonn train-global \
       --era-dir ERA_DIR/daily \
       --target-dir TARGET_DIR \
       --selection selected_pixels_era5_valid.nc \
       --output-dir runs \
       --experiment global_lstm_raw \
       --type lstm \
       --train-years 1993-2014 \
       --validation-years 2015-2016

Each epoch is printed as JSON. The run directory contains ``config.json``,
continuously updated ``history.json``, and ``best_model.pth``.

Evaluate spatial and temporal transfer separately:

.. code-block:: bash

   phenonn evaluate-global \
       --checkpoint runs/global_lstm_raw/best_model.pth \
       --split validation --years 2013-2014 \
       --output runs/global_lstm_raw/evaluation_spatial_validation.json

   phenonn evaluate-global \
       --checkpoint runs/global_lstm_raw/best_model.pth \
       --split train --years 2015-2016 \
       --output runs/global_lstm_raw/evaluation_temporal_validation.json

Do not evaluate ``--split test`` until model and checkpoint selection are
frozen. Compare completed runs with:

.. code-block:: bash

   phenonn compare-global \
       --runs runs/global_lstm_raw runs/global_gru_raw \
       --output runs/global_comparison.json

This writes JSON and CSV summaries ranked by joint validation RMSE.

Data Preparation
----------------

The end-to-end global preparation order is:

1. Download GEOV2 and IPSL PFT/CO2 sources.
2. Build valid-pixel masks and a balanced spatial selection.
3. Extract and derive selected-site ERA5-Land forcing.
4. Validate and freeze the ERA artifact.
5. Build selected-site LAI, PFT, and CO2 targets.
6. Validate and freeze target artifacts.
7. Optionally compute train-primary normalization statistics.
8. Train, validate, and compare models.

Every command, input contract, and output is documented in
``doc/source/global_pipeline.rst``. Data schemas for all workflows are in
``doc/source/data_preparation.rst``.

Command Reference
-----------------

.. code-block:: text

   phenonn train
   phenonn train-flat
   phenonn train-big
   phenonn train-global
   phenonn predict
   phenonn predict-flat
   phenonn evaluate-global
   phenonn evaluate-phenocam
   phenonn compare-global

Run ``phenonn COMMAND --help`` for the parser-owned option list. Legacy CSV
commands use underscore flags such as ``--data_dir``; global commands use
hyphenated flags such as ``--era-dir``.

Known Boundaries
----------------

- There is no ``predict-global``, ``predict-big``, resume, or hyperparameter
  tuning command.
- Global training supports sequence-output architectures only. Scalar linear
  baselines and the incompatible legacy ``transformerbis`` are not exposed.
- Global monitoring is JSON/log based; there is no TensorBoard dashboard.
- ``evaluate-phenocam`` requires an externally prepared prediction table and a
  quality-controlled GCC table.
- ORCHIDEE/FTorch coupling is not included in this checkout.
- See ``doc/source/training.rst`` for limitations in older CSV workflows.

Repository Layout
-----------------

.. code-block:: text

   phenonn/            Package datasets, models, training, prediction, utilities
   scripts/            Global acquisition, preparation, validation, HPC setup
   tests/              Unit and pipeline tests
   doc/                Sphinx documentation
   archive/            Retained historical scientific provenance, not imported
   pyproject.toml      Package, dependencies, and CLI entry point

Generated data, experiments, checkpoints, logs, results, caches, and Sphinx
build output are ignored by Git.

Development
-----------

Use the CI-equivalent tests rather than broad pytest collection:

.. code-block:: bash

   python tests/test_phenonn_installation.py
   python -m unittest tests.test_transformer tests.test_rnn tests.test_fcn \
       tests.test_utils tests.test_model_utils tests.test_transformerbis \
       tests.test_evaluater tests.test_diagnostics
   pre-commit run --all-files
   uv build

Build the documentation from ``doc/`` with ``make html`` after installing
``doc/requirements.txt``.

Project Scope
-------------

The scientific scope is to improve GEOV2-informed LAI simulation, validate
seasonality against independent Phenocam observations, and eventually couple a
frozen model into ORCHIDEE through FTorch to assess carbon, water, and energy
impacts. The coupling stage is intentionally outside the current package.

License
-------

See ``LICENSE`` and the source-file headers for licensing terms.
