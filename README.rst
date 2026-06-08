PhenoNN: Phenology Neural Networks
==================================

|Python Version| |License| |Docs| |Code Style|

PhenoNN is a deep learning framework for phenology prediction using LSTM, GRU,
and Transformer models. It predicts Green Chromatic Coordinate (GCC) and Leaf
Area Index (LAI) from climate data, supporting multiple plant functional types
(PFTs) including Deciduous Broadleaf (DB), Evergreen Needleleaf (EN), and
Grassland (GR). PhenoNN has been developed in the context of the **AI4PEX project** (Research Focus LAND).
AI4PEX is focused on enhancing our understanding of how terrestrial ecosystems respond to climate change
and the feedback of increased atmospheric CO2 levels to the climate system. The project aims to reduce uncertainties
and enhance process representation, namely:

- **Hybrid Modelling and History Matching**: to better predict the instantaneous vegetation responses to water and heat stress.
- **Leverage Deep Learning**: approaches, such as Long-Short Term Memory networks, to simulate phenology and enhance online deep learning frameworks to represent plant carbon dynamics and explore tree mortality drivers.
- **Temperature Sensitivity of Decomposition**: Address the challenge of understanding how temperature affects soil decomposition, which is crucial for ecosystem carbon turnover and land-atmosphere carbon responses to warming.
- **Land-Atmosphere Feedbacks**: Improve the representation of processes that control energy feedbacks to the atmosphere, including regional climate extremes and land carbon uptake, to reduce uncertainties in projected warming trends.

By focusing on these areas, AI4PEX aims to provide a more accurate representation of ecosystem dynamics and feedbacks in climate models.

.. |Python Version| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg
   :target: https://www.python.org/
   :alt: Python Version

.. |License| image:: https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: License: CC BY-NC-SA 4.0

.. |Docs| image:: https://img.shields.io/badge/docs-sphinx-brightgreen.svg
   :target: https://phenonn.readthedocs.io/
   :alt: Documentation

.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style: Black

Table of Contents
-----------------

.. contents:: :local:
   :depth: 2

Features
--------

- **Multiple architectures**: LSTM, LSTM with Attention, GRU, and Transformer models
- **Flexible data formats**: Per-site CSV or flat CSV (features + targets)
- **Comprehensive feature engineering**: GDD, CDD, and Botta onset features
- **Multiple PFT support**: Pre-configured for DB, EN, and GR plant types
- **Easy-to-use CLI**: Simple command-line interface
- **Cross-validation**: Leave-site-out and year-based splitting
- **Hyperparameter tuning**: Integration with Optuna
- **GPU acceleration**: CUDA support for faster training
- **Rich visualizations**: Built-in plotting for model evaluation

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.8 or higher # (3.8, 3.9, 3.10, 3.11, 3.12)
- PyTorch 1.10 or higher
- CUDA-capable GPU (optional, for faster training)
- Git

Quick Install
~~~~~~~~~~~~~

1. **Install uv package manager**:

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
      source ~/.bashrc

2. **Clone and install PhenoNN**:

   .. code-block:: bash

      git clone https://github.com/kardaneh/PhenoNN.git
      cd PhenoNN
      uv venv --python 3.8 # or 3.9, 3.10, 3.11, 3.12
      source .venv/bin/activate
      uv pip install -e . # or uv pip install -e ".[ci,dev]" for development dependencies

3. **Verify installation**:

   .. code-block:: bash

      python tests/test_phenonn_installation.py
      # Should print "SUCCESS! PhenoNN is fully installed and ready to use!"

For detailed installation instructions, see the
`Installation Guide <https://phenonn.readthedocs.io/en/latest/installation.html>`_.

Quick Start
-----------

Per-site CSV Format
~~~~~~~~~~~~~~~~~~~

Train a model using individual site CSV files:

.. code-block:: bash

   # Train an LSTM model
   phenonn train --data_dir ./data/DB/ --type lstm --hidden_size 128 --num_epochs 50

   # Predict using trained model
   phenonn predict --checkpoint ./runs/exp01/checkpoints/best_model.pth --data_dir ./data/DB/

Flat CSV Format
~~~~~~~~~~~~~~~

Train a model using flat feature and target CSV files:

.. code-block:: bash

   # Train on flat CSV format
   phenonn train-flat \
       --features_csv data/features.csv \
       --target_csv data/targets.csv \
       --type lstm \
       --hidden_size 128

   # Predict on flat CSV format
   phenonn predict-flat \
       --checkpoint runs/exp_flat/checkpoints/best_model.pth

Python API
~~~~~~~~~~

Use PhenoNN programmatically in your Python code:

.. code-block:: python

   import phenonn
   import torch
   from phenonn.models import RNN_LSTM
   from phenonn.data import PhenoCamDataset
   from phenonn.training.train import run_training

   # Load data
   dataset = PhenoCamDataset(
       site_files=['DB_site1.csv', 'DB_site2.csv'],
       norm_stats=norm_stats,
       pft_list=['DB', 'EN', 'GR']
   )

   # Create model
   model = RNN_LSTM(
       feature_channel=31,
       output_channel=1,
       hidden_size=64,
       num_layers=2
   )

   # Train model
   run_training()

Data Format
-----------

Per-site CSV Format
~~~~~~~~~~~~~~~~~~~

Each site should have its own CSV file with the naming pattern: `{PFT}_{site}.csv`

**Required columns**:
- year, doy (day of year)
- tmin, tmax (temperature)
- daylength, vpd, prcp, srad, swe
- mat, map (static features)

**Optional columns**:
- lat, lon, elev (site features)
- clay, sand, silt, ph (soil features)

Flat CSV Format
~~~~~~~~~~~~~~~

For large-scale experiments, use flat CSV files:

**features.csv** - Daily data with columns:
- site_id, date, year, month, day
- pft1_frac..pft15_frac (PFT fractions)
- tmin, tmax, daylength, prcp, srad, vpd, swe

**targets.csv** - Sparse LAI observations:
- site_id, date, year, month, day, LAI

For detailed data preparation, see the
`Data Preparation Guide <https://phenonn.readthedocs.io/en/latest/data_preparation.html>`_.

Command Line Interface
----------------------

Main Commands
~~~~~~~~~~~~~

.. code-block:: bash

   # Show all commands
   phenonn --help

   # Train on per-site CSVs
   phenonn train --data_dir PATH --type lstm --hidden_size 64

   # Train on flat CSVs
   phenonn train-flat --features_csv FILE --target_csv FILE

   # Predict with per-site CSVs
   phenonn predict --checkpoint FILE --data_dir PATH

   # Predict with flat CSVs
   phenonn predict-flat --checkpoint FILE

Train Command Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Model options
   --type {lstm,gru,transformer,bitransformer}
   --hidden_size INT          # LSTM/GRU hidden size (default: 32)
   --num_layers INT           # Number of layers (default: 2)
   --seq_length INT           # Window length in days (default: 365)

   # Data options
   --data_dir PATH            # Directory with site CSVs
   --split_mode {site,year}   # Split strategy (default: site)
   --val_fraction FLOAT       # Fraction for validation (default: 0.2)

   # Training options
   --num_epochs INT           # Number of epochs (default: 50)
   --batch_size INT           # Batch size (default: 32)
   --learning_rate FLOAT      # Learning rate (default: 2e-3)
   --patience INT             # Early stopping patience (default: 10)

Predict Command Options
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   --checkpoint PATH          # Path to best_model.pth
   --data_dir PATH            # Directory with site CSVs
   --predict_years YEARS      # Comma-separated years or 'all'
   --predict_sites {val,train,all}  # Which sites to predict
   --output_csv PATH          # Output file path

Examples
--------

Training Examples
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic LSTM training
   phenonn train --data_dir ./data/DB/ --type lstm --num_epochs 100

   # Transformer with year-based split
   phenonn train \
       --data_dir ./data/DB/ \
       --type transformer \
       --split_mode year \
       --train_years 2000-2020 \
       --val_years 2021-2022 \
       --embed_size 64 \
       --nhead 4

   # Flat CSV training with Transformer
   phenonn train-flat \
       --features_csv data/features.csv \
       --target_csv data/targets.csv \
       --type transformer \
       --seq_length 720 \
       --num_epochs 100

Prediction Examples
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Predict on validation sites
   phenonn predict \
       --checkpoint runs/exp01/checkpoints/best_model.pth \
       --data_dir ./data/DB/

   # Predict on all sites for specific years
   phenonn predict \
       --checkpoint runs/exp01/checkpoints/best_model.pth \
       --data_dir ./data/DB/ \
       --predict_sites all \
       --predict_years 2022,2023

   # Flat CSV prediction
   phenonn predict-flat \
       --checkpoint runs/exp_flat/checkpoints/best_model.pth

Python Examples
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from phenonn.models import RNN_LSTM, EncoderTorch
   from phenonn.data import PhenoCamDataset, LAIDataset
   from phenonn.utils import Logger

   # Create LSTM model
   model = RNN_LSTM(
       feature_channel=31,
       output_channel=1,
       hidden_size=128,
       num_layers=2
   )

   # Create Transformer model
   model = EncoderTorch(
       feature_channel=31,
       output_channel=1,
       embed_size=64,
       num_layers=4,
       heads=8,
       seq_length=365
   )

   # Dataset for per-site CSVs
   dataset = PhenoCamDataset(
       site_files=['DB_site1.csv', 'DB_site2.csv'],
       norm_stats=norm_stats,
       pft_list=['DB', 'EN', 'GR'],
       seq_length=365
   )

   # Dataset for flat CSVs
   dataset = LAIDataset(
       features_csv='features.csv',
       target_csv='targets.csv',
       norm_stats=norm_stats,
       seq_length=720
   )

Project Structure
-----------------

.. code-block:: text

   PhenoNN/
   ├── phenonn/                      # Main package
   │   ├── __init__.py              # Package initialization
   │   ├── version.py               # Version information
   │   ├── cli.py                   # Command-line interface
   │   ├── data/                    # Data handling
   │   │   ├── dataset.py           # Per-site CSV dataset
   │   │   ├── dataset_flat.py      # Flat CSV dataset
   │   │   ├── feature_engineering.py  # GDD, CDD features
   │   │   └── normalization.py     # Normalization utilities
   │   ├── models/                  # Model architectures
   │   │   ├── rnn.py               # LSTM, GRU models
   │   │   ├── transformer.py       # Transformer models
   │   │   └── fcn.py               # Fully connected networks
   │   ├── training/                # Training logic
   │   │   ├── train.py             # Per-site CSV training
   │   │   ├── train_flat.py        # Flat CSV training
   │   │   └── hp_tuning.py         # Hyperparameter tuning
   │   ├── prediction/              # Prediction logic
   │   │   ├── predict.py           # Per-site CSV prediction
   │   │   └── predict_flat.py      # Flat CSV prediction
   │   └── utils/                   # Utility modules
   │       ├── logger.py            # Logging utilities
   │       ├── diagnostics.py       # Plotting functions
   │       ├── evaluater.py         # Loss functions
   │       └── wrappers.py          # Model wrappers
   ├── example/                     # Example data
   │   ├── testdata/                # Test climate data
   │   ├── lstm_models/             # Pre-trained models
   │   └── *.csv                    # Example CSV files
   ├── doc/                         # Documentation
   ├── tests/                       # Unit tests
   ├── pyproject.toml               # Project configuration
   └── README.rst                   # This file

Documentation
-------------

Full documentation is available at: https://phenonn.readthedocs.io/

- `Installation Guide <https://phenonn.readthedocs.io/en/latest/installation.html>`_
- `Quickstart <https://phenonn.readthedocs.io/en/latest/quickstart.html>`_
- `Data Preparation <https://phenonn.readthedocs.io/en/latest/data_preparation.html>`_
- `Training Guide <https://phenonn.readthedocs.io/en/latest/training.html>`_
- `Prediction Guide <https://phenonn.readthedocs.io/en/latest/prediction.html>`_
- `API Reference <https://phenonn.readthedocs.io/en/latest/api/modules.html>`_

Build documentation locally:

.. code-block:: bash

   cd doc
   pip install -r requirements.txt  # or uv pip install sphinx sphinx-rtd-theme
   make html
   firefox build/html/index.html

Contributing
------------

Contributions are welcome! Please see our
`Contributing Guide <https://phenonn.readthedocs.io/en/latest/contributing.html>`_.

Development Setup
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/kardaneh/PhenoNN.git
   cd PhenoNN

   # Install with development dependencies
   uv venv --python 3.8
   source .venv/bin/activate
   uv pip install -e ".[dev]"

   # Run tests
   pytest tests/ # not yet implemented

   # Check code style
   pre-commit run --all-files

   # Build documentation
   cd doc && make html

Testing
~~~~~~~

.. code-block:: bash

   # Check the installation
   python tests/test_phenonn_installation.py

   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_dataset.py


License
-------

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license,
  and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must
  distribute your contributions under the same license.

.. image:: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: CC BY-NC-SA 4.0

For more details, see: https://creativecommons.org/licenses/by-nc-sa/4.0/

Citation
--------

If you use PhenoNN in your research, please cite:

.. code-block:: bibtex

   @software{ardaneh_phenonn_2024,
     author = {Barbu, Stefan and Ardaneh, Kazem},
     title = {PhenoNN: Deep Learning for Phenology Prediction},
     year = {2024},
     url = {https://github.com/kardaneh/PhenoNN},
     doi = {10.5281/zenodo.xxxxxxx}
   }

Acknowledgments
---------------

- CNRS / IPSL / Sorbonne University for institutional support
- Max-Planck-Institute for Biogeochemistry Jena
- All contributors and users of PhenoNN

Contact
-------

- **Author**: Kazem Ardaneh
- **Email**: kardaneh@ipsl.fr
- **GitHub**: https://github.com/kardaneh
- **Institution**: IPSL / CNRS / Sorbonne University

---

**PhenoNN** - Making phenology prediction accessible with deep learning.
