PhenoNN: Deep Learning for Phenology Prediction
================================================

|Python Version| |License| |Docs| |Code Style|

PhenoNN is a deep learning package that uses LSTM (Long Short-Term Memory) networks to predict
Green Chromatic Coordinate (GCC) from climate data. It supports multiple plant functional types
(PFTs) including Deciduous Broadleaf (DB), Evergreen Needleleaf (EN), and Grassland (GR).

.. image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg
   :target: https://www.python.org/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-BSD-green.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License

.. image:: https://img.shields.io/badge/docs-sphinx-brightgreen.svg
   :target: https://phenonn.readthedocs.io/
   :alt: Documentation

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style: Black

Features
--------

- **LSTM-based prediction**: Uses Long Short-Term Memory networks for time series prediction
- **Multiple PFT support**: Trained models for DB, EN, and GR plant functional types
- **Flexible feature selection**: Choose from 6, 8, 9, or 14 climate features
- **Easy-to-use CLI**: Simple command-line interface for predictions and training
- **Cross-validation**: Leave-one-site-out cross-validation for robust evaluation
- **GPU acceleration**: CUDA support for faster training and inference
- **Ensemble predictions**: Combines multiple models for improved accuracy

Table of Contents
-----------------

.. contents:: :local:
   :depth: 2

Installation
------------

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- Git

### Quick Install

1. **Install uv package manager**:

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
      source ~/.bashrc

2. **Clone and install PhenoNN**:

   .. code-block:: bash

      git clone https://github.com/kardaneh/PhenoNN.git
      cd PhenoNN
      uv venv
      source .venv/bin/activate
      uv pip install -e .

3. **Verify installation**:

   .. code-block:: bash

      python -c "import phenonn; print(phenonn.__version__)"
      phenonn --help

For detailed installation instructions, including CUDA setup and troubleshooting,
see the `Installation Guide <https://phenonn.readthedocs.io/en/latest/installation.html>`_.

Quick Start
-----------

### Predict GCC using pre-trained models

The simplest way to use PhenoNN is to predict GCC using pre-trained models:

.. code-block:: bash

   # Predict for Grassland (GR) PFT using example data
   phenonn predict GR 4 ./example

   # Predict for Deciduous Broadleaf (DB) PFT
   phenonn predict DB 8 ./example

   # Predict for Evergreen Needleleaf (EN) PFT
   phenonn predict EN 6 ./example

The predictions will be saved as `gcc_pred_test_{PFT}_mfull.csv`.

### Train your own models

Train new models for your specific PFT:

.. code-block:: bash

   # Train a model for DB PFT with 8 features
   phenonn train full DB 8 gcc_lowess 8 --epochs 100

   # Train with hyperparameter tuning
   phenonn hp-tuning full GR 8 gcc_lowess 4

### Python API

Use PhenoNN programmatically in your Python code:

.. code-block:: python

   from phenonn import run_lstm_pred, run_lstm_train
   from phenonn import LSTM, PhenoDataset

   # Make predictions
   run_lstm_pred(
       m='full',
       pft='GR',
       batch_size=4,
       input_path='./example'
   )

   # Train a model
   run_lstm_train(
       m='full',
       pft='DB',
       nr_features=8,
       target='gcc_lowess',
       batch_size=8
   )

   # Load data and model directly
   dataset = PhenoDataset('./example/testdata/', 'GR')
   model = LSTM(target_size=1, input_size=8, hidden_size=64, num_layers=1)

Project Structure
-----------------

.. code-block:: text

   PhenoNN/
   ├── phenonn/                 # Main package
   │   ├── __init__.py         # Package initialization
   │   ├── version.py          # Version information
   │   ├── cli.py              # Command-line interface
   │   ├── lstm.py             # LSTM model definition
   │   ├── lstm_train.py       # Training functions
   │   ├── lstm_pred.py        # Prediction functions
   │   ├── lstm_hp_tuning.py   # Hyperparameter tuning
   │   └── dataloader_phenodata.py  # Data loading and preprocessing
   ├── example/                # Example data and models
   │   ├── testdata/           # Test climate data
   │   ├── lstm_models/        # Pre-trained models
   │   └── gcc_rcc_mins_site_veg.csv  # Site minima
   ├── doc/                    # Documentation
   ├── tests/                  # Unit tests
   ├── pyproject.toml          # Project configuration
   └── README.rst              # This file

Input Data Format
-----------------

PhenoNN expects climate data in CSV format with daily time series. Each site should have its own file:

- **File naming**: `{PFT}_{site}.csv` (e.g., `GR_bullshoals.csv`)
- **Time span**: Minimum 2 years of data to predict 1 year of GCC
- **Required variables**: tmin, tmax, daylength, vpd, swa, radiation, mat, map

For detailed data preparation instructions, see the `Data Preparation Guide <https://phenonn.readthedocs.io/en/latest/data_preparation.html>`_.

Command Line Interface
----------------------

PhenoNN provides a unified CLI with three main commands:

### Predict

.. code-block:: bash

   phenonn predict [PFT] [BATCH_SIZE] [INPUT_PATH] [OPTIONS]

Options:
- `--m`: Model type (full or block size, default: full)

### Train

.. code-block:: bash

   phenonn train [M] [PFT] [NR_FEATURES] [TARGET] [BATCH_SIZE] [OPTIONS]

Options:
- `--epochs`: Number of epochs (default: 150)
- `--learning-rate`: Learning rate (default: 0.01)
- `--hidden-size`: LSTM hidden size (default: 64)
- `--dropout`: Dropout rate (default: 0.0)
- `--patience`: Early stopping patience (default: 30)

### Hyperparameter Tuning

.. code-block:: bash

   phenonn hp-tuning [M] [PFT] [NR_FEATURES] [TARGET] [BATCH_SIZE]

Examples
--------

### Basic prediction

.. code-block:: bash

   # Predict GCC for Grassland sites
   phenonn predict GR 4 ./example

   # Output saved to: gcc_pred_test_GR_mfull.csv

### Training a model

.. code-block:: bash

   # Train a model for DB PFT with custom parameters
   phenonn train full DB 8 gcc_lowess 8 \
       --epochs 200 \
       --learning-rate 0.005 \
       --hidden-size 128 \
       --dropout 0.2 \
       --patience 50

### Hyperparameter optimization

.. code-block:: bash

   # Run hyperparameter tuning for GR PFT
   phenonn hp-tuning full GR 8 gcc_lowess 4

### Python scripting

.. code-block:: python

   import phenonn
   import matplotlib.pyplot as plt

   # Load pre-trained model
   from phenonn import LSTM
   model = LSTM(target_size=1, input_size=8, hidden_size=64, num_layers=1)
   model.load_state_dict(torch.load('example/lstm_models/mfull_GR_8f_0'))

   # Make predictions
   predictions = model(test_data)

   # Plot results
   plt.plot(predictions.detach().numpy())
   plt.title('GCC Predictions')
   plt.show()

Documentation
-------------

Full documentation is available at: https://phenonn.readthedocs.io/

- `Installation Guide <https://phenonn.readthedocs.io/en/latest/installation.html>`_
- `Quickstart <https://phenonn.readthedocs.io/en/latest/quickstart.html>`_
- `API Reference <https://phenonn.readthedocs.io/en/latest/api/modules.html>`_
- `Data Preparation <https://phenonn.readthedocs.io/en/latest/data_preparation.html>`_

To build the documentation locally:

.. code-block:: bash

   cd doc
   make html
   firefox build/html/index.html

Contributing
------------

Contributions are welcome! Please see our `Contributing Guide <https://phenonn.readthedocs.io/en/latest/contributing.html>`_ for details.

### Development Setup

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/kardaneh/PhenoNN.git
   cd PhenoNN

   # Install with development dependencies
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"

   # Run tests
   pytest tests/

   # Check code style
   black phenonn/
   ruff check phenonn/

License
-------

This project is licensed under the BSD License - see the `LICENSE <https://github.com/kardaneh/PhenoNN/blob/main/LICENSE>`_ file for details.

Citation
--------

If you use PhenoNN in your research, please cite:

.. code-block:: bibtex

   @software{ardaneh_phenonn_2024,
     author = {Ardaneh, Kazem},
     title = {PhenoNN: Deep Learning for Phenology Prediction},
     year = {2024},
     url = {https://github.com/kardaneh/PhenoNN},
   }

Acknowledgments
---------------

- The IPSL-AID team for inspiration and project structure
- PyTorch team for the deep learning framework
- All contributors and users of PhenoNN

Contact
-------

- **Author**: Kazem Ardaneh
- **Email**: kazem.arrdaneh@gmail.com
- **GitHub**: https://github.com/kardaneh

---

**PhenoNN** - Making phenology prediction accessible with deep learning 🚀
