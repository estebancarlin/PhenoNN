.. PhenoNN documentation master file

PhenoNN: Phenology Neural Networks
==================================

PhenoNN is a deep learning package that uses LSTM networks to predict Green Chromatic Coordinate (GCC)
from climate data. It supports multiple plant functional types (PFTs) including Deciduous Broadleaf (DB),
Evergreen Needleleaf (EN), and Grassland (GR). PhenoNN has been developed in the context of the **AI4PEX project**.

Key Features
------------

- **Multiple architectures**: LSTM, LSTM with Attention, GRU, and Transformer models
- **Flexible data formats**: Per-site CSV or flat CSV (features + targets)
- **Comprehensive feature engineering**: GDD, CDD, and Botta onset features
- **Multiple PFT support**: Pre-configured for DB, EN, and GR plant types
- **Easy-to-use CLI**: Simple command-line interface
- **Cross-validation**: Leave-site-out and year-based splitting
- **Hyperparameter tuning**: Integration with Optuna
- **GPU acceleration**: CUDA support for faster training
- **Rich visualizations**: Built-in plotting for model evaluation

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   overview

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   data_preparation
   training
   prediction
   hyperparameter_tuning

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   testing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
