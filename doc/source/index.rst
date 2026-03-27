.. PhenoNN documentation master file

PhenoNN: Deep Learning for Phenology Prediction
================================================

PhenoNN is a deep learning package that uses LSTM networks to predict Green Chromatic Coordinate (GCC)
from climate data. It supports multiple plant functional types (PFTs) including Deciduous Broadleaf (DB),
Evergreen Needleleaf (EN), and Grassland (GR).

Key Features
------------

- **LSTM-based prediction**: Uses Long Short-Term Memory networks for time series prediction
- **Multiple PFT support**: Trained models for DB, EN, and GR plant functional types
- **Flexible feature selection**: Choose from 6, 8, 9, or 14 climate features
- **Easy-to-use CLI**: Simple command-line interface for predictions and training
- **Cross-validation**: Leave-one-site-out cross-validation for robust evaluation

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
