Training
========

Training Command
----------------

.. code-block:: bash

   phenonn train [m] [pft] [nr_features] [target] [batch_size] [options]

Arguments
---------

- **m**: Block size for shuffling or "full" for no shuffling
- **pft**: Plant functional type (DB, EN, GR)
- **nr_features**: Number of features (6, 8, 9, 14)
- **target**: Target variable to predict
- **batch_size**: Batch size for training

Target Variables
----------------

The following target variables can be predicted:

- ``gcc``: Raw Green Chromatic Coordinate
- ``rcc``: Raw Red Chromatic Coordinate
- ``gcc_lowess``: GCC with LOWESS smoothing
- ``rcc_lowess``: RCC with LOWESS smoothing
- ``gcc_norm``: Normalized GCC
- ``rcc_norm``: Normalized RCC
- ``gcc_lowess_norm``: Normalized GCC with LOWESS
- ``rcc_lowess_norm``: Normalized RCC with LOWESS

Training Parameters
-------------------

Key training parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - ``--epochs``
     - Maximum number of training epochs
     - 150
   * - ``--learning-rate``
     - Initial learning rate
     - 0.01
   * - ``--hidden-size``
     - LSTM hidden layer size
     - 64
   * - ``--dropout``
     - Dropout rate for regularization
     - 0.0
   * - ``--patience``
     - Early stopping patience
     - 30

Cross-Validation
----------------

PhenoNN uses leave-one-site-out cross-validation:
- Training data: All sites except one
- Validation data: The held-out site
- Models are trained for each fold, resulting in multiple models per PFT

Example
-------

.. code-block:: bash

   # Train a model for DB PFT with 8 features
   phenonn train full DB 8 gcc_lowess 8 --epochs 100 --learning-rate 0.01
