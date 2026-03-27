Quickstart
==========

This guide will help you get started with PhenoNN quickly.

Predicting GCC with Pre-trained Models
--------------------------------------

The simplest way to use PhenoNN is to predict GCC using pre-trained models.

1. **Prepare your data**: Ensure your climate data is in the correct format.

2. **Run prediction**:

   .. code-block:: bash

      phenonn predict GR 4 ./example

   This command:
   - Uses pre-trained models for Grassland (GR) PFT
   - Batch size of 4
   - Reads data from the example directory

3. **Output**: The predictions will be saved to `gcc_pred_test_GR_mfull.csv`.

Training Your Own Models
------------------------

To train a new model for a specific PFT:

.. code-block:: bash

   phenonn train full DB 8 gcc_lowess 8 --epochs 100

Parameters:
- `full`: No shuffling (use full dataset)
- `DB`: Plant functional type (Deciduous Broadleaf)
- `8`: Number of features (dynamic + static climate)
- `gcc_lowess`: Target variable
- `8`: Batch size
- `--epochs 100`: Train for 100 epochs

Hyperparameter Tuning
---------------------

Run hyperparameter tuning to find optimal settings:

.. code-block:: bash

   phenonn hp-tuning full GR 8 gcc_lowess 4

Python API
----------

You can also use PhenoNN programmatically:

.. code-block:: python

   from phenonn import LSTM, PhenoDataset
   from phenonn import run_lstm_pred, run_lstm_train

   # Make predictions
   run_lstm_pred(m='full', pft='GR', batch_size=4, input_path='./example')

   # Train a model
   run_lstm_train(
       m='full',
       pft='DB',
       nr_features=8,
       target='gcc_lowess',
       batch_size=8
   )
