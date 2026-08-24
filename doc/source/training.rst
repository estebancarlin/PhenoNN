Training Commands
=================

Global Training
---------------

``train-global`` is the primary audited workflow:

.. code-block:: bash

   phenonn train-global \
       --era-dir data/era5_selected/daily \
       --target-dir data/selected_targets \
       --selection data/era5_selected/selected_pixels_era5_valid.nc \
       --output-dir runs \
       --experiment global_gru_raw \
       --type gru \
       --train-years 1993-2014 \
       --validation-years 2015-2016 \
       --hidden-size 64 --num-layers 2 \
       --batch-size 32 --num-epochs 50

Supported model types are ``lstm``, ``gru``, ``transformer``,
``bitransformer``, and ``fcn``. All preserve the repository's model
implementations and use ``Every10DaysWrapper``.

Important sampling options are ``--train-chunks-per-epoch``,
``--train-sites-per-epoch``, ``--train-years-per-epoch``, and
``--max-validation-sites``. Set ``--max-validation-sites 0`` to use all
validation primaries. The default random seed is 42.

Monitoring and Outputs
~~~~~~~~~~~~~~~~~~~~~~

Each epoch prints a JSON object containing sampled sites/years and train and
validation metrics. The run directory contains:

.. code-block:: text

   config.json
   history.json
   best_model.pth

``history.json`` is rewritten after every epoch. There is currently no resume,
early-stopping, scheduler, or TensorBoard integration in ``train-global``.

Per-Site Training
-----------------

.. code-block:: bash

   phenonn train \
       --data_dir data/sites \
       --type lstm \
       --split_mode site --val_fraction 0.2 \
       --output_dir runs --experiment site_lstm \
       --seq_length 365 --num_epochs 50

Use ``--split_mode year --train_years ... --val_years ...`` for temporal
splits. The safest current model choices in this legacy path are LSTM, GRU, FCN,
and linear baselines. Some parser-advertised Transformer variants have stale
factory contracts and should not be treated as validated.

Flat Training
-------------

.. code-block:: bash

   phenonn train-flat \
       --features_csv data/features.csv \
       --target_csv data/targets.csv \
       --type lstm \
       --split_mode site \
       --output_dir runs --experiment flat_lstm

LSTM, GRU, and the flat BiTransformer branch are implemented. ``aelstm`` is
accepted by the parser but not implemented. The legacy native Transformer
factory currently has a constructor mismatch.

Streaming Yearly Training
-------------------------

.. code-block:: bash

   phenonn train-big \
       --features_dir data/yearly_features \
       --target_dir data/yearly_targets \
       --row_min 900 --row_max 1099 \
       --col_min 1500 --col_max 1699 \
       --train_years 1993-2014 \
       --val_years 2015-2018 \
       --type lstm \
       --output_dir runs --experiment big_lstm

The executable filename is singular ``feature_{year}.csv``. ``train-big`` uses
raw inputs and NaN-safe MSE regardless of other parsed loss choices. No matching
``predict-big`` command exists.

Exact Options
-------------

Run ``phenonn COMMAND --help`` for the authoritative parser-owned options and
defaults. Legacy CSV commands use underscores; global commands use hyphens.
