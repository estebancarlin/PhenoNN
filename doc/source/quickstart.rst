Quick Start
===========

Inspect Commands
----------------

.. code-block:: bash

   phenonn --help
   phenonn train-global --help
   phenonn evaluate-global --help

Global NetCDF Training
----------------------

.. code-block:: bash

   phenonn train-global \
       --era-dir data/era5_selected/daily \
       --target-dir data/selected_targets \
       --selection data/era5_selected/selected_pixels_era5_valid.nc \
       --output-dir runs \
       --experiment global_lstm_raw \
       --type lstm \
       --train-years 1993-2014 \
       --validation-years 2015-2016 \
       --num-epochs 50

Available global model types are ``lstm``, ``gru``, ``transformer``,
``bitransformer``, and ``fcn``. Transformer variants generally require smaller
batch sizes because attention scales quadratically with sequence length.

Per-Site CSV Training
---------------------

.. code-block:: bash

   phenonn train \
       --data_dir data/sites \
       --type lstm \
       --split_mode year \
       --train_years 2000-2020 \
       --val_years 2021-2022 \
       --output_dir runs \
       --experiment site_lstm

   phenonn predict \
       --checkpoint runs/site_lstm/checkpoints/best_model.pth \
       --data_dir data/sites \
       --predict_sites val \
       --output_csv runs/site_lstm/predictions.csv

Flat CSV Training
-----------------

.. code-block:: bash

   phenonn train-flat \
       --features_csv data/features.csv \
       --target_csv data/targets.csv \
       --type lstm \
       --output_dir runs \
       --experiment flat_lstm

   phenonn predict-flat \
       --checkpoint runs/flat_lstm/checkpoints/best_model.pth \
       --features_csv data/features.csv \
       --target_csv data/targets.csv \
       --predict_sites val \
       --output_csv runs/flat_lstm/predictions.csv

Next Steps
----------

- See :doc:`data_preparation` for exact schemas.
- See :doc:`global_pipeline` to build global inputs from source products.
- See :doc:`training` for all training paths and limitations.
- See :doc:`prediction` for split evaluation and run comparison.
