Prediction and Evaluation
=========================

Per-Site Prediction
-------------------

.. code-block:: bash

   phenonn predict \
       --checkpoint runs/site_lstm/checkpoints/best_model.pth \
       --data_dir data/sites \
       --predict_sites val \
       --predict_years 2021,2022 \
       --output_csv runs/site_lstm/predictions.csv

The checkpoint must come from ``train`` and the same site schema/PFT naming must
be available. Plots are written alongside predictions.

Flat Prediction
---------------

.. code-block:: bash

   phenonn predict-flat \
       --checkpoint runs/flat_lstm/checkpoints/best_model.pth \
       --features_csv data/features.csv \
       --target_csv data/targets.csv \
       --predict_sites val \
       --output_csv runs/flat_lstm/predictions.csv

Pass both CSV paths explicitly when moving a checkpoint between machines.

Global Evaluation
-----------------

``evaluate-global`` reconstructs model and data settings from a global
checkpoint and always evaluates primary sites only.

Spatial validation uses held-out validation sites in years already represented
by training:

.. code-block:: bash

   phenonn evaluate-global \
       --checkpoint runs/global_lstm_raw/best_model.pth \
       --split validation --years 2013-2014 \
       --max-sites 500 \
       --output runs/global_lstm_raw/evaluation_spatial_validation.json

Temporal validation uses training-region primaries in held-out years:

.. code-block:: bash

   phenonn evaluate-global \
       --checkpoint runs/global_lstm_raw/best_model.pth \
       --split train --years 2015-2016 \
       --max-sites 500 \
       --output runs/global_lstm_raw/evaluation_temporal_validation.json

Use ``--max-sites 0`` for every matching primary. Evaluate ``--split test
--years 2017-2018`` only once the final model selection is frozen.

Compare Runs
------------

.. code-block:: bash

   phenonn compare-global \
       --runs runs/global_lstm_raw runs/global_gru_raw \
       --output runs/global_comparison.json

Each run needs ``config.json`` and ``history.json``. Files named
``evaluation_*.json`` are included automatically. Supplying ``--output`` writes
both JSON and a same-stem CSV.

Phenocam Evaluation
--------------------

``evaluate-phenocam`` compares physical model LAI with quality-controlled GCC
through correlation, independently scaled seasonal shape, and phenological
timing. It deliberately does not report physical LAI error against GCC.

Prediction input:

.. code-block:: text

   site,date,lai_pred

Observation input:

.. code-block:: text

   site,date,gcc[,roi,quality]

.. code-block:: bash

   phenonn evaluate-phenocam \
       --predictions results/phenocam_lai_predictions.csv \
       --observations data/phenocam_gcc.csv \
       --output-dir results/phenocam \
       --roi-column roi \
       --quality-column quality \
       --valid-quality good,acceptable

Dates are joined exactly and ROIs remain separate. Outputs are aligned rows,
site-year metrics, and a provenance-bearing JSON summary. There is no current
``predict-global`` command, so camera-coordinate prediction inputs must be
prepared externally before this evaluator is used.
