Testing
=======

Installation Smoke Test
-----------------------

The installation helper is a script, not a pytest-native test module:

.. code-block:: bash

   python tests/test_phenonn_installation.py

Authoritative Unit Tests
------------------------

CI uses an explicit unittest suite:

.. code-block:: bash

   python -m unittest tests.test_transformer tests.test_rnn tests.test_fcn \
       tests.test_utils tests.test_model_utils tests.test_transformerbis \
       tests.test_evaluater tests.test_diagnostics

Global-pipeline changes should also run:

.. code-block:: bash

   python -m unittest tests.test_build_selected_pixels \
       tests.test_build_stratified_selection tests.test_download_era5_land \
       tests.test_download_era5_selected tests.test_validate_era5_selected \
       tests.test_build_selected_targets tests.test_dataset_netcdf \
       tests.test_train_global tests.test_evaluate_global \
       tests.test_evaluate_phenocam tests.test_compare_global

On Windows, set ``MPLBACKEND=Agg`` for plotting tests if Tk is unavailable.
Broad ``pytest tests/`` collection is not authoritative because it collects the
installation script incorrectly and can expose Windows logger teardown issues.

CI-Equivalent Checks
--------------------

.. code-block:: bash

   pre-commit run --all-files
   uv build

Documentation
-------------

.. code-block:: bash

   python -m pip install -r doc/requirements.txt
   cd doc
   make html

Treat Sphinx warnings as documentation defects even when a build completes.
