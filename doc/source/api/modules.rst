cat > doc/source/api/modules.rst << 'EOF'
API Reference
=============

This page provides the complete API reference for PhenoNN.

Main Package
------------

.. automodule:: phenonn
   :members:
   :undoc-members:
   :show-inheritance:

Data Module
-----------

.. automodule:: phenonn.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __getitem__, __len__

.. automodule:: phenonn.data.dataset_flat
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __getitem__, __len__

.. automodule:: phenonn.data.feature_engineering
   :members:
   :undoc-members:
   :show-inheritance:

Models Module
-------------

.. automodule:: phenonn.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.models.rnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.models.transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.models.transformerbis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.models.fcn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.models.linear_baseline
   :members:
   :undoc-members:
   :show-inheritance:

Training Module
---------------

.. automodule:: phenonn.training
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.training.train
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.training.train_flat
   :members:
   :undoc-members:
   :show-inheritance:

Prediction Module
-----------------

.. automodule:: phenonn.prediction
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.prediction.predict
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.prediction.predict_flat
   :members:
   :undoc-members:
   :show-inheritance:

Utils Module
------------

.. automodule:: phenonn.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.utils.logger
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.utils.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.utils.evaluater
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.utils.model_loader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.utils.model_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: phenonn.utils.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

CLI Module
----------

.. automodule:: phenonn.cli
   :members:
   :undoc-members:
   :show-inheritance:

Version Module
--------------

.. automodule:: phenonn.version
   :members:
   :undoc-members:
   :show-inheritance:
EOF
