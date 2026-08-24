PhenoNN Documentation
=====================

PhenoNN is a Python/PyTorch package for LAI and phenology modelling. Its primary
workflow combines selected-site ERA5-Land forcing, GEOV2 LAI, annual ORCHIDEE
PFT fractions, and atmospheric CO2. Per-site, flat-CSV, and streaming yearly-CSV
workflows are also retained.

Start with :doc:`quickstart`. Use :doc:`global_pipeline` when rebuilding the
audited global dataset from source products.

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
   global_pipeline
   training
   prediction

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
