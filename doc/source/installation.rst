Installation
============

Requirements
------------

- Python 3.8 or newer
- PyTorch ``>=1.10,<2.5``
- A CUDA-capable GPU is optional but recommended for global training
- Git

Standard Environment
--------------------

.. code-block:: bash

   git clone https://github.com/estebancarlin/PhenoNN.git
   cd PhenoNN
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[ci,dev]"
   python tests/test_phenonn_installation.py

The ``ci`` extra supplies the scientific runtime stack used by CI. The ``dev``
extra supplies pre-commit and Ruff. Install pytest separately if you intend to
use it; it is not currently included in ``dev``.

Windows CUDA Environment
------------------------

The verified local GPU setup uses conda environment ``phenonn``, Python 3.11,
and PyTorch 2.4.1+cu124. Install the CUDA PyTorch wheel appropriate for the
machine before the editable package.

.. code-block:: powershell

   conda activate phenonn
   python -m pip install -e ".[ci,dev]"
   python -m pip install pytest
   python tests/test_phenonn_installation.py

Data Acquisition Extras
-----------------------

Install the data extra for CDS/ARCO and large-array tooling:

.. code-block:: bash

   python -m pip install -e ".[ci,data]"

GEOV2 download requires ``GEODES_API_KEY``. ERA5 acquisition requires accepted
licences and CDS credentials in ``~/.cdsapirc``.

Documentation Environment
-------------------------

.. code-block:: bash

   python -m pip install -r doc/requirements.txt
   cd doc
   make html

The generated site is written to ``doc/build/html``.
