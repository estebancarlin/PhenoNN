Installation
============

Prerequisites
-------------

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- Git (for cloning the repository)

Installing uv Package Manager
-----------------------------

uv is a fast Python package installer and resolver. Install it using:

.. code-block:: bash

   # Install uv using curl (Linux/macOS)
   curl -LsSf https://astral.sh/uv/install.sh | sh

You should see output similar to:

.. code-block:: text

   downloading uv 0.11.2 x86_64-unknown-linux-gnu
   installing to /home/username/.local/bin
     uv
     uvx
   everything's installed!

Add uv to your PATH
~~~~~~~~~~~~~~~~~~~

To use uv from anywhere, add it to your shell configuration:

.. code-block:: bash

   # Add to ~/.bashrc
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Reload your shell configuration
   source ~/.bashrc

Verify the installation:

.. code-block:: bash

   uv --version
   which uv

You should see:

.. code-block:: text

   uv 0.11.2
   /home/username/.local/bin/uv

Installing PhenoNN
------------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/kardaneh/PhenoNN.git
      cd PhenoNN

2. **Create a virtual environment**:

   .. code-block:: bash

      uv venv

   This creates a virtual environment in the ``.venv`` directory.

3. **Activate the virtual environment**:

   .. code-block:: bash

      # On Linux/macOS
      source .venv/bin/activate

      # On Windows
      .venv\Scripts\activate

   You should see ``(PhenoNN)`` appear at the beginning of your terminal prompt.

4. **Install PhenoNN with all dependencies**:

   For basic installation:

   .. code-block:: bash

      uv pip install -e .

   For installation with documentation dependencies (for building docs):

   .. code-block:: bash

      uv pip install -e ".[docs]"

   For installation with development dependencies (testing, formatting):

   .. code-block:: bash

      uv pip install -e ".[dev]"

   To install all extras:

   .. code-block:: bash

      uv pip install -e ".[docs,dev]"

Verification
------------

Verify the installation by running:

.. code-block:: bash

   # Check Python version
   python --version

   # Check PhenoNN version
   python -c "import phenonn; print(phenonn.__version__)"

   # Check CLI
   phenonn --help

   # Check PyTorch installation
   python -c "import torch; print(f'PyTorch {torch.__version__}')"

You should see output similar to:

.. code-block:: text

   Python 3.10.12
   0.1.0
   usage: phenonn [-h] [--version] {predict,train,hp-tuning} ...
   PyTorch 2.4.1

Dependencies
------------

PhenoNN automatically installs the following main dependencies:

- **PyTorch** >= 2.4.0: Deep learning framework
- **NumPy** >= 1.20.0: Numerical computing
- **Pandas** >= 2.0.3: Data manipulation
- **Matplotlib** >= 3.3.0: Plotting and visualization
- **Scikit-learn** >= 1.3.2: Machine learning metrics
- **SciPy** >= 1.10.1: Scientific computing
- **TQDM** >= 4.65.1: Progress bars
- **Xarray** >= 2023.1.0: Labeled multi-dimensional arrays
- **NetCDF4** >= 1.7.2: NetCDF file support
- **Seaborn** >= 0.13.2: Statistical data visualization

Optional Dependencies
---------------------

If you installed with extras:

- **docs**: Sphinx and related packages for building documentation
- **dev**: Development tools (pre-commit, ruff, pytest)

CUDA Support (Optional)
-----------------------

If you have a CUDA-capable GPU and want GPU acceleration:

.. code-block:: bash

   # For CUDA 11.8
   uv pip install torch --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   uv pip install torch --index-url https://download.pytorch.org/whl/cu121

Verify CUDA is available:

.. code-block:: bash

   python -c "import torch; print(torch.cuda.is_available())"

Should return ``True`` if CUDA is properly configured.

Troubleshooting
---------------

If you encounter issues:

**1. ModuleNotFoundError**

Ensure your virtual environment is activated:

.. code-block:: bash

   source .venv/bin/activate

**2. PyTorch installation issues**

Install PyTorch separately:

.. code-block:: bash

   uv pip uninstall torch
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu

**3. Permission errors**

Use `--user` flag or ensure proper permissions:

.. code-block:: bash

   uv pip install --user -e .

**4. Python version issues**

PhenoNN requires Python 3.8+. Check your version:

.. code-block:: bash

   python --version

For more help, visit the GitHub repository: https://github.com/kardaneh/PhenoNN
