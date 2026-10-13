Contributing
============

Thanks for considering a contribution to PhenoNN. This page describes
the conventions the repository follows, so that a pull request lands
without back-and-forth.

Code of conduct
---------------

Be respectful. Discussions happen in issues and pull requests; assume
good faith, keep the tone technical, and stay on the topic of the
change.

Report a problem
----------------

Open an issue with:

- a short title describing the symptom,
- the command you ran (the exact ``python -m …`` line),
- the traceback or the unexpected output,
- the version of the package
  (``python -c "import phenonn; print(phenonn.__version__)"``).

If the problem depends on your data, mention which files are involved
(features, targets, PFT) without sharing the data itself.

Development setup
-----------------

The project uses `uv <https://docs.astral.sh/uv/>`_ for packaging and
dependency resolution.

**Base install** (only what the package needs at runtime):

.. code-block:: bash

    git clone https://github.com/kardaneh/PhenoNN.git
    cd PhenoNN
    uv venv
    source .venv/bin/activate
    uv pip install -e .

**Developer install** (adds pytest, pytest-cov, ruff, black, pydocstyle,
pre-commit):

.. code-block:: bash

    uv pip install -e ".[dev]"

**Documentation install** (adds Sphinx and its extensions):

.. code-block:: bash

    uv pip install -e ".[docs]"

**Both**, for a full development environment:

.. code-block:: bash

    uv pip install -e ".[dev,docs]"

The ``-e`` flag installs the package in editable mode: changes to the
Python source are picked up immediately, without a reinstall. Once the
base install is done, ``phenonn`` is importable from anywhere in the
virtual environment, and the ``phenonn`` console script is available.

Once the environment is ready, install the git hook so the linters run
on every commit::

    pre-commit install

Development workflow
--------------------

1. **Fork** the repository on GitHub and clone your fork.
2. **Create a branch** with a descriptive name::

       git checkout -b fix/dataset-nan-window
       git checkout -b feat/attnlstm-dropout

3. **Make your change** in small, self-contained commits.
4. **Run the checks** (see the next section) before pushing.
5. **Open a pull request** against ``main``. Describe the change in
   one paragraph, list the tests you ran, and, if the change touches
   behaviour, mention what to expect on a small smoke run.

Checks before pushing
---------------------

Run all of these locally; the CI runs the same commands.

.. code-block:: bash

    # 1. Fast installation check (imports, public API, model factory, CLI).
    python tests/test_phenonn_installation.py

    # 2. Unit tests (standard-library unittest, no pytest needed).
    python -m unittest discover -s tests

    # 3. Lint, format, and docstring convention.
    #    `pre-commit` is configured to run ruff, black and pydocstyle
    #    on every file. Install it once with `pre-commit install`, or
    #    run it on demand:
    pre-commit run --all-files

    # 4. Documentation build, warnings treated as errors.
    cd docs && make clean && make html

Any failure must be fixed before the pull request is opened — CI will
re-run the exact same commands and refuse the merge otherwise.

Coding conventions
------------------

**Python version.** The package targets Python 3.10 and above. Do not
introduce syntax that requires 3.11 or later (``X | None`` is fine,
``typing.Self`` is not).

**Style.** Formatting is handled by ``black`` (line length 100) and
``ruff`` (see ``pyproject.toml``). Both run through ``pre-commit`` on
every commit; run ``pre-commit run --all-files`` before opening a pull
request to catch anything the hook did not cover.

**Docstrings.** NumPy style, as parsed by Napoleon. Every public class,
function and method must have at least a one-line summary. Functions
with parameters must list them under ``Parameters`` and describe the
return value under ``Returns``. The helpers in
:mod:`phenonn.utils.evaluater` and :mod:`phenonn.models.rnn` are the
reference style.

**Public API.** The public surface of the package is declared in the
``__all__`` lists of the sub-package ``__init__`` files. Adding a
public name means:

1. putting it in the module that defines it,
2. re-exporting it from the sub-package ``__init__``,
3. adding it to the sub-package's ``__all__``.

The test ``tests/test_phenonn_installation.py::test_public_api`` fails
if the three are out of sync, so a new public name is a small PR on its
own.

**Legacy.** Anything that is not reachable from a ``.sh`` file and not
imported by the kept package belongs in ``legacy/``. Do not add code to
``legacy/``. Do not import from ``legacy/`` in new code. The directory
is scheduled for removal in January 2027.

**Diagrams in docstrings.** Avoid box-drawing characters and Unicode
arrows. reStructuredText parses them as malformed markup, and the Sphinx
build fails with ``-W``. Use a fenced ``::`` literal block if you need a
diagram, or describe the flow in prose.

**Tests.** A new module should come with a ``tests/test_<module>.py``
file that:

- derives from ``unittest.TestCase``,
- has a ``run_tests()`` function at the bottom that builds a suite and
  returns ``result.wasSuccessful()``,
- ends with ``if __name__ == "__main__": sys.exit(0 if run_tests() else 1)``.

See ``tests/test_evaluater.py`` for the template.

Adding a new model
------------------

The pattern is:

1. Create ``phenonn/models/<arch>.py`` with a class that takes
   ``(B, C_in, L)`` and returns ``(B, C_out, L)``. Follow the wrapper
   convention: no logic about the 36-day output, no PFT weighting — the
   wrappers do that.
2. Register it in ``phenonn/utils/model_loader.py`` in the
   ``build_model`` and ``build_model_pft`` dispatch tables.
3. Add it to ``phenonn/models/__init__.py`` (import and ``__all__``).
4. Document it in ``docs/source/api/phenonn.models.rst`` by adding an
   ``.. automodule:: phenonn.models.<arch>`` block.
5. Add a case to
   ``tests/test_phenonn_installation.py::test_model_factory`` so a
   forward pass on a tiny input is checked.

If the architecture needs a new training flag, add it in
``train_full_ram.parse_args`` and document it in
``docs/source/training.rst``.

Adding a new analysis
---------------------

Post-training studies go under ``phenonn/analysis/``. Each study is a
directory with an ``__init__.py`` (docstring only, no imports) and one
or more scripts. The scripts are runnable with ``python -m
phenonn.analysis.<study>.<script>`` and are invoked from a SLURM job
under ``phenonn/analysis/<study>/jobs/``.

Analysis code may import the training and prediction internals, but
never modifies them.

Documentation
-------------

Documentation lives in ``docs/source/``. It is built with Sphinx and
the NumPy docstring convention. The API reference is organised one
``.rst`` file per Python package, with every module of that package
inlined as ``.. automodule::`` blocks.

To build the docs locally::

    cd docs
    make clean
    make html

The output lands in ``docs/build/html/``. Open
``docs/build/html/index.html`` in a browser.

Warnings are treated as errors in CI: the ``docs.yml`` workflow calls
``make html`` inside ``docs/`` with ``SPHINXOPTS=-W``, so a pull request
that adds a warning fails the docs job until it is fixed.

Pull request checklist
----------------------

Before opening the pull request, verify:

- [ ] The change is on a branch with a descriptive name.
- [ ] ``python tests/test_phenonn_installation.py`` passes.
- [ ] ``python -m unittest discover -s tests`` passes.
- [ ] ``pre-commit run --all-files`` reports no diff.
- [ ] ``cd docs && make clean && make html`` succeeds with zero warnings.
- [ ] If the change touches the public API, the ``__all__`` lists and
      the Sphinx API pages are updated.
- [ ] If the change touches a training flag, ``docs/source/training.rst``
      lists it.
- [ ] If the change touches a prediction flag, ``docs/source/prediction.rst``
      lists it.
- [ ] The pull request description states what changed and why, and
      mentions the tests you ran.

License
-------

By contributing, you agree that your contribution is licensed under the
same CC BY-NC-SA 4.0 license that covers the repository. See
``LICENSE``.
