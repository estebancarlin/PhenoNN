Contributing
============

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/estebancarlin/PhenoNN.git
   cd PhenoNN
   uv venv --python 3.8
   source .venv/bin/activate
   uv pip install -e ".[ci,dev]"
   python -m pip install pytest
   pre-commit install

Fork and Upstream Remotes
-------------------------

This working repository uses two remotes:

.. code-block:: text

   origin    https://github.com/estebancarlin/PhenoNN.git
   upstream  https://github.com/kardaneh/PhenoNN.git

Normal branches and changes are pushed to ``origin``. ``upstream`` is a
read-only reference to the laboratory repository for later comparison. Fetching
upstream does not overwrite local work:

.. code-block:: bash

   git fetch upstream
   git log --left-right --graph --oneline main...upstream/main
   git diff main...upstream/main

When upstream changes should be integrated, first create a dedicated branch so
the comparison and conflict resolution remain reviewable:

.. code-block:: bash

   git switch -c sync/upstream-YYYY-MM-DD
   git merge upstream/main
   git push -u origin sync/upstream-YYYY-MM-DD

Do not force-push ``main`` to synchronize repositories. The tag
``upstream-base-2026-08-24`` records the exact upstream commit from which this
fork's global-pipeline work diverged.

Before Submitting Changes
-------------------------

1. Preserve the ``(batch, features, sequence)`` package tensor convention.
2. Keep raw physical inputs as the global default.
3. Do not use validation/test observations to compute normalization statistics.
4. Do not evaluate the locked global test split during model selection.
5. Run the relevant unit tests, pre-commit, and documentation build.
6. Keep generated data, experiments, plots, and checkpoints outside Git.

Repository Organization
-----------------------

- Package code belongs under ``phenonn/``.
- Acquisition and artifact-building tools belong under ``scripts/``.
- Tests mirror active source under ``tests/``.
- User documentation belongs under ``doc/source/``.
- Historical scientific provenance belongs under ``archive/`` and must include
  a README explaining why it is retained and what replaces it.

Do not add browser snapshots, local session transcripts, cache directories, or
machine-specific experiment outputs to the repository.
