# PhenoNN Agent Guide

## Onboarding Context

- This repository is a standalone Python/PyTorch LAI and phenology modeling package. GEOV2 acquisition and selected-site processing are present under `scripts/`; ORCHIDEE, Fortran90, FTorch, and production coupling code are not present.
- The intended job context is to improve LAI simulation with GEOV2-derived data, validate against independent Phenocam time series, couple the model into ORCHIDEE through FTorch, and assess carbon, water, and energy impacts; preserve that scientific scope when proposing changes.
- Before substantial implementation, summarize the relevant existing pipeline and surface questions about the intended LAI/ORCHIDEE interface rather than assuming missing coupling requirements; use the repository to answer factual questions first.

## Source Layout

- `phenonn/data/` contains per-site CSV (`PhenoCamDataset`), flat feature/target CSV (`LAIDataset`), and streaming yearly-file (`BigLAIDataset`) data paths.
- `phenonn/data/feature_engineering.py` adds annual-reset GDD, CDD, NCD, and Botta forcing features before normalization.
- `phenonn/models/` contains LSTM/GRU, native Transformer, Transformer variants, FCN, and linear baselines; `phenonn/utils/model_loader.py` is the model factory and applies output-shape wrappers.
- The package convention is input tensors `(batch, features, sequence)` and model outputs `(batch, output, sequence)`; `transformerbis` models internally use `(batch, sequence, features)` and are adapted by `permuteWrapper`.
- `phenonn/training/` has per-site, flat-CSV, streaming yearly-file, and selected-site NetCDF workflows. `phenonn/prediction/` has per-site/flat inference plus global evaluation, run comparison, and prepared Phenocam evaluation.
- `scripts/` contains the ordered GEOV2, PFT/CO2, selection, ERA5-Land, target, audit, and normalization pipeline documented in `doc/source/global_pipeline.rst`.
- `archive/` contains historical scientific provenance only and is not imported by current code.
- `phenonn/__init__.py` imports most of the package, selects CUDA when available, and prints device information on first import; import side effects are expected.

## Setup

- `pyproject.toml` requires Python `>=3.8`; the checked-in `.python-version` is `3.8`, while the Windows CUDA environment verified here is conda environment `phenonn` with Python 3.11 and PyTorch `2.4.1+cu124`.
- Project metadata constrains PyTorch to `>=1.10,<2.5`; for the Windows GPU setup use `conda activate phenonn` and install the CUDA wheel explicitly before `python -m pip install -e ".[ci,dev]"`.
- The `dev` extra does not actually include `pytest` despite the docs saying it does; install it separately when using pytest: `python -m pip install pytest`.
- There is no lockfile or conda environment file. CI uses `uv venv` followed by `uv pip install -e .[ci,dev]`; the conda setup above is the practical Windows/CUDA equivalent.

## Commands

- Installation smoke test: `python tests/test_phenonn_installation.py`.
- Authoritative CI unit tests: `python -m unittest tests.test_transformer tests.test_rnn tests.test_fcn tests.test_utils tests.test_model_utils tests.test_transformerbis tests.test_evaluater tests.test_diagnostics`.
- The CI workflow also runs `pre-commit run --all-files` and `uv build`; run those when validating a CI-equivalent change.
- Documentation: from `doc/`, install `doc/requirements.txt`, then run `make html`; the configured Sphinx source is `doc/source/conf.py` and output is `doc/build/html`.
- CLI entry point is `phenonn` from `phenonn.cli:main`; supported commands are `train`, `train-big`, `train-flat`, `train-global`, `predict`, `predict-flat`, `evaluate-global`, `evaluate-phenocam`, and `compare-global`.

## Data And Experiments

- Per-site files are named `{PFT}_{site}.csv` and require daily `year`, `doy`, meteorological variables, static climate/site variables, and LAI/GCC target columns as appropriate; the documented minimum history is two years for a one-year prediction.
- Flat training uses daily `features.csv` plus sparse `targets.csv`; each sample is a `(site_id, year)` pair with 36 LAI observations on days 5, 15, and 25 of each month, and uses 29 features including PFT fractions and derived phenology features.
- Streaming training expects per-year files in separate feature/target directories. Follow executable `dataset_big.py` naming (`feature_{year}.csv` and `target_{year}.csv`), not the inconsistent plural spelling in some docstrings; target year `Y` also needs feature history from `Y-1`.
- Site splits evaluate spatial generalization; year splits evaluate temporal generalization. Normalization statistics are computed from training data for the regular and flat paths, while the streaming `BigLAIDataset` intentionally uses raw physical units.
- Training writes experiment artifacts under `runs/` by default, including checkpoints, normalization statistics, logs, and diagnostic plots; these paths are ignored by git.

## Test Quirks

- `tests/test_phenonn_installation.py` is a script with a parameterized import helper, not a pytest-native test module; run it directly rather than collecting it with `pytest`.
- On Windows, set matplotlib to a non-interactive backend for plotting tests, e.g. PowerShell `$env:MPLBACKEND="Agg"`, because the default Tk backend may be unavailable.
- A broad `pytest tests/` run currently exposes repository issues unrelated to CUDA: logger tests leave file handlers open, causing Windows `PermissionError` during teardown, and the installation helper is collected incorrectly. The CI-style unittest suite is the reliable baseline.

## CI And Style

- CI targets Ubuntu with Python 3.8, installs `.[ci,dev]`, runs the pre-commit hooks, then the explicit unittest list, and finally builds the package.
- Pre-commit applies whitespace/file checks plus Ruff linting and formatting; its pinned hook is Ruff `v0.5.0`, while the project extra declares a newer minimum, so use the configured pre-commit hook when reproducing CI formatting.
- Avoid committing generated `runs/`, `checkpoints/`, `logs/`, `stats/`, `results/`, `data/`, test plots, caches, or build artifacts; the existing `.gitignore` defines these boundaries.
