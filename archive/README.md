# Archive

This directory contains historical scientific provenance that is not imported,
packaged, tested, or supported as current PhenoNN code.

## `historical_global_lai/`

Historical global-grid preprocessing and training configuration retained to
document earlier feature formulas, PFT labels, selection logic, and data
contracts. Current replacements live in:

- `scripts/build_selected_pixels.py`
- `scripts/build_stratified_selection.py`
- `scripts/download_era5_selected.py`
- `scripts/build_selected_targets.py`
- `phenonn/data/dataset_netcdf.py`
- `phenonn/training/train_global.py`

Do not execute archived modules as part of the current pipeline. Scientific
equivalence with current artifacts is not assumed unless separately audited.
