# Data and Operations Scripts

Run scripts from the repository root with `python scripts/NAME.py --help`.

Canonical global order:

1. `download_geodes_lai.py`
2. `download_ipsl_forcings.py`
3. `build_selected_pixels.py`
4. `build_stratified_selection.py`
5. `download_era5_selected.py`
6. `validate_era5_selected.py`
7. `build_selected_targets.py`
8. `validate_selected_targets.py`
9. `compute_global_norm_stats.py` (optional ablation only)

`download_era5_land.py` is a standalone bounded-region CDS downloader and does
not directly produce the selected-site daily files consumed by `train-global`.

See `doc/source/global_pipeline.rst` for complete commands and data flow.

`hpc/` contains environment setup snippets for specific computing platforms;
these are not portable defaults.
