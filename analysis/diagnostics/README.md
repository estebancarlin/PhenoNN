# analysis/diagnostics/ — one-shot scripts

Run path-style from the repo root (see `../README.md` for setup and full
commands). Every script has `--help`.

| Script | Purpose |
| --- | --- |
| `annual_trend_study.py` | LAI trend analysis from a predictions CSV |
| `climatology_baseline.py` | Climatology predictions from a predictions CSV |
| `climatology_skill.py` | Climatology-only reference skill |
| `diagnose_grid_alignment.py` | ERA5 / pixel-grid alignment check |
| `diagnose_selected_pixels.py` | Selected-pixels diagnostics |
| `inspect_era5_pixelset.py` | ERA5 pixelset inspector |
| `pft_amplitude_correlation.py` | Amplitude correlation from predictions |
| `pft_curves.py` | Per-PFT LAI curves from a checkpoint |
| `pft_diagnostics.py` | PFT diagnostics on a pixel pool |
| `print_lai_pft_sites.py` | Print LAI/PFT values for given sites |
| `spatial_plot_selected_pixels.py` | Spatial plot of the selected pixels |
