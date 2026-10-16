# analysis/ — post-training studies and diagnostics (not packaged)

Hand-run analyses. Nothing here is imported by `phenonn` or covered by CI
(they need `$DATA`, checkpoints and long runs). Layout:

- `diagnostics/` — one-shot scripts (see its README for the per-script table).
- `feature_ablation/`, `perturbation/`, `year_generalization/`,
  `pft_linearity/`, `memory_analysis/` — multi-step studies, each with
  its own README. Invoked with   `python -m analysis.<study>.<script>`
  from the repo root. SLURM `jobs/` are carried as reference (paths are
  Hal/sbarbu-specific); regenerate them from the study READMEs before
  submitting.

## Setup (every time)

```bash
cd /home/ecarlin/PhenoNN
source .venv/bin/activate
export DATA=/data/ecarlin/pixelset1
```

`phenonn` is installed editable in the venv; `analysis` resolves from the
repo root. All run outputs go under `runs/` (gitignored).

## Dual-resolution pixelset

LAI/PFT cubes are 0.05° (`pix_...` ids, 1,649,351 sites); ERA5 cubes are
deduplicated 0.1° (`E...` ids, 649,763 sites). `selected_pixels01_1.nc` is the
bridge (`--parent_map`), same pairing as `scripts/smoke_train.sh`. Any script
reading ERA5 with a `pix_...` pool needs
`--parent_map "$DATA/selected_pixels01_1.nc"` or its site match is empty.

## Run directly on `$DATA` (all verified)

```bash
# Climatology reference (default n_sites=20000 is slow, start small)
.venv/bin/python analysis/diagnostics/climatology_skill.py \
    --target_dir "$DATA/LAI_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --clim_years 1992-2009 --eval_years 2010-2019 \
    --n_sites 200 --output_dir runs/clim_skill

.venv/bin/python analysis/diagnostics/inspect_era5_pixelset.py \
    --era5_dir "$DATA/era5_pixelset" \
    --year_start 2000 --year_end 2001 \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --output_csv runs/era5_missing.csv

.venv/bin/python analysis/diagnostics/diagnose_grid_alignment.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --era5_dir "$DATA/era5_pixelset" --year 2000

.venv/bin/python analysis/diagnostics/spatial_plot_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --output runs/spatial_selected_pixels.png

.venv/bin/python analysis/diagnostics/print_lai_pft_sites.py \
    --target_dir "$DATA/LAI_pixelset" --pft_dir "$DATA/PFT_pixelset" \
    --years 2000-2001 --sites pix_0351_01105,pix_0351_01120

# PFT trio (keep range tiny — one subprocess per year)
.venv/bin/python analysis/diagnostics/diagnose_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_diag.png
.venv/bin/python analysis/diagnostics/diagnose_selected_pixels_bis.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_bis_diag.png
.venv/bin/python analysis/diagnostics/diagnose_selected_pixels_pft.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_pft_diag.png

# TRAINS one net per site — keep tiny, run last
.venv/bin/python analysis/diagnostics/per_pixel_baseline.py \
    --features_dir "$DATA/era5_pixelset" --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --train_years 2000-2001 --val_years 2002 \
    --type lstm --hidden_size 16 --num_layers 1 --num_epochs 2 \
    --n_sites 2 --output_csv runs/per_pixel_baseline.csv

.venv/bin/python analysis/diagnostics/pft_diagnostics.py \
    --selected_pixels "$DATA/greedy_PFT_10000_0.1/selected_pixels_PFT10.nc" \
    --era5_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --year_start 2000 --year_end 2001 --output_dir runs/pft_diag_out
```

## Via the smoke checkpoint

`runs/smoke/smoke_lstm/checkpoints/best_model.pth` is `lstm` + `pft_mixing`,
val 2002 (8 sites). Its stored data dirs are stale: always pass explicit dirs
and `--parent_map`. The predict CSV has exactly the columns the CSV scripts
expect (`site_id,year,month,day,doy,lai_pred,lai_obs,...`).

```bash
# Multi-year predictions: climatology_baseline needs ≥2 years
# (leave-one-year-out, min_years=2) — a single-year CSV yields 0
# climatology rows and exits with a message instead of NaNs.
.venv/bin/python -m phenonn.prediction.predict \
    --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth \
    --features_dir "$DATA/era5_pixelset" --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --predict_years 2000-2002 \
    --output_csv runs/smoke_pred.csv

.venv/bin/python analysis/diagnostics/annual_trend_study.py \
    --predictions runs/smoke_pred.csv --years 2000-2002 --output_dir runs/trend_out

.venv/bin/python analysis/diagnostics/climatology_baseline.py \
    --predictions runs/smoke_pred.csv --eval_years 2002

.venv/bin/python analysis/diagnostics/pft_amplitude_correlation.py \
    --predictions runs/smoke_pred.csv --pft_dir "$DATA/PFT_pixelset" \
    --output_dir runs/pft_amp_out

.venv/bin/python analysis/diagnostics/pft_curves.py \
    --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth \
    --features_dir "$DATA/era5_pixelset" --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --n_sites 2 --year 2002 --output runs/pft_curves.png
```

## Still blocked (no command to give)

`stress_sensitivity` (needs `bitransformer_v2`), `compare_mixing` (needs a
no-mixing checkpoint; smoke is mixing), `plot_stress_sensitivity` (needs the
`.npy`), `phenology_dates` (need THEIA `target_{year}.csv` files).
