# analysis/ — post-training diagnostics (not packaged)

Hand-run analyses. Nothing here is imported by `phenonn` or covered by CI
(they need `$DATA`, checkpoints and long runs). Layout:

- `diagnostics/` — one-shot scripts, documented in the reference below.

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
LAI/PFT-only reads do not need the parent map.

## Shared prerequisite: predictions CSV

Most CSV-based diagnostics consume the output of `phenonn.prediction.predict`.
For the smoke checkpoint:

```bash
.venv/bin/python -m phenonn.prediction.predict \
    --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth \
    --features_dir "$DATA/era5_pixelset" --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --predict_years 2000-2002 \
    --output_csv runs/smoke_pred.csv
```

The smoke checkpoint is `lstm` + `pft_mixing`, val 2002 (8 sites). Its stored
data dirs are stale: always pass explicit dirs and `--parent_map`. The CSV has
`site_id,year,month,day,doy,lai_pred,lai_obs,...`, which is what the CSV
scripts expect.

## Diagnostics reference

### Data-only scripts

#### `climatology_skill.py`

Reference skill of the climatology itself. Torch-free: run it as a plain
script, not via a `phenonn` import (the package `__init__` imports torch).

- Builds `clim(site, dekad)` = mean LAI over `--clim_years`.
- Predicts the same value for every `--eval_years` year and scores it.
- Reports pooled RMSE/MAE/bias, overall R², site-year R², **interannual R²**
  (the key number: a climatology is ~0 by construction), and the per-site R²
  distribution.
- Writes `clim_predictions.csv`, `summary.txt`, `clim_skill.png`.

```bash
.venv/bin/python analysis/diagnostics/climatology_skill.py \
    --target_dir "$DATA/LAI_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --clim_years 1992-2009 --eval_years 2010-2019 \
    --n_sites 200 --output_dir runs/clim_skill
```

`--n_sites` defaults to 20000 and is slow; start with 200. `--n_sites 0` uses
all sites. Overlapping `clim_years` and `eval_years` prints an optimistic-
baseline warning.

#### `inspect_era5_pixelset.py`

Health check for `ERA5_daily_pixelset_{Y}.nc`. Reports per year: `time`
length (catches 365 vs 366 mismatches), number of selected cells, total values,
NaN count and NaN %. Pools NaN stats per feature over all years. Auto-adds
`SMI` when present.

```bash
.venv/bin/python analysis/diagnostics/inspect_era5_pixelset.py \
    --era5_dir "$DATA/era5_pixelset" \
    --year_start 2000 --year_end 2001 \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --output_csv runs/era5_missing.csv
```

Variants:
- Omit `--selected_pixels`/`--parent_map` to inspect every site in the files.
- `--features Tmin,tp_sum` restricts the feature list.
- Omit `--output_csv` for console-only output.

Mixed time lengths are expected (365/366); `pft_diagnostics.py` aligns both on a
fixed 366-day grid, so they no longer drop years.

#### `diagnose_grid_alignment.py`

Diagnoses geolocation mismatch between ERA5 pixels and LAI/PFT pixels of the
same `site_id`.

- Mode 1 (default): joins the stored ERA5 `latitude`/`longitude` coords to the
  `selected_pixels*.nc` reference coords and reports per-site `d_lat`/`d_lon`
  statistics, the most common rounded offsets, and the worst sites.
- Mode 2 (`--era5_grid RAW.nc`): additionally prints the raw ERA5 grid geometry
  (origin, step, direction, size), the LAI/PFT reference grid, and reproduces
  `lat_idx` / `lon_idx` → coordinate mapping to show which part of the
  assumption breaks.

Console-only; no files.

```bash
.venv/bin/python analysis/diagnostics/diagnose_grid_alignment.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --era5_dir "$DATA/era5_pixelset" --year 2000
```

Add `--era5_grid /path/to/raw/ERA5-Land/file.nc` for mode 2; `--max_report N`
changes how many worst sites are listed (default 20).

#### `spatial_plot_selected_pixels.py`

Plots every site in a `selected_pixels*.nc` on a world map. Uses cartopy
coastlines when installed; otherwise falls back to a plain lon/lat grid with a
warning. Longitudes 0–360 are wrapped to −180…180.

```bash
.venv/bin/python analysis/diagnostics/spatial_plot_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --output runs/spatial_selected_pixels.png
```

Optional: `--markersize`, `--color`, `--dpi`. Input can be any selected-pixels
layout with `site`, `latitude`, `longitude`, including PFT subsets.

#### `print_lai_pft_sites.py`

Print-only (no plot, no files). For explicit `--sites`, prints per year the 15
PFT fractions and all 36 dekadal LAI values, plus mean/min/max. Alignment is by
`site_id`; unknown sites print as NaN rather than being silently mismatched.

```bash
.venv/bin/python analysis/diagnostics/print_lai_pft_sites.py \
    --target_dir "$DATA/LAI_pixelset" --pft_dir "$DATA/PFT_pixelset" \
    --years 2000-2001 --sites pix_0351_01105,pix_0351_01120
```

`--years` accepts `2000-2001` or `2000,2005,2010`. `--pft_min_frac 0.05` hides
small PFT fractions. Site IDs must exist in the cubes.

#### `pft_diagnostics.py`

Diagnostics on one PFT-dominated pool (`selected_pixels_PFT{k}.nc` from
`phenonn.data_creation.filter_pixels_by_pft`). Torch-free and scipy-free,
read-only.

Produces four figures:
- `lai_seasonal.png` — mean dekadal LAI ±1 std, min/max envelope, all
  `(site, year)` curves pooled.
- `features_hist.png` — one histogram per meteo feature, clipped to the
  0.5–99.5 percentile range.
- `features_dist_by_year.png` — one density curve per year per feature.
- `features_seasonal.png` — mean daily feature cycle on a fixed 366-day grid
  (leap and non-leap years both contribute).

```bash
.venv/bin/python analysis/diagnostics/pft_diagnostics.py \
    --selected_pixels "$DATA/greedy_PFT_10000_0.1/selected_pixels_PFT10.nc" \
    --era5_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --year_start 2000 --year_end 2001 --output_dir runs/pft_diag_out
```

LAI is read with the subset's `site_id`s; only ERA5 is translated through
`--parent_map`. `--features` restricts the meteo variables (default
`config.METEO_BASE`).

#### `diagnose_selected_pixels.py`

Combined spatial + PFT-composition diagnostic. Averages PFT fractions per cell
over every `PFTmap_{Y}.nc` in `[year_start, year_end]`, then writes:
- `_dominant_pft.png` — map coloured by majority PFT.
- `_pure_gtNN.png` — cells whose top PFT fraction exceeds `--pure_threshold`.
- `_purity_table.csv/.png` — per-PFT cell counts above 0.5/0.7/0.9/0.95.
- `_density_DEGdeg.png` — number of selected cells per `--block_deg` block.

```bash
.venv/bin/python analysis/diagnostics/diagnose_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_diag.png
```

One throwaway subprocess is spawned per year (HDF5 isolation), so keep the year
range tiny. Output names derive from the `--output` stem. Options:
`--pure_threshold` (default 0.6), `--block_deg` (10), `--raster_deg` (0.1),
`--dpi`.

### Predictions-CSV scripts

#### `annual_trend_study.py`

Computes per `(site, year)` seven metrics from a predictions CSV or from raw
LAI targets: `LAI_tot`, SOS/EOS at 10 % and 25 % of seasonal amplitude, and
length of season `LOS_10`/`LOS_25`. Plots the yearly median across sites with
IQR envelopes for observation and prediction, one PNG per metric. Writes
`metrics_obs.csv` and, when predictions are present, `metrics_pred.csv`.

```bash
.venv/bin/python analysis/diagnostics/annual_trend_study.py \
    --predictions runs/smoke_pred.csv --years 2000-2002 --output_dir runs/trend_out
```

Variants:
- Obs-only from raw pixelsets:
  ```bash
  .venv/bin/python analysis/diagnostics/annual_trend_study.py \
      --target_dir "$DATA/LAI_pixelset" \
      --sites_mode random --n_sites 500 \
      --output_dir runs/baseline_trends
  ```
  This reads `LAI_dekadal_{Y}.nc` (legacy `target_{Y}.csv` is also supported).
- Restrict sites with `--sites_mode explicit --sites ID1,ID2`.
- Add `--clim_years 1992-2009` for a horizontal observation-climatology
  reference line; also writes `metrics_climatology.csv`.
- `--min_valid` (default 34 valid dekads) and `--min_amplitude` (0.3) protect
  SOS/EOS from missing/noisy curves.

#### `climatology_baseline.py`

Compares the model against a climatology baseline built from the same
predictions CSV:

```text
SS = 1 − MSE_model / MSE_clim
```

`SS > 0` means the model beats climatology. Default is leave-one-year-out
(the baseline for year `y` excludes `y`). Requires at least two years.

```bash
.venv/bin/python analysis/diagnostics/climatology_baseline.py \
    --predictions runs/smoke_pred.csv --eval_years 2002
```

Writes `<base>_clim_per_site.csv`, `_clim_scatter.png`,
`_clim_skill_hist.png`, `_clim_skill_per_doy.png`, and optionally
`_clim_lai_curves_all.png` / `_clim_lai_curves.png`.

Variants:
- `--pooled` — faster, slightly optimistic baseline (includes target year).
- `--clim_years 1992-2009` — fixed reference period, applied uniformly to all
  evaluated rows (LOO disabled). If those years are not in the CSV, pass
  `--clim_target_dir "$DATA/LAI_pixelset"`.
- `--output runs/my_base` — base path for outputs.
- `--n_curves 6 --pft_dir "$DATA/PFT_pixelset"` — add curve grids and PFT
  labels.

#### `pft_amplitude_correlation.py`

Separates two failure modes per dominant PFT:
- **(a) amplitude damping** — predicted/observed correlation is good but the
  predicted seasonal amplitude is too small.
- **(b) lack of predictability** — correlation is ~0.

Assigns each site its dominant PFT from `PFTmap_{pft_year}.nc`, fits the annual
harmonic to each complete `(site, year)` curve, and aggregates median Pearson
r, amplitude ratio `pred/obs`, absolute phase shift (dekads), and amplitude.
Writes `pft_diagnostic_summary.csv`, `mean_curves_per_pft.png`,
`skill_amplitude_per_pft.png`.

```bash
.venv/bin/python analysis/diagnostics/pft_amplitude_correlation.py \
    --predictions runs/smoke_pred.csv --pft_dir "$DATA/PFT_pixelset" \
    --output_dir runs/pft_amp_out
```

Variants:
- `--column norm` uses `lai_pred_norm`/`lai_obs_norm`; use this for
  anomaly-mode runs, where real LAI curves share the climatological cycle.
- `--pft_year 2000` selects the PFT map year (default: smallest year in CSV).

#### `pft_curves.py`

Loads a checkpoint trained with `--pft_mixing`, rebuilds the model, and plots
the 15 “pure” per-PFT LAI curves plus the area-weighted prediction and
observations for each selected `(site, year)`. PFTs below `--min_frac` are
drawn in thin grey.

```bash
.venv/bin/python analysis/diagnostics/pft_curves.py \
    --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth \
    --features_dir "$DATA/era5_pixelset" --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --n_sites 2 --year 2002 --output runs/pft_curves.png
```

Fails if the checkpoint is not `--pft_mixing`. Variants:
- Site selection precedence: `--sites` > `--selected_pixels` > checkpoint
  `val_site_ids` (random `--n_sites`, default 6).
- `--year` defaults to the checkpoint's `val_years`.
- `--min_frac` (0.02), `--ncols` (3), `--seed` (42).
- Anomaly-mode checkpoints print a warning: curves are anomalies, not physical
  LAI.
