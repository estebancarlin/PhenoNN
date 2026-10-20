# Post-training analysis

This directory contains scripts that run after a model has been trained.
Nothing here is imported by the `phenonn` package, and none of it is installed
with `pip`.

The scripts read pixelset files and checkpoints, and write outputs under
`runs/`. A full study can take minutes to hours. CI only runs the fast import
tests in `tests/test_analysis.py` and the feature-ablation tests in
`tests/test_feature_ablation.py`.

## Before you start

Run every command from the repository root:

```bash
cd /path/to/PhenoNN
source .venv/bin/activate
export DATA=/path/to/pixelset1
```

`$DATA` must contain the pixelset files described in the main README. The
analysis modules resolve from the repository root because `phenonn` is
installed in editable mode.

## Quick start: evaluate a checkpoint

This example uses the checkpoint from `scripts/smoke_train.sh`. Replace the
checkpoint path with your own.

1. Generate a predictions CSV:

   ```bash
   python -m phenonn.prediction.predict \
       --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth.tar \
       --features_dir "$DATA/era5_pixelset" \
       --target_dir "$DATA/LAI_pixelset" \
       --pft_dir "$DATA/PFT_pixelset" \
       --parent_map "$DATA/selected_pixels01_1.nc" \
       --predict_sites val --predict_years 2000-2002 \
       --output_csv runs/smoke_pred.csv
   ```

   Most diagnostics use this CSV. Its columns include `site_id`, `year`,
   `doy`, `lai_pred`, and `lai_obs`.

2. Compare the model with a climatology baseline:

   ```bash
   python analysis/diagnostics/climatology_baseline.py \
       --predictions runs/smoke_pred.csv --eval_years 2002
   ```

3. Plot annual trends and seasonal metrics:

   ```bash
   python analysis/diagnostics/annual_trend_study.py \
       --predictions runs/smoke_pred.csv \
       --years 2000-2002 --output_dir runs/trend_out
   ```

The generated files are described in `diagnostics/README.md`.

## Choose a task

| Goal | Start with |
|---|---|
| Check input data or site alignment | [Data checks](diagnostics/README.md#data-checks) |
| Compare predictions with climatology or trends | [Prediction comparisons](diagnostics/README.md#prediction-comparisons) |
| Plot per-PFT curves or skill | [PFT and model checks](diagnostics/README.md#pft-and-model-checks) |
| Measure feature importance | [Feature ablation](feature_ablation/README.md) |
| Submit cluster jobs | [Job wrappers](feature_ablation/jobs/README.md) |

## Common data notes

### Checkpoint names

The trainer writes `runs/<experiment>/checkpoints/run_best_model.pth.tar` by
default. Smoke scripts pass `--prefix ""`, so they write
`best_model.pth.tar`. Analysis scripts accept both names, plus legacy `.pth`
checkpoints.

### Two pixel resolutions

LAI and PFT cubes use 0.05 degree pixels with `pix_...` IDs. ERA5 cubes use
deduplicated 0.1 degree pixels with `E...` IDs.

`selected_pixels01_1.nc` maps the two grids. Any script that reads ERA5 data
for a `pix_...` site pool needs:

```bash
--parent_map "$DATA/selected_pixels01_1.nc"
```

Scripts that read only LAI or PFT data do not need the parent map.

## Feature ablation

Feature ablation retrains the model once per non-PFT feature with that feature
masked out. It then compares each run with a baseline that uses all features.

Run the local smoke test first:

```bash
bash scripts/smoke_feature_ablation.sh
```

For the full study:

```bash
export PHENONN_SITE=hal
bash analysis/feature_ablation/jobs/submit_ablate.sh
python -m analysis.feature_ablation.aggregate --kind features \
    --runs_dir runs/feature_ablation
```

See `feature_ablation/README.md` for the full workflow, grouped and combo
studies, and result interpretation.

## Diagnostics

Every diagnostic script is listed below with a ready-to-paste command and its
main outputs. Run each command from the repository root after the setup steps
above. Use `--help` for the full option list.

### Data checks

#### `inspect_era5_pixelset.py`

Checks ERA5 files for time-length mismatches, site counts, and missing values.

```bash
python analysis/diagnostics/inspect_era5_pixelset.py \
    --era5_dir "$DATA/era5_pixelset" \
    --year_start 2000 --year_end 2001 \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --output_csv runs/era5_missing.csv
```

Omitting `--selected_pixels` and `--parent_map` inspects every site. Omitting
`--output_csv` prints the report to the terminal.

#### `diagnose_grid_alignment.py`

Compares the stored ERA5 coordinates with the LAI/PFT coordinates for the same
site IDs.

```bash
python analysis/diagnostics/diagnose_grid_alignment.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --era5_dir "$DATA/era5_pixelset" --year 2000
```

This script prints to the terminal. Add a raw ERA5 file with
`--era5_grid /path/to/raw/ERA5-Land/file.nc` to inspect the grid geometry.

#### `spatial_plot_selected_pixels.py`

Plots every site in a selected-pixel file on a world map.

```bash
python analysis/diagnostics/spatial_plot_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --output runs/spatial_selected_pixels.png
```

The script uses cartopy when available and falls back to a plain lon/lat plot
otherwise.

#### `print_lai_pft_sites.py`

Prints PFT fractions and all 36 dekadal LAI values for selected sites.

```bash
python analysis/diagnostics/print_lai_pft_sites.py \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --years 2000-2001 \
    --sites pix_0351_01105,pix_0351_01120
```

This script writes no files. Replace the example site IDs with IDs from your
site pool.

#### `pft_diagnostics.py`

Checks one PFT-dominated pool and writes four figures for LAI and meteorological
features.

```bash
python analysis/diagnostics/pft_diagnostics.py \
    --selected_pixels "$DATA/greedy_PFT_10000_0.1/selected_pixels_PFT10.nc" \
    --era5_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --year_start 2000 --year_end 2001 \
    --output_dir runs/pft_diag_out
```

Outputs are `lai_seasonal.png`, `features_hist.png`,
`features_dist_by_year.png`, and `features_seasonal.png`.

#### `diagnose_selected_pixels.py`

Plots the dominant PFT, purity, and spatial density of a selected-pixel file.

```bash
python analysis/diagnostics/diagnose_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_diag.png
```

The script starts one subprocess per year, so keep the year range small.

### Prediction comparisons

#### `climatology_skill.py`

Measures the skill of climatology alone. This is the baseline that a trained
model should beat.

```bash
python analysis/diagnostics/climatology_skill.py \
    --target_dir "$DATA/LAI_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --clim_years 1992-2009 --eval_years 2010-2019 \
    --n_sites 200 --output_dir runs/clim_skill
```

Outputs are `clim_predictions.csv`, `summary.txt`, and `clim_skill.png`.
`--n_sites 0` uses every site.

#### `annual_trend_study.py`

Computes per-site seasonal metrics and plots their yearly medians. The metrics
include total LAI, SOS and EOS, and season length.

```bash
python analysis/diagnostics/annual_trend_study.py \
    --predictions runs/smoke_pred.csv \
    --years 2000-2002 --output_dir runs/trend_out
```

Outputs include `metrics_obs.csv`, `metrics_pred.csv`, and one PNG per metric.
Use `--target_dir "$DATA/LAI_pixelset"` instead of `--predictions` to work from
raw LAI targets.

#### `climatology_baseline.py`

Compares the model with a climatology built from the same predictions CSV.

```bash
python analysis/diagnostics/climatology_baseline.py \
    --predictions runs/smoke_pred.csv --eval_years 2002
```

It reports `SS = 1 - MSE_model / MSE_clim`; a positive value means the model
beats climatology. The default is leave-one-year-out. Use `--pooled` for a
faster but slightly optimistic baseline.

#### `pft_amplitude_correlation.py`

Separates amplitude damping from low predictability for each dominant PFT.

```bash
python analysis/diagnostics/pft_amplitude_correlation.py \
    --predictions runs/smoke_pred.csv \
    --pft_dir "$DATA/PFT_pixelset" \
    --output_dir runs/pft_amp_out
```

Outputs are `pft_diagnostic_summary.csv`, `mean_curves_per_pft.png`, and
`skill_amplitude_per_pft.png`.

### PFT and model checks

#### `pft_curves.py`

Rebuilds a checkpoint trained with `--pft_mixing` and plots the 15 pure PFT
curves, the mixed prediction, and the observations.

```bash
python analysis/diagnostics/pft_curves.py \
    --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth.tar \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --n_sites 2 --year 2002 --output runs/pft_curves.png
```

The script exits with an error if the checkpoint was not trained with
`--pft_mixing`.

## Shared helpers

`_checkpoint_utils.py` contains the checkpoint helpers used by the analysis
scripts:

- `load_state_dict()` accepts current and legacy state-dict keys.
- `find_best_checkpoint()` finds current and legacy checkpoint names.
- `has_completed_run()` checks whether a run has a final checkpoint.

`_cluster.sh` handles HAL, Leonardo, and local runs. It loads site modules when
needed, activates the virtual environment, and adds site-specific `sbatch`
options. HAL and local runs need no modules. Leonardo defaults to
`profile/deeplrn cineca-ai`. Override this with `PHENONN_MODULES`; set it to an
empty value to disable module loading.
