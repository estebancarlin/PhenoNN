# Year-generalization jobs

The scripts submit one job per year split. They source `analysis/_cluster.sh`
for the environment, site presets, dry-run mode, and local execution.

## Scripts

- `submit_year_split.sh`: submits six jobs, one for each split:
  `chrono`, `rand1`, `rand2`, `rand3`, `rand4`, and `coldhot`.
- `run_year_split.sh`: job body for one split. Do not submit it directly.

The job body sets the experiment name to `yeargen_<split>` and skips a split
when its run directory already contains `config.json`.

## Submit the study

`coldhot` needs `runs/year_generalization/year_temp_rank.json`. Create it
first:

```bash
export DATA="$PWD/pixelset1"
export RUNS="$PWD/runs"
export YEAR_ROOT="$RUNS/year_generalization"
export RANK_JSON="$YEAR_ROOT/year_temp_rank.json"

.venv/bin/python -m analysis.year_generalization.rank_years_by_temp \
    --features_dir "$DATA/era5_pixelset" \
    --year_start 1992 --year_end 2019 \
    --out "$RANK_JSON"
```

Then submit all splits:

```bash
export TEMP_RANK_JSON="$RANK_JSON"
export PHENONN_SITE=hal
bash analysis/year_generalization/jobs/submit_year_split.sh
```

On HAL the wrapper defaults to `YEAR_SUBSET=0.25` and `PHENONN_MEM=120G`.
Use `PHENONN_SITE=leonardo` for Leonardo or `PHENONN_SITE=local` for a local
run. Override the subset with `YEAR_SUBSET=...` when using larger nodes.

## Change the year pool

The default pool is `1992-2019`. Override it with `ALL_YEARS`:

```bash
ALL_YEARS=1995-2019 \
PHENONN_SITE=hal \
bash analysis/year_generalization/jobs/submit_year_split.sh
```

The pool must contain more than 18 years. Keep the same pool for every split
when comparing `Delta R2`.

## Dry run and local execution

Print the submission commands without running them:

```bash
PHENONN_DRY_RUN=1 PHENONN_SITE=hal \
bash analysis/year_generalization/jobs/submit_year_split.sh
```

Run one split in the current shell:

```bash
PHENONN_LOCAL=1 PHENONN_SITE=local SPLIT=chrono \
bash analysis/year_generalization/jobs/run_year_split.sh
```

For a local dry run, use `PHENONN_LOCAL=1 PHENONN_DRY_RUN=1`. The wrapper
prints the `bash` command instead of running it.

## Environment variables

- `DATA`: dataset root. The default is resolved by `_cluster.sh`.
- `ALL_YEARS`: full year pool. The default is `1992-2019`.
- `YEAR_SUBSET`: site subset. Defaults to `0.25` for HAL and `0.5` otherwise.
- `TEMP_RANK_JSON`: year ranking JSON. The default is
  `runs/year_generalization/year_temp_rank.json`.
- `SPLIT`: split name for `run_year_split.sh`.
- `PHENONN_ROOT`: repository root. The scripts set it automatically.
- `PHENONN_VENV`: activation script. The default is `<repo>/.venv/bin/activate`.
- `PHENONN_SITE`: `hal`, `leonardo`, or `local`.
- `PHENONN_DRY_RUN=1`: print the submission commands only.
- `PHENONN_LOCAL=1`: run with `bash` instead of `sbatch`.
- `PHENONN_DEPENDENCY`: optional `afterok:<jobid>` dependency.
- `PHENONN_PARTITION`, `PHENONN_ACCOUNT`, `PHENONN_QOS`, `PHENONN_MEM`,
  `PHENONN_TIME`, `PHENONN_GRES`: override site defaults.

## Outputs

Each split writes under `runs/year_generalization/yeargen_<split>/`, including
the checkpoint, `config.json`, and validation metrics. SLURM logs go to
`runs/year_generalization/logs/`. See the
[study guide](../README.md) for split definitions and result interpretation.
