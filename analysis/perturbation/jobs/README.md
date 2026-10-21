# Perturbation jobs

The job scripts run the four perturbation scenarios on one or more trained
checkpoints. They source `analysis/_cluster.sh`, so the data path, virtual
environment, site preset, and submission options are shared with the other
analysis jobs.

## Scripts

- `submit_perturbation.sh`: cluster-aware submission wrapper. This is the
  normal entry point.
- `run_perturbation.sh`: job body for one checkpoint. Runs drought, heatwave,
  warming plus CO2, winter frost, and the aggregation.
- `run_perturbation_eval.sh`: job body for one or more experiments. Runs the
  normal validation prediction first, then the four scenarios and the
  aggregation for each experiment.

Do not edit the job bodies to change site or account settings. Use the
environment variables described below.

## One checkpoint

```bash
export RUNS="$PWD/runs"
export PHENONN_SITE=hal
export EXP=baseline_lstm
export CKPT="$RUNS/baseline/baseline_lstm/checkpoints/run_best_model.pth.tar"
bash analysis/perturbation/jobs/submit_perturbation.sh
```

The default output is `$RUNS/perturbation/<experiment>`. Override it with
`OUT`, and change the site count with `NSITES` (`0` means every site):

```bash
export OUT="$RUNS/perturbation/custom"
export NSITES=0

bash analysis/perturbation/jobs/submit_perturbation.sh
```

## Several experiments

Use `MODE=eval` and a space-separated experiment list. The wrapper looks for
`run_best_model.pth.tar`, `best_model.pth.tar`, and the legacy `.pth` names
under `<RUNS_ROOT>/<experiment>/checkpoints/`.

```bash
export PHENONN_SITE=hal
export MODE=eval
export EXPERIMENTS="exp_a exp_b"
export RUNS_ROOT="$RUNS"
bash analysis/perturbation/jobs/submit_perturbation.sh
```

The evaluation also writes the normal validation prediction to
`<RUNS_ROOT>/<experiment>/pred_val.csv`.

## Site presets

```bash
PHENONN_SITE=hal       # HAL
PHENONN_SITE=leonardo  # Leonardo
PHENONN_SITE=local     # local machine, no sbatch
```

The presets live in `analysis/_cluster.sh`. Set `DATA` when the dataset is not
in `./pixelset1` or `../pixelset1`.

## Dry run and local execution

Write the submission commands without running them:

```bash
PHENONN_DRY_RUN=1 PHENONN_SITE=hal \
CKPT="$CKPT" \
bash analysis/perturbation/jobs/submit_perturbation.sh
```

Run the job in the current shell instead of submitting it:

```bash
PHENONN_LOCAL=1 PHENONN_SITE=local \
CKPT="$CKPT" \
NSITES=100 \
bash analysis/perturbation/jobs/submit_perturbation.sh
```

## Environment variables

- `CKPT`: checkpoint for `MODE=run`.
- `OUT`: output directory for `MODE=run`.
- `MODE`: `run` or `eval`. The default is `run`.
- `EXPERIMENTS`: space-separated experiment names for `MODE=eval`.
- `RUNS_ROOT`: root containing the experiment folders for `MODE=eval`.
- `NSITES`: random site subsample. The default is `5000`.
- `DATA`: dataset root. The default is resolved by `_cluster.sh`.
- `PHENONN_ROOT`: repository root. The wrappers set it automatically.
- `PHENONN_VENV`: activation script. The default is `<repo>/.venv/bin/activate`.
- `PHENONN_SITE`: `hal`, `leonardo`, or `local`.
- `PHENONN_DRY_RUN=1`: print the submission commands only.
- `PHENONN_LOCAL=1`: run with `bash` instead of `sbatch`.
- `PHENONN_DEPENDENCY`: optional `afterok:<jobid>` dependency.
- `PHENONN_PARTITION`, `PHENONN_ACCOUNT`, `PHENONN_QOS`, `PHENONN_MEM`,
  `PHENONN_TIME`, `PHENONN_GRES`: override site defaults.

## Outputs

Each experiment writes:

```text
<OUT>/<scenario>/deltas.csv
<OUT>/<scenario>/summary.txt
<OUT>/<scenario>/delta_seasonal.png
<OUT>/<scenario>/delta_hist.png
<OUT>/comparison.csv
<OUT>/comparison.txt
<OUT>/comparison.png
```

See the [study guide](../README.md) for scenario definitions and result
interpretation.
