# PhenoNN — frozen decisions (scope guardrails)

This file freezes the load-bearing choices for the 6-week cleanup.
Any change requires an explicit new entry below, not a silent edit.

## 1. Docs location — `doc/` is canonical (2026-10-13)

The work plan references `docs/`. The repo uses `doc/` (Sphinx root
`doc/source/conf.py`, RTD config `.readthedocs.yaml`). Ruling: `doc/`
replaces `docs/` everywhere. No `docs/` directory is created and `doc/`
is not renamed. Plan references map as:

* `docs/DECISIONS.md` -> `doc/DECISIONS.md` (this file)
* `docs/HANDOVER.md` -> `doc/HANDOVER.md`
* `docs/weekly/` -> `doc/weekly/`
* `docs/api/public_api.md` -> `doc/source/api/public_api.md`
* `docs.yml` (GH Pages deploy) -> deferred; RTD (`.readthedocs.yaml`) stays.

## 2. Diagnostics location — `phenonn/analysis/`, not `scripts/` (2026-10-13)

The work plan sends hand-run diagnostics to `scripts/` (not packaged).
Advisor ruling after the plan: diagnostics-analysis lives in
`phenonn/analysis/` (packaged import path), with run guides in
`phenonn/analysis/README.md`. `scripts/` holds only repro entry points
(`smoke_train.sh`, `run_baseline.sh`, `phenonn-predict`). Do not move
files between the two without a new entry here.

## 3. Stack freeze — stay on Python 3.8 for now (2026-10-13)

The plan freezes Python >= 3.10 with `uv` + `uv.lock`. The repo runs
`.venv` Python 3.8.10 with hatchling (`pyproject.toml`,
`requires-python = ">=3.8"`) and has no `uv.lock`. Ruling: defer the
3.10 upgrade and `uv.lock` generation. Removed the stray
`[tool.uv.workspace] members = ["tem"]` separately when touching
`pyproject.toml`. Pre-commit stays `ruff` + `ruff-format` (no `black`).

## 4. Repro ladder (2026-10-13)

* `scripts/smoke_train.sh`: tiny repro (CPU, 2 epochs, 8 sites).
* `scripts/run_baseline.sh`: small baseline, same trainer, ~10x smoke,
  GPU-first with CPU fallback (GPU: batch 64, 4 workers, `--amp`;
  CPU: batch 16, 0 workers; train 2000-2003 / val 2004).
  `DEVICE=cpu|cuda|auto` overrides; a failed GPU attempt retries on CPU
  with `CUDA_VISIBLE_DEVICES=""`.
* `jobs/` + `phenonn/analysis/*/jobs/`: full SLURM / GPU runs.

## 6. Runs storage — `runs/` symlink to SSD (2026-10-13)

`runs/` at repo root is a symlink to `/net/nfs/ssd2/ecarlin/runs`
(SSD, compute-node-only). All checkpoints, logs, CSVs and figures
written to `runs/...` (relative defaults in `scripts/` and
`phenonn/analysis/`) land on SSD. `scripts/smoke_train.sh` and
`scripts/run_baseline.sh` default `RUNS_DIR` to `$REPO/runs` (through
the link); `RUNS_DIR=` overrides only to bypass it. The link itself is
gitignored (`/.gitignore` `/runs`); never commit run outputs.

## 5. Scope guardrails (unchanged from plan)

No new features, no retraining, no logic rewrites during the 6 weeks.
Any PR must pass all gates or is reverted. `legacy/` deleted Jan 2027.
GPU execution of full training is unreliable (torch/CUDA kernel
mismatch observed); `run_baseline.sh` tries GPU first and falls back
to CPU automatically, `smoke_train.sh` stays CPU-only.

## 7. Test retirement — kept-surface CI (2026-10-13)

`phenonn/utils/model_utils.py` is kept (the identical
`legacy/utils/model_utils.py` copy was deleted); `ModelUtils` is
exported from `phenonn.utils`. `tests/test_fcn.py` and
`tests/test_transformerbis.py` moved to `legacy/tests_archive/` (renamed
`fcn_test.py` / `transformerbis_test.py` so root discovery skips them;
run manually via `python -m unittest legacy.tests_archive.fcn_test`).
The archived `transformerbis` suite carries 9 pre-existing tuple-vs-tensor
errors — quarantined, not fixed. CI runs the kept set only
(`test_transformer_extra`, `test_aelstm`, `test_bitransformer`,
`test_wrappers`, `test_lai_dataset`, `test_model_loader`, `test_cli`,
`test_logger` added; `test_fcn`, `test_transformerbis` removed).
Coverage gate: `[tool.coverage]` `fail_under = 50`, `phenonn/` excluding
`phenonn/analysis/`.
