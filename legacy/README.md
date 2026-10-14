# Legacy — do not use for new work

Superseded pipelines kept only so old checkpoints / scripts keep importing.
**Deleted January 2027.**

Contents
- `data/`: `dataset.py`, `dataset_flat.py`, `dataset_big.py`
- `training/`: `train.py`, `train_flat.py`, `train_big.py`, `run_sweep_*`,
  `run_daily_extra.py`, `pure_pft_greedy_nn.py`
- `prediction/`: `predict_flat.py`
- `models/` (Week 1 move): `attn_lstm.py`, `dimchangemodule.py`, `fcn.py`,
  `linear_baseline.py`, `transformerbis.py`
- `tests_archive/`: `fcn_test.py`, `transformerbis_test.py` — archived
  2026-10-13 (kept-surface CI no longer runs them). Run manually from the
  repo root, e.g. `python -m unittest legacy.tests_archive.fcn_test -v`.

`model_utils.py` now lives only at `phenonn/utils/model_utils.py`
(kept); the identical `legacy/utils/` copy was removed 2026-10-13.

Import legacy code as `legacy.models.<name>`
(no `pip` packaging; repo-root namespace packages only).
