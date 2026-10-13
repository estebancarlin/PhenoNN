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
- `utils/` (Week 1 move): `model_utils.py`

Import legacy code as `legacy.models.<name>` / `legacy.utils.<name>`
(no `pip` packaging; repo-root namespace packages only).
