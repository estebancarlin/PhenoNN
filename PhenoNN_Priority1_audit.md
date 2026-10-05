# PhenoNN — Priority 1 Audit

**Scope:** *Brief `PhenoNN_ORCHIDEE_brief.md`, Priority 1 — "Comprendre et consolider le modèle ML existant".* This document reviews each bullet of Priority 1 against the actual code, tests, docs, and upstream sources (github.com/kardaneh/PhenoNN, phenonn.readthedocs.io), reports verified findings with `file:line` references, and flags consolidation actions.

**Date:** 2026-08-09
**Repo head:** `3ca7f2f` (`fix: update the gitignore for ignoring the test plots`, 2026-06-10) — local branch `main` in sync with `origin/main`
**Package version:** `0.1.0`
**Environment used for verification:** conda env `phenonn` (Python 3.8.20, torch 2.4.1+cu121, CPU)
**Method:** static code reading + runtime verification (model instantiation and forward passes through `load_model`, loss-function factory checks) + CI test suite re-run + web search for published metrics.

---

## Executive summary

The codebase **implements and tests** the models, training loops, and data loaders described in Priority 1, and the CI unit-test gate passes (**150/150 OK**). However, the state is **not "consolidated"**:

1. **Three `--type` paths in the per-site training pipeline are broken at instantiation** (`transformer`, `transformerbis`, `bitransformer` raise `TypeError` because `phenonn/utils/model_loader.py` calls the model constructors with keyword arguments that do not exist in their signatures). Only `lstm`, `gru`, `fcn`, `linear`, `linear_perday`, `1year_lstm`, `1year_bitransformer` work.
2. **Two advertised loss functions (`wmse`, `logcosh`) are not implemented** in `get_loss_function` → selecting them via `--loss_type` crashes at runtime, despite being in the CLI choices.
3. **No benchmark evidence exists anywhere** (repo, docs, GitHub, readthedocs, web) to say which architecture is "best" on RMSE/R²/bias. Production of a metric table and an arch-selection experiment is *de facto* part of the consolidation work.
4. **The shipped example data is not loadable by the current data pipeline** (`example/testdata/GR_bullshoals.csv` is a PhenoCam GCC export; `example/gcc_rcc_mins_site_veg.csv` has GCC columns, not `lai_min`/`lai_max`). The "test on bullshoals first" rule therefore cannot currently be run as-is.
5. **No GEOV2 / CGLS-LAI or PhenoCam v2.0 data preparation exists in the repo.** `train-flat` is the GEOV2-shaped template (36 dekads/year via `Every10DaysWrapper`) but no GEOV2 data prep is shipped.
6. The pre-trained artifacts in `example/lstm_models/` are **legacy state_dicts** (`lstm.weight_ih_l0`, `fc_1.weight`, `fc.weight`) that **match no architecture currently in the repo** and are referenced by no loader.

---

## 1. Step 1a — `models/`: read `rnn.py` and `transformerbis.py` (the "most advanced" model)

### 1.1 Inventory of shipped architectures
`phenonn/models/__init__.py` exports: `RNN_LSTM`, `RNN_GRU` (rnn.py), `FCN` (fcn.py), `EncoderTorch` (transformer.py), `CombinedModel`, `BiTransformer` (transformerbis.py), `LinearBaseline`, `PerDayLinearBaseline` (linear_baseline.py). Shape conventions: `(B, C_in, L)` → `(B, C_out, L)` for RNN/FCN/transformer; transformerbis models operate on `(B, L, C)` and are wrapped by `permuteWrapper` when used.

### 1.2 `phenonn/models/rnn.py` — findings
- **Implemented as *unidirectional*, documented as *bidirectional* (HIGH).** `rnn.py:192` instantiates `birectional=False` (sic — the comment reads `# BBBBBidirectional`), while the module docstring, class docstrings, and attribute comments repeatedly describe the models as "bidirectional" and even claim the final Conv1d input is `hidden_size` "because bidirectional doubles hidden size" (`rnn.py:196-198`). Runtime behaviour is a standard unidirectional LSTM/GRU. This is a documentation↔implementation mismatch, not a crash.
- `forward()` permutes `(B,C,L)`→`(B,L,C)`, initializes zero hidden states (`init_hidden`, `rnn.py:204-234`), runs `self.rnn`, then a `Conv1d(hidden_size → output_channel, kernel=1)`.
- Gradients, shape, serialization, and device tests all pass (`tests/test_rnn.py`, 19 test methods).

### 1.3 `phenonn/models/transformerbis.py` — findings
- `CombinedModel` (l.180) and `BiTransformer` (l.361) are two-stage designs: `lin1 → torch.Transformer → lin2` produces a 1-channel "stress" signal which is concatenated with **trailing PFT channels**, then fed through stacked causal `TransformerEncoderLayer`s (lower/upper-triangular masks) to `lin4` → `(B, L, output_dim)`. `return_stress=True` exposes the intermediate scalar signal.
- `positional_encoder` is constructed in both models but **never used in `forward()`** (dead code; `transformerbis.py:268-270`, 465-467) — position information is only handled implicitly by the encoder layers.
- **Signature drift vs. the wrapping code (root cause of the `bitransformer`/`transformerbis` crashes)** — see 1.5: public signatures use `feed_forward_trans/feed_forward_encoder/d_model/...` and `n_pft`, while `model_loader.py` passes `hidden_dim/hidden_dim_trans/n_pft` and different `d_model` defaults.
- `lin3 = Linear(1 + n_pft, d_model)` (`BiTransformer`, l.450) — the exact trailing-PFT count must match at train time. In the per-site pipeline the PFT block is a one-hot of `len(pft_list)` channels appended *last* (`dataset.py:505-510`), which is consistent only if `n_pft == len(pft_list)`.

### 1.4 Other models (context)
- `transformer.py` — `EncoderTorch`: learnable `nn.Embedding` positional embeddings (`transformer.py:102`), `TransformerEncoder` stack, final Conv1d. **No `causal` parameter in `__init__`** (see crash below).
- `fcn.py` — `FCN`: flattens `(B, C×L)`, stack of `FCBlock`s (Linear+BatchNorm+ReLU), reshapes to `(B, C_out, L)`; optional `dim_expand`.
- `linear_baseline.py` — `LinearBaseline` (full-window linear, `C×L` weights) and `PerDayLinearBaseline` (last-day features only); both output `(B,1)` and are un-wrapped.
- `utils/wrappers.py` — `SingleDayWrapper` (last timestep `(B,C,1)→(B,C)`), `permuteWrapper` (`(B,L,C)↔(B,C,L)`), `LastNDaysWrapper`, `Every10DaysWrapper` (maps to the **36 dekadal LAI positions** days 5/15/25/month, `wrappers.py:90-103`, 238-283). `_OBS_POSITIONS` is precomputed for a non-leap year.

### 1.5 Runtime verification of `model_loader.py` (HIGH — three broken `--type`s)
`phenonn/utils/model_loader.py` builds + wraps models for `train.py` and `predict.py`. Verified by instantiating every `--type`:

| `--type` | Result | Cause |
|---|---|---|
| `lstm`, `gru` | OK | |
| `fcn` / `fullyconnected` | OK | |
| `linear` / `linear_perday` | OK | |
| `1year_lstm` | OK | `LastNDaysWrapper(365)` |
| `1year_bitransformer` | OK | uses the *correct* `BiTransformer` kwargs (`model_loader.py:222-234`) |
| `transformer` | **TypeError** | `EncoderTorch(..., causal=False)` — no `causal` arg (`model_loader.py:193`) |
| `transformerbis` | **TypeError** | `CombinedModel(..., n_pft=10)` — no `n_pft` arg (`model_loader.py:196-207`) |
| `bitransformer` | **TypeError** | `BiTransformer(..., hidden_dim=..., hidden_dim_trans=...)` — signature is `feed_forward_trans/feed_forward_encoder/d_model/...` (`model_loader.py:209-220`) |

Consequences:
- In `train.py` the CLI choices (`train.py:194-210`) include `transformer` and `bitransformer` — **both crash** the moment training starts; `transformerbis` is not even offered in the CLI. `1year_bitransformer` (the BiTransformer family) is the only transformer variant reachable in this pipeline.
- **Contrast:** `train_flat.py:build_model` constructs `EncoderTorch` (no `causal` kwarg) and `BiTransformer` with the correct signature itself, so `--type transformer` and `--type bitransformer` **work in the flat and big pipelines**. The breakage is confined to `model_loader` (per-site `train`/`predict`).
- The unit tests never passed through `model_loader`, which is why CI stayed green (`tests/test_rnn.py`, `test_transformer.py`, `test_transformerbis.py` instantiate the model classes directly with consistent kwargs).

### 1.6 Extra broken CLI path in `train-flat`
`train_flat.py:254` lists `aelstm` in `--type` choices, but `build_model` has **no `aelstm` branch** (`train_flat.py:394-441`) → `ValueError("Unsupported model type: aelstm")`. The `--n_attn_blocks` / `--dropout_att` args are likewise unused.

### 1.7 Pre-trained artifacts are orphaned/legacy
`example/lstm_models/m0_{DB,EN,GR}_8f_{0..24}` (90 files) are raw `torch` `state_dict`s with keys `lstm.weight_ih_l0 / lstm.bias_ih_l0 / fc_1.weight / fc.bias / fc.weight`. No module in the repo builds a model with those parameter names (current `RNN_LSTM` uses `rnn.*` + `final.*`), and no code references these files (`grep 8f|lstm_models|m0_` over `*.py` → no hits). They are leftovers from a previous GCC-era workflow and **cannot be loaded or evaluated by the current code** without writing a legacy loader.

---

## 2. Step 1b — `training/`: loss, hyperparameters, cross-validation

### 2.1 The three training pipelines
| | `train.py` (per-site CSVs) | `train_flat.py` | `train_big.py` |
|---|---|---|---|
| Data | `{PFT}_{site}.csv` per site | `features.csv` + `targets.csv` flat | per-year folders `features_{year}.csv` / `target_{year}.csv` over a row/col grid |
| Task | predict `LAI(t)` from 365-d window; or full-year (`1year_*`, 730-in → 365-out) | per `(site_id, year)`: `(31, 720)` → `(1, 36)` dekads | same as flat, streaming per-epoch site/year sampling |
| Default loss | `nmae` (`train.py:229`) | `mse` (`train_flat.py:319`) | hard-wired **NaN-safe MSE** (`train_big.py:189-206`) ignoring missing dekads |
| Optimizer | Adam, lr 2e-3, wd 1e-5 (`train.py:660-662`) | Adam, lr 1e-3, wd 1e-5 (`train_flat.py:614-616`) | Adam (inherits flat defaults) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=5) | same | same |
| Early stop | patience=10 | patience 10 | yes |
| Grad clip | `--max_grad_norm 1.0` | 1.0 | 1.0 |
| Seed | 42 | 42 | 42 |
| Split | `site` (leave-site-out, 20%) or `year` (`train.py:167-191`) | `site` or `year` | random disjoint site pools `n_val_sites` + `val_fraction_of_grid` |
| Norm stats | train-only, saved `norm_stats.json` | train-only | **no normalization (raw units)** (`train_big.py:62-72` docstring) |

### 2.2 Loss functions — implemented vs. advertised
`utils/evaluater.py:get_loss_function` implements: `mse`, `rmse`, `mae`, `nmae` (`NMAELoss`), `nmse` (`NMSELoss`), `smoothl1`, `huber`, `gradient` (`GradientAwareLoss` = MSE base + λ·MSE on temporal diffs). **Verified runtime failure:** `wmse` and `logcosh` (listed in `train.py` `--loss_type` choices, `train.py:229-241`, and in the module docstring) raise `ValueError: Unsupported loss type`. `GradientAwareLoss` is the custom piece; it requires `n_target_days=2` (`train.py:520`).

### 2.3 Cross-validation strategy
- `site` split = random 20% holdout with `np.random.RandomState(seed)` (`dataset.py:736-764`, `dataset_flat.py:118-134`); **norm stats always from train sites only** (`train.py:497`).
- `year` split = same sites, disjoint target years (e.g. train 1993–2010 / val 2011–2019 in `run`).
- `train_big` builds a fixed disjoint validation site pool once at startup.
- Metrics computed during validation: **loss, RMSE, pooled R², and median per-site R²** (`train.py:392-431`). The flat/big `validate` compute loss, RMSE, R². **No bias (MBE) is reported during training**, although `mbe_all()` exists in `evaluater.py:365-381` and `predict.py` reports per-site/-year RMSE + R² only.
- Reproducibility: seeds fixed (default 42), `config.json` + `norm_stats.json` saved per run, best checkpoint stores `args`, `pft_list`, train/val file lists, `lai_norms`.

### 2.4 Observations
- **Inconsistent defaults between pipelines** (loss `nmae` vs `mse`, lr 2e-3 vs 1e-3, seq 365 vs 720) — expected (different tasks) but worth documenting explicitly for the ORCHIDEE target.
- No Optuna/hyperparameter-tuning module exists (`phenonn/training/` has only `train(_flat/_big).py`), although README.rst and readthedocs advertise "Hyperparameter tuning: integration with Optuna". The `hyperparameter_tuning.rst` doc page exists but no code implements it.

---

## 3. Step 1c — `data/`: GCC/RCC/GEOV2 → input tensors

### 3.1 Per-site pipeline (`dataset.py`)
- Feature groups (module constants): `DYNAMIC_FEATURES = [tmin, tmax, daylength, vpd, prcp, srad, swe]` **+ 7 derived** (`gdd_0/5/10`, `cdd`, `botta_threshold`, `botta_forcing`, `ncd`) via `add_GDD_features=True` (`dataset.py:122-131, 147-148`) → **14 dynamic channels**; `CYCLIC_FEATURES` is currently **empty** (`add_cyclic_features=False`, `dataset.py:156-158`) — the `doy_sin/doy_cos` columns are still added in `load_site` (`dataset.py:215-216`) but not used; `STATIC_FEATURES = [mat,map]` (`dataset.py:134`); PFT one-hot of `len(pft_list)` (last channels, `dataset.py:505-510`). Total for `feature_mode=all`: `14 + 0 + 2 + n_pfts`.
- `load_site` (`dataset.py:195-223`): sorts by `(year,doy)`, builds cyclic columns, calls `add_derived_features`, `ffill().bfill()` on dynamic columns.
- Normalization (`compute_norm_stats`, `dataset.py:250-327`): z-score per feature over **train sites only**; `log1p` applied before stats to `swe/vpd/prcp/gdd_0/5/10/cdd/elev` (`dataset.py:160-169, 282-285`); static features averaged one-value-per-site; `LAI` target z-scored globally **or** per-site min-max from `--gcc_norms_csv` (`load_lai_norms`, `dataset.py:226-244`) — note this CSV is expected to have `lai_min`/`lai_max` columns.
- Samples: sliding windows ending on target day, default `stride=7` (`train.py:271-276`); multi-day targets (`n_target_days=2`) for gradient loss; full-year mode samples one 730-day window per year; optional residual learning (`obs − prev_pred`) from a `predictions.csv` (`dataset.py:412-433`).
- **Target column is `LAI`**, feature columns `year, doy, tmin, tmax, ..., mat, map` — this is the required per-site CSV contract for `train.py`.

### 3.2 Flat pipeline (`dataset_flat.py`) — the GEOV2-shaped template
- `features.csv` (per `(site_id, date)`): `pft1_frac..pft15_frac` + 7 meteo + 7 derived = **31 channels** (`dataset_flat.py:79-101`); `targets.csv`: 36 LAI rows/year (days 5/15/25, `dataset_flat.py:105-106`).
- One sample per `(site_id, year)`: window = `seq_length=720` days ending last day of target year → tensor `(31, 720)`; target `(1, 36)`. `Every10DaysWrapper` selects the 36 positions from the model's last 365 outputs.
- PFT **fractions** are z-scored too, and the trailing 15 channels are consumed by `BiTransformer(n_pft=15)` (`train_flat.py:423-438`).

### 3.3 Feature engineering (`feature_engineering.py`)
- GDD at 0/5/10 °C, CDD below 5 °C, NCD (count), and Botta et al. (2000) onset forcing:
  `G_thres = 964·exp(−0.0058·NCD) − 12.8`, `botta_forcing = GDD5 / max(G_thres, 1)` (`feature_engineering.py:75-141`). All resets annually on Jan 1; deterministic.

### 3.4 Example data — **not consumable by the current loaders** (HIGH for the "test on bullshoals first" rule)
- `example/testdata/GR_bullshoals.csv` header: `date,tmin,tmax,vpd,sm,srad,dl,swe,prcp,MAP_worldclim,MAT_worldclim,GLDAS_soilfraction_*,GLDAS_soiltex,gcc,rcc,gcc_lowess,...` — a **PhenoCam GCC export**. It has **no `year`, no `doy`, no `daylength`, no `mat/map`, no `LAI`**, so `load_site()` fails immediately (`KeyError` on `df["year"]` in `add_derived_features`). It is also not a `dataset_flat` file (`site_id`/PFT-fraction columns absent).
- `example/gcc_rcc_mins_site_veg.csv` header: `sitename,veg,gcc,rcc,gcc_lowess,rcc_lowess` → **no `lai_min`/`lai_max`** → `load_lai_norms()` raises `KeyError`.
- `example/gcc_pred_test_GR_mfull.csv`: a plain GCC prediction matrix (unnamed index × day) — historical output, not training input.
- Conclusion: **no file currently in `example/` can be fed to `train`, `train-flat`, or `train-big`**, so the brief’s "prepare/validate every new GEOV2/PhenoCam feature on the `bullshoals` example first" cannot be executed without first re-exporting bullshoals in the consumed format (or adding a loader for the PhenoCam export format).

### 3.5 GEOV2 / PhenoCam-v2.0 / CGLS-LAI status
- No GEOV2/CGLS-LAI or PhenoCam v2.0 (ORNL DAAC) ingestion code is present anywhere in the repo. `train-flat` is the template that matches GEOV2's 10-day cadence (36 obs/yr = CGLS LAI), and `train-big` reads gridded per-year CSVs over `--row_min/row_max/col_min/col_max` windows (see `run` for a live config pointing at `/net/nfs/ssd1/sbarbu/PhenoNN/...` on HPC), but the actual GEOV2 data-prep step is **not shipped**.
- This matches the brief’s text (data lives on lab/cluster, Q2 in "questions à poser").

---

## 4. Step 1d — which architecture is the "best", and on what basis?

**Verdict: undeterminable from the current repo — no metrics exist.** Findings:

1. **No benchmark/results artifact in the repo:** no `runs/` outputs, no metric tables, no loss-history PNGs, no report/note. The only saved models (`example/lstm_models/`) are legacy, unloadable (see §1.7), and carry no accompanying scores.
2. **No stated comparison in docs:** readthedocs `index`/`overview` present PhenoNN as "LSTM networks to predict GCC" (i.e., the historical LSTM is the *implies best*), but **no RMSE/R²/bias numbers** are reported for any architecture (checked `doc/source/*.rst`, README.rst, GitHub README, readthedocs).
3. **The brief itself designates `transformerbis.py` as "version la plus aboutie"** — this is a code-complexity/size statement (the module is the biggest), **not a metric-based claim**.
4. **Web search** for PhenoNN metrics/publication returns nothing specific; the nearest hits are *PhenoFormer* (arXiv:2410.23327) and *NJU/GLASS LAI reconstruction* (Sci. Data 2023) — different projects. No PhenoNN preprint or technical note found. AI4PEX context (README) confirms the project parentage.
5. **Consequence:** the "best" choice (FCN vs linear vs LSTM/GRU vs Transformer vs BiTransformer) must be re-established empirically. This is one of the brief's own questions to the outgoing intern ("Quelle architecture est actuellement retenue… et sur quelles métriques ?").

**Feasible comparator levels today:** linear baselines (`LinearBaseline`, `PerDayLinearBaseline`) exist precisely as lower bounds; `train.py` already emits pooled/per-site R² during validation and `predict.py` emits per-site/year RMSE + R²; but a *same-split, same-metric* cross-architecture benchmark script does not exist yet.

---

## 5. Cross-cutting findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| C1 | HIGH | `transformer`, `transformerbis`, `bitransformer` `--type`s crash in `model_loader.py` (per-site `train`/`predict`); signature mismatch vs `transformerbis.py`/`transformer.py` | §1.5 |
| C2 | HIGH | `--loss_type wmse` / `logcosh` crash (`get_loss_function` unhandled) though advertised | §2.2 |
| C3 | HIGH | No example data loadable by current loaders (bullshoals = GCC export; norms CSV = GCC not LAI) → "test on bullshoals first" blocked | §3.4 |
| C4 | MED | `rnn.py` unidirectional implementation documented as bidirectional | §1.2 |
| C5 | MED | No GEOV2/CGLS/PhenoCam-v2 data prep in repo | §3.5 |
| C6 | MED | `aelstm` in `train-flat` choices but unimplemented; `n_attn_blocks`/`dropout_att` unused | §1.6 |
| C7 | MED | Legacy `example/lstm_models/*` state_dicts unloadable by any current architecture | §1.7 |
| C8 | LOW | `positional_encoder` dead code in both transformerbis models | §1.3 |
| C9 | LOW | `torch.Transformer` in `CombinedModel`/`BiTransformer` is a **causal-agnostic** encoder-decoder used as an encoder (`x` as both src and tgt) — actual causality is only enforced downstream in the encoder stack; worth an explicit note before relying on "causal" claims | `transformerbis.py:245, 504` |
| C10 | LOW | README/readthedocs advertise Optuna tuning, missing module | §2.4 |
| C11 | INFO | No normalized-vs-raw inconsistency flagged: `train_big` trains on raw units while flat/per-site z-score — relevant when comparing architectures | §2.1 |
| C12 | INFO | `pyproject.toml` `[tool.uv.workspace] members = ["tem"]` references a non-existent dir (present on `origin/main` too) — harmless for pip installs, error-prone under `uv` | `pyproject.toml:66-67` |

CI status note: `python -m unittest tests.test_transformer tests.test_rnn tests.test_fcn tests.test_utils tests.test_model_utils tests.test_transformerbis tests.test_evaluater tests.test_diagnostics` → **Ran 150 tests, OK** (CPU). The AI4PEX/per AGENTS "not-in-CI" tests (`test_runner`, `test_logger`) were not part of the CI gate.

---

## 6. Recommended consolidation actions (Priority 1)

1. **Fix `utils/model_loader.py`** so `transformer` (drop `causal=`), `transformerbis`/`bitransformer` (use correct signatures; align `n_pft` with the dataset's trailing-PFT channels) instantiate; add a unit test that builds **every** `--type` through `load_model` (as done manually here in §1.5).
2. **Either implement `wmse`/`logcosh` or remove them from the CLI choices**; add a test asserting every advertised loss type is constructible.
3. **Re-export `bullshoals` in a loadable format** (or add a converter PhenoCam-export → `dataset.py`/`dataset_flat.py` schema) so the brief's "test on bullshoals first" rule is usable; likewise provide a correct `lai_min/lai_max` norms CSV.
4. **Write a benchmark harness**: same seed, same `site`-split, same metric set (loss, RMSE, pooled R², median per-site R², **bias/MBE**) across `LinearBaseline`, `PerDayLinearBaseline`, `FCN`, `RNN_LSTM`, `RNN_GRU`, `EncoderTorch`, `BiTransformer`, on `train-flat`-style GEOV2 data — this produces the missing evidence for "which architecture is best".
5. **Port the per-site `predict.py` metric reporting (per-site/year RMSE + R²) into the flat/big paths** and add bias reporting if the ORCHIDEE validation (Priority 3) requires it.
6. Decide disposition of legacy `example/lstm_models/*` (remove, or add a legacy loader for reproducibility before re-training).
7. Document the cause/ceiling explicitly (GEOV2 = NN output; PhenoNN learns to emulate another ML product) in the upcoming technical note.

---

## 7. Appendix — verification commands (repro)

```bash
# Env
conda activate phenonn   # Python 3.8.20, torch 2.4.1+cu121

# CI gate
python -m unittest tests.test_transformer tests.test_rnn tests.test_fcn \
  tests.test_utils tests.test_model_utils tests.test_transformerbis \
  tests.test_evaluater tests.test_diagnostics   # → Ran 150 tests, OK

# Model instantiation through load_model (all --type)
python - <<'PY'
import argparse
from phenonn.utils.model_loader import load_model
for t in ["lstm","gru","transformer","transformerbis","bitransformer",
          "1year_lstm","1year_bitransformer","fcn","fullyconnected",
          "linear","linear_perday"]:
    a = argparse.Namespace(type=t, feature_channel=14, output_channel=1,
        seq_length=365, hidden_size=32, num_layers=2, embed_size=64,
        nhead=4, forward_expansion=4, dropout=0.0, dropout_trans=0.0,
        feed_forward_trans=4, feed_forward_encoder=4, n_target_days=1,
        full_year=False)
    try:
        load_model(a); print("[OK ]", t)
    except Exception as e:
        print("[ERR]", t, "->", type(e).__name__, e)
PY
# → ERR: transformer (causal), transformerbis (n_pft), bitransformer (hidden_dim)

# Loss factory coverage
from phenonn.utils.evaluater import get_loss_function
import argparse
a = argparse.Namespace(beta_delta=1.0)
for t in ["mse","mae","nmae","nmse","smoothl1","huber","gradient","wmse","logcosh","rmse"]:
    try:
        get_loss_function(t, a); print("[OK ]", t)
    except Exception as e:
        print("[ERR]", t, e)
PY
# → ERR: wmse, logcosh (ValueError: Unsupported loss type)
```