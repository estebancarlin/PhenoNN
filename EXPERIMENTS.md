# Experiment Log

This file records local experiments performed with the RAM pixelset workflow.
It is a reproducibility record, not a replacement for an independent evaluation
protocol.

## Baseline Model Comparison

### Data and training protocol

- Features: local ERA5 augmented daily pixelset data.
- Targets: local GEOV2 LAI dekadal pixelset data.
- Training years: 1993-2014.
- Validation years: 2015-2016.
- Selection labels: train (`split=0`) and validation (`split=1`) only.
- Input window: 720 days.
- Per epoch: 1,000 training sites and 5 training years.
- Fixed validation set per run: 500 sites.
- Optimizer: Adam, learning rate `1e-3`, MSE loss, 50 maximum epochs,
  patience 11, bf16 AMP.
- Model dimensions: `hidden_size=64`, 2 recurrent layers. Attention-LSTM
  additionally uses `d_model=64`, 2 attention blocks, 4 heads, and
  `stress_dim=8`.

### Initial single-seed comparison

The initial comparison used seed 42. Attention-LSTM was best on validation and
on the subsequently inspected 2017-2018 test set. The test results are
historical and must not be used for further model selection.

| Model | Validation RMSE | Validation R2 | Test RMSE | Test centered R2 |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 0.4716 | 0.8564 | 0.5643 | 0.7490 |
| BiTransformer V2 | 0.4432 | 0.8731 | 0.5841 | 0.7441 |
| Attention-LSTM | **0.4400** | **0.8750** | **0.5233** | **0.7934** |
| AELSTM | 0.4532 | 0.8674 | 0.5733 | 0.7339 |

## Seed Confirmation: Attention-LSTM vs LSTM

The seed-42 result was replicated with seeds 17 and 73. Each model pair with a
given seed used the same data split and training configuration. Validation-only
metrics are reported below.

| Model | Seed 17 RMSE | Seed 42 RMSE | Seed 73 RMSE | Mean RMSE +/- SD | Mean R2 +/- SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention-LSTM | 0.4313 | 0.4400 | 0.4564 | **0.4426 +/- 0.0127** | **0.8732 +/- 0.0050** |
| LSTM | 0.4697 | 0.4716 | 0.4518 | 0.4644 +/- 0.0109 | 0.8602 +/- 0.0090 |

| Seed | LSTM RMSE - Attention-LSTM RMSE |
| --- | ---: |
| 17 | +0.0384 |
| 42 | +0.0315 |
| 73 | -0.0045 |
| Mean | **+0.0218** |

Attention-LSTM wins 2 of 3 matched seeds and lowers mean validation RMSE by
4.7% relative to LSTM. The result is sufficient to select Attention-LSTM as
the improvement target. Three seeds do not provide a precise statistical
confidence interval, so the result should be described as a practical model
selection decision rather than a definitive architecture claim.

The seed-73 outcome was effectively tied and its Attention-LSTM best checkpoint
was early (epoch 12), so training stability and early-stopping sensitivity are
important considerations in follow-up work.

## Attention-LSTM Optimization

All optimization runs retain the train/validation protocol above and use seeds
17, 42, and 73. Metrics are from the best validation-RMSE checkpoint for each
run. The trainer saves this checkpoint as `checkpoints/best_rmse_model.pth`,
separately from the checkpoint selected by the training loss.

| Stage | Configuration | Mean RMSE +/- SD | Mean R2 | Decision |
| --- | --- | ---: | ---: | --- |
| Baseline | 64/64, LR `1e-3`, MSE | 0.4426 +/- 0.0127 | 0.8732 | Reference |
| 1 | 64/64, LR `1e-4`, MSE | 0.4390 +/- 0.0133 | 0.8751 | Retain LR `1e-4` |
| 2 | 128/128, LR `1e-4`, MSE | 0.4255 +/- 0.0132 | 0.8826 | Retain 128/128 |
| 3 | 128/128, LR `1e-4`, dropout `0.1` | 0.4847 +/- 0.0312 | 0.8477 | Reject dropout |
| 4 | 128/128, LR `1e-4`, correlation weight `0.05` | **0.4192 +/- 0.0132** | **0.8861** | Selected |
| 4 | 128/128, LR `1e-4`, amplitude weight `0.05` | 0.4269 +/- 0.0182 | 0.8818 | Reject |
| 4 | 128/128, LR `1e-4`, correlation/amplitude weights `0.05/0.05` | 0.4265 +/- 0.0157 | 0.8821 | Reject |

The correlation-loss model improves validation RMSE for every seed relative to
the 128/128 MSE model:

| Seed | MSE RMSE | Correlation-loss RMSE | Improvement |
| --- | ---: | ---: | ---: |
| 17 | 0.4222 | 0.4143 | 0.0079 |
| 42 | 0.4400 | 0.4341 | 0.0059 |
| 73 | 0.4142 | 0.4091 | 0.0051 |

It lowers mean RMSE by 1.5% relative to the selected MSE model and by 5.3%
relative to the original 64/64 Attention-LSTM baseline. Amplitude loss alone
and combined with correlation loss do not improve RMSE, so they are not
retained.

## Approved Next Experiments

Run all selection and tuning experiments on the 2015-2016 validation protocol
only. Do not generate new predictions for the inspected 2017-2018 test set
until model choices are frozen.

1. Test correlation-loss weight `0.1` on the selected 128/128, LR `1e-4`
   configuration for seeds 17, 42, and 73. Retain it only if it improves on
   correlation weight `0.05` by mean validation RMSE.
2. Repeat the selected final configuration for seeds 17, 42, and 73 before
   adopting it if the correlation-weight sweep changes the selected setting.
3. Evaluate documented PFT-mixing, normalization, and feature-ablation
   experiments only after the core Attention-LSTM configuration is fixed.

LSTM is retained as the frozen baseline. AELSTM and BiTransformer V2 are not
scheduled for additional tuning unless a later scientific comparison requires
them.

## Upstream 0.05-Degree Audit (2026-10-01)

### Record of the review

- Imported `upstream/main` commit `4a4c8be` in local merge commit `84943d1`.
  The import adds the 0.05-degree parent-map pipeline, daily-LAI mode, larger
  sweep tooling, climatology skill analysis, feature ablation, perturbation,
  PFT-mixing, and temporal-generalization workflows.
- Reviewed `Rapport_de_stage_4A_PPeylinpdf.pdf` and `PhenoNN_final.pdf`.
  The report is the primary source; the presentation/poster contains earlier,
  incompatible experiments and must not be pooled with report results.
- The report's headline AELSTM result (`R2=0.9553`, RMSE `0.3054`) uses a
  temporal split with the same sites in training (`1992-2009`) and validation
  (`2010-2019`). It is not comparable with this project's spatially isolated
  validation protocol (`1993-2014` versus `2015-2016`).
- The local selected Attention-LSTM result remains `RMSE=0.4192 +/- 0.0132`,
  mean `R2=0.8861`, from the spatial validation protocol. It is the empirical
  reference until models are compared under one frozen protocol.
- The report's main scientific finding is weak interannual skill: the best
  per-site-climatology anomaly result is AELSTM `R2=0.0780`, RMSE `0.2183`,
  compared with a zero-anomaly climatology RMSE of `0.2274`. Raw-LAI seasonal
  skill should therefore not be interpreted as strong anomaly prediction.
- The report's raw-LAI table is internally non-reproducible as written:
  climatology has higher global R2 (`0.9636`) but worse RMSE (`0.3570`) than
  AELSTM (`0.9553`, `0.3054`). With the stated common pooled R2 definition,
  that requires different masks or populations, or a reporting/calculation
  error. Recompute both on identical finite observations before using the
  comparison.
- The report appendix poster is a separate 40,000-cell US-plains experiment
  with a spatial 80/20 split and different years. Its AELSTM `R2=0.9071`,
  RMSE `0.2685`, and adjacent-time daily-change result are not comparable with
  either the report's global temporal split or this project's spatial split.

### What upstream changed technically

| Area | Upstream method | Local state before import | Implication |
| --- | --- | --- | --- |
| Spatial support | 0.05-degree LAI/PFT sites mapped to deduplicated 0.1-degree ERA5 parents | Selected-site 0.1-degree workflow | More target/PFT detail, but weather remains 0.1 degree and cannot create sub-grid meteorological information. |
| Inputs | 730 days, 12 weather variables including day length and 30-day precipitation SMI, CO2, 15 PFT fractions (28 channels) | 720 days, 11 dynamic weather channels, CO2, 15 PFTs (27 channels) | Existing checkpoints and statistics are schema-incompatible; regenerate data/statistics for a 28-channel run. |
| Target | 36 observed dekads; optional 365-day linear interpolation for training | 36 observed dekads | Daily interpolation adds smooth synthetic supervision, not new observations. |
| Loss/training | Huber default, 300 epochs, scheduler, early stopping, broad sweep | MSE plus selected correlation term, 50-epoch comparison budget | Upstream had a substantially larger optimization budget. |
| Split | Primarily temporal, same locations across periods | Clustered spatial holdout | Their task is easier and answers a different scientific question. |
| Structure | Optional PFT-mixing output layer | Global-LAI prediction selected | PFT curves are weakly identified because only their weighted sum is observed. |
| Evaluation | Climatology, anomaly, site-year, interannual and perturbation analyses | Raw RMSE/R2 and spatial evaluation | These diagnostics are required before scientific or ORCHIDEE claims. |

### Recommended next experiments and expected outcomes

Run each candidate on the frozen local spatial validation split. Do not use the
already inspected 2017-2018 test set for selection. Report physical-unit RMSE,
pooled and site-centred R2, per-site-year R2, interannual R2, and a
training-period per-site climatology baseline using the identical target mask.

| Priority | Experiment | Expected outcome | Decision rule |
| --- | --- | --- | --- |
| 1 | Add a training-period per-site climatology and anomaly evaluation to every selected model | Clarifies whether improvements are seasonal-shape fit or real interannual skill; raw RMSE may remain good while anomaly skill is near zero | Require an improvement over climatology on held-out spatial sites before claiming added predictive value. |
| 2 | Rebuild one matched 28-channel dataset: 730-day window plus day length; retain local spatial split and training-only normalization | Day length may improve autumn and the extra 10 days remove an arbitrary mismatch; expect a modest, uncertain raw-RMSE change rather than an upstream-sized jump | Keep only if mean validation RMSE improves across seeds and climatology-relative metrics do not regress. |
| 3 | Compare selected Attention-LSTM with AELSTM under the exact same local data, epochs, schedule, and seeds | AELSTM may not win: upstream's raw-LAI advantage was under an easier same-site split and a larger sweep | Treat architecture as a replacement only if it improves mean spatial RMSE and centred/interannual metrics. |
| 4 | Compare MSE plus correlation loss against NaN-safe Huber, with the same 128/128 Attention-LSTM and longer early-stopped budget | Huber may improve robustness to GEOV2 outliers; correlation loss already improved local RMSE, so neither result should be assumed | Select by the best-RMSE checkpoint on spatial validation, then compare climatology-relative metrics. |
| 5 | Test daily-interpolated training against dekadal-only training, evaluating only the 36 real dekads | May smooth seasonal curves and stabilize gradients, but cannot add observational information and can bias timing between dekads | Retain only if real-dekad spatial metrics improve; do not report interpolated-day validation as independent accuracy. |
| 6 | Perform grouped feature ablation and add actual/reanalysed root-zone soil water if available | Day length and temperature should matter; replacing SMI can improve water-limited regions more plausibly than further architecture tuning | Evaluate by climate/PFT region and retain a feature only with reproducible held-out benefit. |
| 7 | Run PFT mixing and virtual-pure-cell linearity tests after the global model is frozen | It may improve ORCHIDEE interface interpretability but can reduce or leave unchanged aggregate LAI accuracy; pure PFT curves are not uniquely identified | Do not call outputs PFT-specific LAI without purity, greedy-reference, and linearity diagnostics. |
| 8 | Make the Attention-LSTM strictly causal before perturbation or coupling work | May slightly lower in-sample skill because the current centred convolution reads `t+1`, but removes future-information leakage | Use causal models for all memory, perturbation, and prospective ORCHIDEE experiments. |

### Scientific expectations

The highest-confidence near-term improvement is not a large raw-RMSE jump. It
is a defensible evaluation showing whether spatially held-out models beat a
training-only climatology on interannual anomalies. More sites, 730-day input,
day length, a longer schedule, and a Huber comparison can improve raw spatial
RMSE, but no evidence supports expecting the report's `R2=0.955` under this
harder split. Daily interpolation and PFT mixing are useful methodological
experiments, not assumed accuracy improvements.

For the ORCHIDEE objective, use the least invasive coupling first: derive SOS
and EOS from a strictly causal PhenoNN trajectory and retain ORCHIDEE carbon
allocation. Directly overwriting LAI would violate allocation, carbon, and
nitrogen mass-balance constraints. Evaluate any coupling against independent
carbon, water, and energy flux observations, not only GEOV2 LAI.

## Discussion: Spatial Robustness, Anomalies, and ORCHIDEE Coupling (2026-10-05)

### Comparing spatial robustness

The upstream temporal result and the local spatial result answer different
questions. A fair comparison requires both approaches to use the same data
schema, target mask, normalization, seeds, training budget, and checkpoint
selection rule.

Evaluate both models using at least four blocked protocols:

1. Same sites and future years: temporal extrapolation at known sites.
2. Unseen sites and matched years: geographic transfer.
3. Unseen sites and future years: spatiotemporal generalization.
4. Held-out geographic or climate/PFT regions: out-of-distribution transfer.

Random pixel splits should be avoided where neighboring pixels are correlated;
spatial clusters, tiles, or leave-region-out splits are preferable.

Report physical-unit RMSE and MAE, site-centred R2, per-site-year metrics,
bias, amplitude error, and errors around SOS, peak LAI, and EOS. Stratify the
results by climate, PFT composition, aridity, and latitude. A climatology
baseline must be included in every protocol.

### Anomaly comparison

Anomalies should be evaluated explicitly as:

    anomaly = observed LAI - climatological LAI

The climatology must be computed from training years only. For temporal splits,
this can be a site-by-dekad climatology. For unseen-site spatial splits, a
site-specific climatology is not operationally available; use a climatology
estimated from training sites using regional, climate, latitude, and PFT
information. A climatology calculated from held-out observations may be
reported as an oracle diagnostic, but not as a deployable baseline.

Report anomaly RMSE, MAE, correlation, anomaly R2, per-site-year anomaly R2,
predicted-to-observed anomaly variance ratio, and skill relative to a
zero-anomaly baseline. Absolute-LAI models can be converted to anomalies by
subtracting the same training climatology. Anomaly models can be converted back
to LAI by adding it. Both must be evaluated on the same original dekadal
observations; interpolated daily targets must not be treated as independent
observations.

The upstream anomaly result (`R2=0.0780`, RMSE `0.2183`) compared with a
zero-anomaly climatology (`RMSE=0.2274`) indicates modest interannual skill.
Therefore, the upstream raw-LAI advantage may primarily reflect seasonal
climatology reconstruction rather than strong prediction of year-to-year
departures.

### Implications for ORCHIDEE

The intended ORCHIDEE use changes the model-selection priorities. For offline
reconstruction at known sites, same-site temporal performance is useful. For an
ESM phenology component, the model must also transfer across locations, remain
causal, respond plausibly to weather and CO2 perturbations, and operate outside
the historical climate envelope.

The current Attention-LSTM's centred convolution can read `t+1`. This may help
offline validation but is future-information leakage for an online coupling.
The coupled model should therefore be strictly causal, or its complete-season
forecasting use case must be explicitly justified.

Raw-LAI RMSE alone is insufficient for selecting an ORCHIDEE replacement.
Perturbation tests for drought, heat, cold, precipitation, and CO2 should be
combined with checks of PFT consistency, extrapolation behavior, and carbon,
water, and energy fluxes.

Directly overwriting ORCHIDEE LAI risks inconsistency with carbon allocation,
leaf nitrogen, photosynthesis, respiration, and canopy water fluxes. A safer
coupling path is to use PhenoNN to provide SOS, EOS, or phenology transition
rates while retaining ORCHIDEE's carbon allocation and LAI state evolution.
