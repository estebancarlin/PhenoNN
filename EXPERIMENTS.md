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

## Approved Next Experiments

Run all selection and tuning experiments on the 2015-2016 validation protocol
only. Do not generate new predictions for the inspected 2017-2018 test set
until model choices are frozen.

1. Compare Attention-LSTM learning rates `1e-3` and `1e-4`.
2. Compare the current 64/64 configuration with `hidden_size=128` and
   `d_model=128` at the selected learning rate.
3. Add dropout only if the larger configuration indicates overfitting.
4. On the most stable configuration, test small correlation-loss and
   amplitude-loss weights to improve seasonal shape and reduce amplitude
   damping/low-LAI bias.
5. Repeat the selected final configuration for seeds 17, 42, and 73 before
   adopting it.
6. Evaluate documented PFT-mixing, normalization, and feature-ablation
   experiments only after the core Attention-LSTM configuration is fixed.

LSTM is retained as the frozen baseline. AELSTM and BiTransformer V2 are not
scheduled for additional tuning unless a later scientific comparison requires
them.
