# Memory analysis

Status: design only. This directory contains the experimental protocol, not a
runnable analysis pipeline.

The study asks whether PhenoNN learns lagged weather effects and inter-annual
legacy effects that process-based models miss. It is organized around five
experiments:

- M1: memory kernel from local weather sensitivities.
- M2: instantaneous versus memory skill decomposition.
- M3: drought legacy on the following year's start of season.
- M4: delayed effect of a warm spring on summer LAI.
- M5: optional global map of memory length.

The full hypotheses, metrics, success criteria, and dependencies are in
[PROTOCOL.md](PROTOCOL.md).

## Current contents

```text
README.md      this status page
PROTOCOL.md    experimental design
```

There are no runnable scripts yet, so there are no commands to run. The
protocol describes the planned toolkit:

- phenology metrics from the 36 dekadal LAI values,
- a counterfactual inference runner based on forward pre-hooks,
- a drought operator,
- PFT, aridity, and latitude stratifiers,
- XGBoost and growing-degree-day contrast baselines.

Do not treat this directory as a ready analysis study. When the toolkit is
implemented, add one README command per runnable script and add the modules to
the analysis import tests.

## Relationship to the other studies

The planned counterfactual runner follows the same non-invasive approach as
[perturbation](../perturbation/README.md) and
[feature ablation](../feature_ablation/README.md): it edits model inputs or
installs hooks without changing `phenonn/`. The drought legacy experiment also
depends on the perturbation machinery and the shared checkpoint helpers in
`analysis/_checkpoint_utils.py`.

## Scope limits

- The model sees weather only, so measured legacy effects are exogenous. The
  protocol treats them as lower bounds.
- The target is LAI, not gross primary production or carbon flux.
- Causal interpretation of within-year sensitivity requires a causal model
  variant.
