Overview
========

PhenoNN is a deep learning framework for predicting **Leaf Area Index
(LAI)** from climate data. It reads daily ERA5-Land meteorology, CO2
concentration and 15 Plant Functional Type fractions over a two-year
window, and predicts the 36 dekadal LAI observations of a target year.

The scientific context is the AI4PEX project (Research Focus LAND),
which seeks a better representation of terrestrial ecosystems in climate
models. PhenoNN contributes to two of its directions: hybrid modelling
for water and heat stress responses, and deep learning for phenology.

Task formulation
----------------

Each sample corresponds to one ``(site, year)`` pair:

- **Input** — ``(n_features, seq_length)``, with ``seq_length = 720`` by
  default (about two years). Channels are the meteorology variables from
  :mod:`phenonn.utils.config`, plus CO2 concentration, plus the 15 PFT
  fractions, all z-scored using the training statistics.
- **Output** — ``(1, 36)``: the LAI values at the 36 dekadal observation
  days of the target year (days 5, 15, 25 of each month).

The 36 dekadal targets mirror the temporal resolution of the satellite
LAI product used for training.

Architecture
------------

PhenoNN exposes four sequence-to-sequence architectures. All take
``(B, C_in, L)`` tensors and return ``(B, C_out, L)``, and all are
compatible with the wrapper stack described below.

- :class:`~phenonn.models.rnn.RNN_LSTM` — LSTM encoder with a 1×1
  convolutional output head. The smallest and fastest baseline.
- :class:`~phenonn.models.aelstm.AELSTM` — LSTM encoder followed by
  transformer-style self-attention blocks, then a 1×1 convolutional
  head. The attention layer re-weights the LSTM's hidden states so
  informative timesteps (green-up, drought events) contribute more.
- :class:`~phenonn.models.bitransformer.BiTransformerV2` — two-stage
  transformer: stage 1 turns weather features into a small "stress"
  vector per day; stage 2 combines that stress with the PFT fractions
  to produce LAI. Both stages use causal masking.
- :class:`~phenonn.models.transformer.CausalSparseTransformer` —
  Informer-style encoder–decoder that outputs directly at the 36
  dekadal positions, without the wrapper stack.

Wrapper stack
-------------

The wrapper stack sits between the model and the training loop. It
handles the shape and semantics of the model's output for the specific
task:

- :class:`~phenonn.utils.wrappers.Every10DaysWrapper` — keeps the last
  365 days of the model output and selects the 36 dekadal positions.
  Used by every model except ``CausalSparseTransformer``.
- :class:`~phenonn.utils.wrappers.DailyWrapper` — keeps the full last
  365 days. Used when training on a daily-interpolated LAI target.
- :class:`~phenonn.utils.wrappers.PFTMixingWrapper` — used with
  ``--pft_mixing``. The base model outputs 15 pure-LAI curves (one per
  PFT); this wrapper mixes them into a single cell-level LAI using the
  local PFT fractions.
- :class:`~phenonn.utils.wrappers.permuteWrapper` — adapter for models
  that expect ``(B, L, C)`` internally (the transformers). Applied
  automatically by the model loader.

Plant Functional Types
----------------------

PFTs follow the ORCHIDEE 15-class convention (see
:data:`phenonn.utils.config.PFT_NAMES`):

- bare soil
- tropical broadleaf evergreen / raingreen
- temperate needleleaf evergreen
- temperate broadleaf evergreen / summergreen
- boreal needleleaf evergreen
- boreal broadleaf summergreen
- boreal needleleaf summergreen
- temperate C3 grass / C4 grass
- C3 agriculture / C4 agriculture
- tropical C3 grass
- boreal C3 grass

Two ways of using them:

- **With** ``--pft_mixing`` — the model outputs one pure-LAI curve per
  PFT, and the wrapper takes the area-weighted sum. This is the default
  for the pixel-set pipeline and matches the additive interpretation of
  mixed pixels.
- **Without** ``--pft_mixing`` — the model receives the PFT fractions as
  input channels and predicts the cell-level LAI directly. Simpler, but the
  model must learn the mixing relation from data.

Feature sets
------------

The channels of the input tensor are defined in
:mod:`phenonn.utils.config`. They fall into four groups:

- **Meteorology** (from ``ERA5_daily_pixelset_{Y}.nc``):
  ``Tmin``, ``Tmax``, ``Tmean``, ``ssrd_sum``, ``strd_sum``, ``tp_sum``,
  ``VPD_max``, ``VPD_mean``, ``Rn_tot``, ``PET``, plus the engineered
  ``daylength`` and ``SMI``.
- **Optional thermal phenology** (ORCHIDEE v4.2 proxies, computed
  offline): GDD, NCD, NGD, Botta threshold and forcing, rising and
  falling temperature trends. Off by default; turn on with
  ``add_pheno_features`` in :mod:`phenonn.utils.config`.
- **CO2** — one value per year, broadcast over the whole window.
- **PFT fractions** — 15 channels.

The total is available as ``FEATURE_CHANNELS``; on the standard
configuration it is 28 (10 + 2 meteorology, 1 CO2, 15 PFT).

Input data format
-----------------

PhenoNN reads pixelset NetCDF files, one per year. LAI/PFT targets use
``site_id`` strings of the form ``pix_LLLL_OOOOO`` on the native 0.05°
grid:

- ``ERA5_daily_pixelset_{YYYY}.nc`` — dims ``(time, site)``, 12
  dynamic variables (10 base meteorology + ``daylength`` + ``SMI``). On
  the report route the site IDs are deduplicated 0.1° ``E...`` cells and
  a parent map joins them to the 0.05° targets.
- ``LAI_dekadal_{YYYY}.nc`` — dims ``(dekad, site)``, variable ``LAI``
  with 36 dekads per year.
- ``PFTmap_{YYYY}.nc`` — dims ``(pft, site)``, variable ``pft_frac``
  with 15 PFTs.

The dataset reconstructs the two-year window by concatenating the
``Y-1`` and ``Y`` files and keeping the last ``seq_length`` days.
Samples whose ``Y-1`` file is missing, or whose window contains a NaN,
are dropped at construction time.

See :doc:`data_preparation` for the exact layout and how to build these
files.

Training pipeline
-----------------

The trainer is
:mod:`phenonn.training.train_full_ram`. It loads the raw features,
targets and PFT vectors for the working set of sites and years into
memory once, then sub-selects sample indices every epoch. The working
set is the union of the sites the sampler will ever draw given the
number of epochs — so RAM stays bounded regardless of the total pool
size.

Each epoch:

1. Sample a subset of sites and years with a fixed seed.
2. Take the corresponding samples from the resident dataset.
3. Train one epoch with the chosen loss (MSE, MAE, Huber, smooth-L1,
   NMSE, NMAE or the gradient-aware variant).
4. Validate on a disjoint site pool or on disjoint years, depending on
   the overlap mode.
5. Step the learning-rate scheduler and, if the validation loss
   improved, write ``run_best_model.pth.tar`` (or ``best_model.pth.tar`` with ``--prefix ""``).

Early stopping, checkpoint snapshots and diagnostic plots are built in.

Prediction pipeline
-------------------

:mod:`phenonn.prediction.predict` loads a checkpoint, rebuilds the model
with the same wrapper stack, builds the dataset over the requested sites
and years, runs inference, and writes:

- a per-``(site, year, dekad)`` CSV with predictions and observations
  (both in normalised and physical units);
- a metrics summary covering pooled, per-site, site-year and interannual
  R²;
- diagnostic plots (predicted-vs-observed hexbin, seasonal curves for a
  representative subset of sites and a grid of all of them).

The predictor reads the data directories from the checkpoint's saved
arguments, so a checkpoint is self-describing.


Next steps
----------

- :doc:`installation` — install PhenoNN.
- :doc:`quickstart` — a first end-to-end run in a few minutes.
- :doc:`data_preparation` — the pixelset NetCDF layout.
- :doc:`training` — every training flag and model type.
- :doc:`prediction` — inference flags and output format.
- :doc:`api/modules` — the full API reference.
