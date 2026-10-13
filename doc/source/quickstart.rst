Quickstart
==========

This guide walks you through a first end-to-end run of PhenoNN. By the
end you will have trained a model on a small subset of the pixelset data
and produced a set of LAI predictions.

Prerequisites
-------------

- PhenoNN installed (see :doc:`installation`).
- A pixelset data directory containing the per-year NetCDF files:

  - ``ERA5_daily_pixelset_{YYYY}.nc`` — daily meteorology
  - ``LAI_dekadal_{YYYY}.nc`` — 36 dekadal LAI observations
  - ``PFTmap_{YYYY}.nc`` — 15 PFT fractions
  - ``selected_pixels.nc`` — the site pool
  - ``norm_stats.json`` — per-feature normalisation statistics (optional)
  - a CO2 lookup file (optional)

A ready-made dataset that matches this layout is available on the IPSL
cluster. If you have your own, make sure the file names match
``phenonn.utils.config``.

Run the smoke test
------------------

The fastest way to confirm the installation and the data path is to run
the smoke training script. It trains a tiny model on a handful of sites
for two epochs, on CPU, in a couple of minutes::

    bash scripts/smoke_train.sh

If your data is not in the default location, point the script at it::

    DATA=/path/to/pixelset1 bash scripts/smoke_train.sh

On success you will see::

    [LAIDataset] N samples kept | dropped: ...
    Model           : lstm + PFTMixingWrapper
    Epoch 1/2  sites=8  years=[2000, 2001]
      train=...  val=...  RMSE=...  R²=...  lr=...  (...s)
    ...
    Training complete. Best epoch=..., best_val_loss=...
    Checkpoint dir : runs/smoke/smoke_lstm/checkpoints
    best_model.pth
    last_model.pth

Train a model
-------------

The training entry point is the RAM-resident trainer. It loads the
requested sites and years into memory once, then sub-selects sample
indices each epoch::

    python -m phenonn.training.train_full_ram \
        --features_dir /path/to/era5_pixelset \
        --target_dir   /path/to/LAI_pixelset \
        --pft_dir      /path/to/PFT_pixelset \
        --selected_pixels /path/to/selected_pixels.nc \
        --stats_path      /path/to/norm_stats.json \
        --co2_path        /path/to/CO2.txt \
        --train_years 2000-2001 \
        --val_years   2002 \
        --type lstm --hidden_size 128 --num_layers 2 \
        --pft_mixing --pft_meteo_only \
        --num_epochs 50 --batch_size 32 \
        --output_dir runs --experiment my_first_run

Key flags
~~~~~~~~~

- ``--type`` — one of ``lstm``, ``aelstm``, ``bitransformer_v2``,
  ``transformer``.
- ``--pft_mixing`` — output 15 pure-LAI curves, mixed to a single LAI
  using the local PFT fractions.
- ``--pft_meteo_only`` — with ``--pft_mixing``, the model sees only the
  meteorology channels; the PFT fractions enter as mixing weights only.
- ``--subset`` — fraction of the site pool to keep (for a fast run, use
  a small value such as ``0.001``).
- ``--n_sites_per_epoch`` / ``--n_years_per_epoch`` — how many sites
  and years to sample each epoch.

Outputs land in ``runs/<experiment>/``:

- ``checkpoints/best_model.pth`` — best validation checkpoint
- ``checkpoints/last_model.pth`` — last epoch
- ``logs/train.log`` — per-epoch metrics
- ``loss_history.png`` / ``metric_history.png`` — training curves
- ``config.json`` — the exact arguments used

Predict with a checkpoint
-------------------------

The predictor loads a checkpoint, rebuilds the model, runs inference on
the requested sites and years, and writes a CSV of predictions::

    python -m phenonn.prediction.predict \
        --checkpoint runs/my_first_run/checkpoints/best_model.pth \
        --predict_sites val \
        --output_csv runs/my_first_run/predictions.csv

The data directories are read back from the checkpoint's saved arguments,
so you do not need to repeat them. Override with ``--features_dir``,
``--target_dir``, ``--pft_dir`` if the checkpoint stored stale paths.

The output CSV has one row per ``(site, year, dekad)``::

    site_id, year, month, day, doy, lai_pred, lai_obs,
    lai_pred_norm, lai_obs_norm, error

Alongside it, the predictor writes:

- ``*_metrics.txt`` — pooled metrics and R² decomposition
- ``*_pred_vs_obs.png`` — predicted-vs-observed hexbin
- ``*_lai_curves.png`` — 3 representative sites (low / mid / high R²)
- ``*_lai_curves_all.png`` — grid of per-site curves

Python API
----------

The three pieces you compose in Python are the dataset, the model
loader, and the training loops::

    from types import SimpleNamespace
    import torch

    from phenonn.data import RamLAIDataset
    from phenonn.utils import build_model_pft
    from phenonn.utils import make_loss_fn

    # 1. Dataset
    ds = RamLAIDataset(
        features_dir="/path/to/era5_pixelset",
        target_dir="/path/to/LAI_pixelset",
        pft_dir="/path/to/PFT_pixelset",
        years=[2010, 2011],
        site_ids=["pix_0123_04567", "pix_0124_04568"],
        seq_length=720,
        norm_stats=None,      # or a dict with mean/std per feature
        normalize_lai=True,
    )
    features, targets = ds[0]
    # features: (n_features, 720)   targets: (1, 36)

    # 2. Model
    args = SimpleNamespace(
        type="lstm",
        seq_length=720,
        hidden_size=128,
        num_layers=2,
        pft_mixing=True,
        pft_meteo_only=True,
        normalize_lai=True,
        # the fields below are read only by the architectures that need them
        d_model=None, nhead=4, forward_expansion=4,
        dropout1=0.0, dropout2=0.0, dropout_att=0.0,
        dropout_trans=0.0, dropout_encoder=0.0,
        feed_forward_trans1=4, feed_forward_trans2=4,
        feed_forward_encoder=4,
        n_attn_blocks=2, stress_dim=8, num_layers1=2, d_layers=2,
        embed_size=64, daily_lai=False,
    )
    model = build_model_pft(args, norm_stats=None)

    # 3. Forward
    preds = model(features.unsqueeze(0))    # (1, 1, 36)

For training, use the same two functions the CLI uses::

    from phenonn.training.train_full_ram import (
        load_shared, train_one_config, parse_args,
    )

    cli_args = parse_args([
        "--features_dir", "/path/to/era5_pixelset",
        "--target_dir",   "/path/to/LAI_pixelset",
        "--pft_dir",      "/path/to/PFT_pixelset",
        "--selected_pixels", "/path/to/selected_pixels.nc",
        "--train_years", "2010-2015", "--val_years", "2016",
        "--type", "lstm", "--hidden_size", "128",
        "--pft_mixing", "--pft_meteo_only",
        "--num_epochs", "10",
        "--output_dir", "runs", "--experiment", "api_demo",
    ])
    shared = load_shared(cli_args, logger, device)   # loads the working set once
    train_one_config(cli_args, shared, logger)       # trains one config

``load_shared`` is the expensive step (disk read + RAM materialisation).
If you are sweeping hyperparameters, call it once and reuse ``shared``
across several ``train_one_config`` calls.

Next steps
----------

- :doc:`data_preparation` — the exact NetCDF layout PhenoNN expects.
- :doc:`training` — every training flag, model type, and loss option.
- :doc:`prediction` — inference flags and output format.
- :doc:`api/modules` — the full API reference.
