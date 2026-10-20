#!/usr/bin/env python3
"""
run_ablate_combo_models.py — fixed combo ablation across model types/sizes.

Trains the models in ``_common.COMBO_CONFIGS`` with the combined ablation
(``tp_sum + ssrd_sum + strd_sum + VPD_max + Tmean`` removed), on ONE shared
dataset load. By default each config is trained twice — no-ablation baseline
first, then the ablated model — so ΔR² can be computed per config.

    python -m analysis.feature_ablation.run_ablate_combo_models \\
        <train_full_ram flags: --loss_type mse --patience 7 --pft_mixing …>

Use ``--skip-baseline`` when the ``*_baseline`` runs already exist.
"""

import argparse
import copy
import os

import numpy as np
import torch

from analysis._checkpoint_utils import has_completed_run
from analysis.feature_ablation._common import (
    COMBO_CONFIGS,
    COMBO_FEATURES,
    combo_baseline_exp_name,
    combo_exp_name,
)
from analysis.feature_ablation.run_ablate_inproc import _install, _restore
from phenonn.training import train_full_ram as T
from phenonn.utils.config import ALL_FEATURES
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_done(args_i):
    run_dir = os.path.join(args_i.output_dir, args_i.experiment)
    return has_completed_run(run_dir)


def _run_one(base_args, shared, logger, typ, hid, stress, channels, ablated):
    args_i = copy.deepcopy(base_args)
    args_i.type = typ
    args_i.hidden_size = hid
    if stress is not None:
        args_i.stress_dim = stress
    if ablated:
        args_i.experiment = combo_exp_name(typ, hid, stress)
        args_i.ablate_features = ",".join(COMBO_FEATURES)
        label = "combo"
    else:
        args_i.experiment = combo_baseline_exp_name(typ, hid, stress)
        args_i.ablate_features = ""
        label = "baseline"

    if _is_done(args_i):
        logger.info(f"[skip] {args_i.experiment} — already completed.")
        return

    paths_i, cfg_logger = T.setup_directories_and_logging(args_i)
    cfg_logger.show_header(
        f"combo {label} {typ} h{hid}" + (f" s{stress}" if stress is not None else "")
    )
    _seed_all(args_i.seed)

    orig = _install(T, channels) if ablated else None
    try:
        T.train_one_config(args_i, shared, cfg_logger, paths_i)
    finally:
        if orig is not None:
            _restore(T, orig)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not train the no-ablation baseline for each config.",
    )
    known, rest = ap.parse_known_args()

    base_args = T.parse_args(rest)  # standard train_full_ram CLI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, "ablate_combo_models")
    FileUtils.makedir(sweep_dir)
    logger = Logger(
        console_output=True,
        file_output=True,
        log_file=os.path.join(sweep_dir, "load.log"),
    )
    logger.show_header(
        f"PhenoNN combo-ablation model/size — {len(COMBO_CONFIGS)} configs"
        + ("" if known.skip_baseline else " + baselines")
    )

    channels = [ALL_FEATURES.index(f) for f in COMBO_FEATURES]
    logger.info(f"Combo removed   : {COMBO_FEATURES}  channels={channels}")

    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, logger, device)

    for typ, hid, stress in COMBO_CONFIGS:
        if not known.skip_baseline:
            _run_one(
                base_args,
                shared,
                logger,
                typ,
                hid,
                stress,
                channels,
                ablated=False,
            )
        _run_one(base_args, shared, logger, typ, hid, stress, channels, ablated=True)


if __name__ == "__main__":
    main()
