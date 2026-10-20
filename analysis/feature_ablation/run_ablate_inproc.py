#!/usr/bin/env python3
"""
run_ablate_inproc.py — in-process feature/group ablation sweep.

Loads the RAM dataset ONCE and trains several configs sequentially on the same
GPU — pass a slice with ``--indices`` so a job does several runs/GPU. Two modes:

``--mode groups`` (default)
    Leave-one-GROUP-out: physically related features are removed together (the
    two radiation fluxes, the three temperatures, VPD max+mean, …), plus a
    baseline. Experiment names: ``ablate_<model>_<group>``.

``--mode features``
    Leave-one-FEATURE-out over the non-PFT channels, plus a baseline. Experiment
    names match the per-feature jobs (``abl_baseline`` / ``abl_<feature>``) so
    ``aggregate.py --kind features`` reads them directly. This is also what the
    smoke driver uses.

Ablation = mean-imputation: the z-scored input channels of the group are zeroed
before every forward (their mean is 0 → the channel carries no information), so
ΔR² vs the baseline measures that group's contribution. Installed as a forward
pre-hook by wrapping the two model factories in train_full_ram's namespace — no
change to phenonn/ and state dicts stay compatible (the hook adds no params).

Usage
-----
    python -m analysis.feature_ablation.run_ablate_inproc \\
        --mode groups --model aelstm --indices 0-2 \\
        <train_full_ram flags: --loss_type mse --patience 7 --pft_mixing …>

    python -m analysis.feature_ablation.run_ablate_inproc \\
        --mode features --model lstm --features Tmean,tp_sum \\
        <train_full_ram flags>

Every flag other than --mode / --model / --features / --indices is passed to
train_full_ram.parse_args.
"""

import argparse
import copy
import os

import numpy as np
import torch

from analysis._checkpoint_utils import has_completed_run
from analysis.feature_ablation._common import (
    BASELINE_EXP,
    GROUPS,
    NON_PFT_FEATURES,
    exp_name,
    group_exp_name,
)
from analysis.feature_ablation._hooks import ZeroChannels
from phenonn.training import train_full_ram as T
from phenonn.utils.config import ALL_FEATURES, PFT_START
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils


def _channels(feats):
    chs = []
    for f in feats:
        ch = ALL_FEATURES.index(f)
        if ch >= PFT_START:
            raise SystemExit(
                f"{f!r} is a PFT channel; only non-PFT features " f"are ablatable."
            )
        chs.append(ch)
    return chs


def _install(module, channels):
    """Wrap build_model / build_model_pft in `module` so the returned model
    zeroes the given input channels before every forward. Returns the originals."""

    hook = ZeroChannels(channels)

    orig = (module.build_model, module.build_model_pft)

    def build_model(args):
        m = orig[0](args)
        m.register_forward_pre_hook(hook)
        return m

    def build_model_pft(args, norm_stats):
        m = orig[1](args, norm_stats)
        m.register_forward_pre_hook(hook)
        return m

    module.build_model, module.build_model_pft = build_model, build_model_pft
    return orig


def _restore(module, orig):
    module.build_model, module.build_model_pft = orig


def _parse_indices(spec, n):
    if not spec:
        return list(range(n))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        idxs = list(range(int(a), int(b) + 1))
    else:
        idxs = [int(x) for x in spec.split(",")]
    for i in idxs:
        if not (0 <= i < n):
            raise SystemExit(f"index {i} out of range 0..{n - 1}")
    return idxs


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_done(args_i):
    run_dir = os.path.join(args_i.output_dir, args_i.experiment)
    return has_completed_run(run_dir)


def _split_features(spec: str):
    return [f.strip() for f in spec.split(",") if f.strip()]


def _specs(mode: str, model: str, features: str):
    """Return ``[(label, feature_list, experiment_name), ...]`` for one mode."""
    if mode == "groups":
        return [(name, feats, group_exp_name(model, name)) for name, feats in GROUPS]

    chosen = _split_features(features) or list(NON_PFT_FEATURES)
    unknown = [f for f in chosen if f not in NON_PFT_FEATURES]
    if unknown:
        raise SystemExit(
            f"Unknown or non-ablatable feature(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(NON_PFT_FEATURES)}"
        )
    specs = [("baseline", [], BASELINE_EXP)]
    specs += [(feat, [feat], exp_name(feat)) for feat in chosen]
    return specs


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mode", default="groups", choices=["groups", "features"])
    ap.add_argument("--model", default="aelstm")
    ap.add_argument(
        "--features",
        default="",
        help="features mode only: comma-separated subset (default: all).",
    )
    ap.add_argument(
        "--indices",
        default="",
        help="Runs for THIS process: '0-2', '0,3', or empty for all. "
        "Index 0 is the baseline in features mode. Slice across GPUs.",
    )
    known, rest = ap.parse_known_args()

    if known.mode == "groups" and known.features:
        ap.error("--features is only valid with --mode features")

    base_args = T.parse_args(rest)  # standard train_full_ram CLI

    specs = _specs(known.mode, known.model, known.features)
    idxs = _parse_indices(known.indices, len(specs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, f"ablate_{known.mode}_{known.model}")
    FileUtils.makedir(sweep_dir)
    load_logger = Logger(
        console_output=True,
        file_output=True,
        log_file=os.path.join(sweep_dir, "load.log"),
    )
    load_logger.show_header(
        f"PhenoNN {known.mode} ablation — {known.model} "
        f"({len(idxs)}/{len(specs)} runs)"
    )

    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, load_logger, device)

    for i in idxs:
        name, feats, experiment = specs[i]
        channels = _channels(feats)
        args_i = copy.deepcopy(base_args)
        args_i.type = known.model
        args_i.experiment = experiment
        args_i.ablate_features = ",".join(feats)

        if _is_done(args_i):
            load_logger.info(f"[skip] {args_i.experiment} — already completed.")
            continue

        paths_i, cfg_logger = T.setup_directories_and_logging(args_i)
        cfg_logger.show_header(
            f"{known.model} ablate [{name}] " f"channels={channels} feats={feats}"
        )

        _seed_all(args_i.seed)
        orig = _install(T, channels) if channels else None
        try:
            T.train_one_config(args_i, shared, cfg_logger, paths_i)
        finally:
            if orig is not None:
                _restore(T, orig)


if __name__ == "__main__":
    main()
