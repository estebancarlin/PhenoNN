#!/usr/bin/env python3
"""
train_ablate.py — train the PhenoNN model with one input feature masked out.

Thin launcher around `phenonn.training.train_full_ram`. It reads one extra flag,
`--ablate_feature NAME`, then trains exactly as `train_full_RAM` would, except
the named input channel is zeroed before every forward pass (train AND val).

Why zeroing works as "removing" a feature
------------------------------------------
Features are z-scored by the dataset, so a channel's mean is 0. Setting that
channel to 0 for every sample = mean-imputation: the channel becomes constant
and carries no information, so the model cannot use it. The architecture and the
full input are unchanged, which keeps ΔR² vs the baseline directly comparable
(standard input-ablation).

Only NON-PFT features may be ablated (channels before PFT_START). PFT fractions
are kept in every run.

No change to `phenonn/` — the masking is a forward pre-hook installed by
monkeypatching the two model factories in `train_full_RAM`'s namespace. The hook
adds no parameters, so saved checkpoints are byte-compatible with a normal run.
The ablated feature name is also written to the checkpoint as
``ablate_features`` so prediction and other analysis scripts can replay it.

Usage
-----
    # baseline (nothing masked):
    python -m analysis.feature_ablation.train_ablate --ablate_feature none \\
        <all the usual phenonn.training.train_full_ram args> --experiment abl_baseline

    # ablate one feature:
    python -m analysis.feature_ablation.train_ablate --ablate_feature Tmean \\
        <all the usual phenonn.training.train_full_ram args> --experiment abl_Tmean

`gen_jobs.py` writes one SLURM script per feature (+ baseline) with these calls.
"""

import sys

from analysis.feature_ablation._hooks import ZeroChannels
from phenonn.utils.config import ALL_FEATURES, PFT_START

NON_PFT_FEATURES = list(ALL_FEATURES[:PFT_START])


def _pop_ablate_feature(argv) -> str:
    """Remove `--ablate_feature NAME` (or `=NAME`) from argv and return NAME.

    Stripping it before `phenonn.training.train_full_ram` parses argv keeps both its
    argparse and its `_cli_explicit_args()` scan happy (they read sys.argv).
    Returns "" when the flag is absent.
    """
    flag = "--ablate_feature"
    for i, a in enumerate(argv):
        if a == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value (a feature name or 'none').")
            val = argv[i + 1]
            del argv[i : i + 2]
            return val
        if a.startswith(flag + "="):
            val = a.split("=", 1)[1]
            del argv[i]
            return val
    return ""


def _resolve_channel(name: str):
    """Map a feature name to its input-channel index, or None for the baseline."""
    if name.strip().lower() in ("", "none", "baseline"):
        print("[ablation] baseline — no feature masked", flush=True)
        return None
    if name not in ALL_FEATURES:
        raise SystemExit(
            f"Unknown feature {name!r}.\nAblatable (non-PFT) features: "
            f"{', '.join(NON_PFT_FEATURES)}"
        )
    ch = ALL_FEATURES.index(name)
    if ch >= PFT_START:
        raise SystemExit(
            f"{name!r} is a PFT channel (index {ch} >= PFT_START={PFT_START}); "
            f"this study ablates non-PFT features only."
        )
    print(f"[ablation] masking feature {name!r} -> input channel {ch}", flush=True)
    return ch


def _install_ablation(module, ch: int) -> None:
    """Patch build_model / build_model_pft in `module` so the returned model
    zeroes input channel `ch` (dim 1: features) before every forward."""
    hook = ZeroChannels([ch])

    orig_build_model = module.build_model
    orig_build_model_pft = module.build_model_pft

    def build_model(args):
        m = orig_build_model(args)
        m.register_forward_pre_hook(hook)
        return m

    def build_model_pft(args, norm_stats):
        m = orig_build_model_pft(args, norm_stats)
        m.register_forward_pre_hook(hook)
        return m

    module.build_model = build_model
    module.build_model_pft = build_model_pft


def main():
    name = _pop_ablate_feature(sys.argv)
    ch = _resolve_channel(name)

    import phenonn.training.train_full_ram as trainer

    if ch is not None:
        _install_ablation(trainer, ch)

    # Persist the ablation in the checkpoint so predict.py can replay exactly
    # the same masked input at inference time.
    orig_parse_args = trainer.parse_args

    def parse_args(argv=None):
        args = orig_parse_args(argv)
        args.ablate_features = "" if ch is None else name
        return args

    trainer.parse_args = parse_args
    trainer.main()


if __name__ == "__main__":
    main()
