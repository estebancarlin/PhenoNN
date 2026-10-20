#!/usr/bin/env python3
"""
smoke.py — short end-to-end feature-ablation test (no SLURM, no full runs).

Runs the ``SMOKE`` profile: 2008-2009 train / 2010 val, five sites, one epoch,
plus the baseline, then aggregates the selected features. Data is loaded once
because the runs happen in one in-process sweep.

    python -m analysis.feature_ablation.smoke
    python -m analysis.feature_ablation.smoke --features Tmean
    python -m analysis.feature_ablation.smoke --data /path/to/pixelset1

Outputs go to ``runs/feature_ablation_smoke`` by default.
"""

import argparse
import os
import sys
from pathlib import Path

from analysis.feature_ablation._profiles import (
    OUTPUT_DIRS,
    SMOKE,
    cli_args,
    data_cli_args,
    default_data_dir,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        default="Tmean,tp_sum",
        help="Comma-separated features (default: Tmean,tp_sum).",
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("DATA", ""),
        help="Dataset root (default: $DATA or <repo>/pixelset1 then <repo>/../pixelset1).",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIRS["smoke"],
        help="Training output root (default: runs/feature_ablation_smoke).",
    )
    parser.add_argument("--project", default=os.environ.get("PHENONN_ROOT", ""))
    args = parser.parse_args()

    project = Path(args.project).resolve() if args.project else Path.cwd()
    data = args.data or default_data_dir(project)
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    if not features:
        parser.error("--features must name at least one feature")

    train_argv = [
        "--mode",
        "features",
        "--model",
        SMOKE["type"],
        "--features",
        ",".join(features),
        *data_cli_args(data),
        "--prefix",
        "",
        "--output_dir",
        args.output_dir,
        *cli_args(SMOKE),
    ]

    from analysis.feature_ablation import aggregate, run_ablate_inproc

    old_argv = sys.argv
    try:
        sys.argv = ["analysis.feature_ablation.run_ablate_inproc", *train_argv]
        run_ablate_inproc.main()

        sys.argv = [
            "analysis.feature_ablation.aggregate",
            "--kind",
            "features",
            "--runs_dir",
            args.output_dir,
            "--features",
            ",".join(features),
            "--output",
            str(Path(args.output_dir) / "feature_ablation.png"),
        ]
        aggregate.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
