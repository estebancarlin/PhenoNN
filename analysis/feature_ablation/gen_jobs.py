#!/usr/bin/env python3
"""
gen_jobs.py — write portable job scripts for the feature-ablation study.

The scripts are generated into ``--jobs_dir`` (default
``runs/feature_ablation/jobs``) and are site-agnostic: they contain no
partition/account/QoS/memory directives. Those are supplied by
``analysis/_cluster.sh`` at submission time (``PHENONN_SITE=hal|leonardo`` or
the matching ``PHENONN_*`` environment variables).

Studies:
    features  baseline + one job per non-PFT feature (``--profile paper``)
    groups    in-process leave-one-group-out slices (default 3 groups/GPU)
    combo     fixed combo ablation across model types/sizes
              (``--skip-baseline`` reuses existing baselines)
    all       all of the above

Typical use::

    export PHENONN_ROOT="$PWD"
    python -m analysis.feature_ablation.gen_jobs --profile paper
    bash runs/feature_ablation/jobs/submit_all.sh

    # short test, no SLURM needed
    python -m analysis.feature_ablation.gen_jobs --profile smoke
    PHENONN_LOCAL=1 bash runs/feature_ablation_smoke/jobs/submit_all.sh
"""

import argparse
import os
import shlex
from pathlib import Path

from analysis.feature_ablation._profiles import (
    DATASET_FILES,
    FEATURES,
    OUTPUT_DIRS,
    PROFILES,
    data_cli_args,
    default_data_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SBATCH_HEADER = """#!/usr/bin/env bash
#SBATCH --job-name={job}
#SBATCH --output={logdir}/{tag}_%j.out
#SBATCH --error={logdir}/{tag}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
export PHENONN_ROOT={project}
export DATA={data}
export PHENONN_VENV={venv}
source "$PHENONN_ROOT/analysis/_cluster.sh"
phenonn_cluster_init
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p {logdir}
"""


def _quote(value):
    return shlex.quote(str(value))


def _join_args(args):
    return " ".join(_quote(a) for a in args)


def _profile_common(profile):
    out = []
    for key, value in profile.items():
        flag = f"--{key}"
        if value is True:
            out.append(flag)
        elif value is False or value is None:
            continue
        else:
            out.extend([flag, str(value)])
    return _join_args(out)


def _dataset_args(data):
    return _join_args(data_cli_args(data))


def _dataset_paths(data):
    root = Path(data)
    paths = [str(root / filename) for filename in DATASET_FILES.values()]
    return _join_args(paths)


def _header(job, tag, logdir, project, data, venv, cpus):
    return SBATCH_HEADER.format(
        job=job,
        tag=tag,
        logdir=logdir,
        project=_quote(project),
        data=_quote(data),
        venv=_quote(venv),
        cpus=cpus,
    )


def _write(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path.name


def _feature_script(feat, common, project, data, venv, out_dir, logdir):
    baseline = feat is None
    tag = "baseline" if baseline else f"ablate_{feat}"
    job = "abl_baseline" if baseline else f"abl_{feat}"
    experiment = job
    mask = "none" if baseline else feat
    cmd = (
        f'"$PHENONN_PYTHON" -m analysis.feature_ablation.train_ablate '
        f"--ablate_feature {_quote(mask)} {common} {_dataset_args(data)} "
        f'--prefix "" --output_dir {_quote(out_dir)} '
        f"--experiment {_quote(experiment)}"
    )
    return _header(job, tag, logdir, project, data, venv, 8) + "\n" + cmd + "\n"


def _group_script(model, lo, hi, common, project, data, venv, out_dir, logdir):
    tag = f"groups_{lo}_{hi}"
    cmd = (
        f'"$PHENONN_PYTHON" -m analysis.feature_ablation.run_ablate_inproc '
        f"--mode groups --model {_quote(model)} --indices {lo}-{hi} "
        f"{common} {_dataset_args(data)} "
        f'--prefix "" --output_dir {_quote(out_dir)}'
    )
    return (
        _header(f"abl-groups-{lo}-{hi}", tag, logdir, project, data, venv, 12)
        + "\n"
        + cmd
        + "\n"
    )


def _combo_script(common, project, data, venv, out_dir, logdir, skip_baseline=False):
    extra = "--skip-baseline " if skip_baseline else ""
    cmd = (
        f'"$PHENONN_PYTHON" -m analysis.feature_ablation.run_ablate_combo_models '
        f"{extra}{common} {_dataset_args(data)} "
        f'--prefix "" --output_dir {_quote(out_dir)}'
    )
    return (
        _header("abl-combo-models", "combo_models", logdir, project, data, venv, 12)
        + "\n"
        + cmd
        + "\n"
    )


SUBMIT_ALL = """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PHENONN_ROOT={project}
source "$PHENONN_ROOT/analysis/_cluster.sh"
phenonn_cluster_init
mkdir -p {logdir}
if [[ "${{PHENONN_DRY_RUN:-0}}" != "1" ]]; then
    phenonn_require_data {data_paths}
fi
mapfile -t jobs < <(find "$HERE" -maxdepth 1 -name 'run_*.sh' -print | sort)
if [[ "${{#jobs[@]}}" -eq 0 ]]; then
    echo "no run_*.sh jobs found in $HERE" >&2
    exit 1
fi
phenonn_submit "${{jobs[@]}}"
"""


def _parse_features(value):
    if not value:
        return None
    return [f.strip() for f in value.split(",") if f.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default="paper", choices=sorted(PROFILES))
    parser.add_argument(
        "--study",
        default="all",
        choices=["features", "groups", "combo", "all"],
    )
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated per-feature subset (default: profile setting).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override the model for the group sweep (default: profile type).",
    )
    parser.add_argument(
        "--per-gpu",
        type=int,
        default=3,
        help="Groups per in-process job (default: 3).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Combo only: reuse existing baselines instead of retraining them.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PHENONN_ROOT", str(REPO_ROOT)),
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("DATA", ""),
        help="Dataset root (default: $DATA or <repo>/pixelset1 then <repo>/../pixelset1).",
    )
    parser.add_argument(
        "--venv",
        default=os.environ.get("PHENONN_VENV", ""),
        help="Path to the venv activate script (default: <project>/.venv/...).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Training output root (default: profile-specific runs/<study>).",
    )
    parser.add_argument(
        "--jobs_dir",
        default="",
        help="Where the scripts are written (default: <output_dir>/jobs).",
    )
    parser.add_argument(
        "--common_args",
        default="",
        help="Override the generated training arguments with this string.",
    )
    args = parser.parse_args()
    if args.per_gpu < 1:
        parser.error("--per-gpu must be at least 1")

    from analysis.feature_ablation._common import GROUPS, NON_PFT_FEATURES

    profile = PROFILES[args.profile]
    project = str(Path(args.project).expanduser().resolve())
    data = args.data or default_data_dir(project)
    venv = args.venv or str(Path(project) / ".venv" / "bin" / "activate")
    out_dir = args.output_dir or OUTPUT_DIRS[args.profile]
    jobs_dir = Path(args.jobs_dir or Path(out_dir) / "jobs")
    logdir = str((Path(out_dir) / "logs").resolve())
    jobs_dir.mkdir(parents=True, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    for old in jobs_dir.glob("run_*.sh"):
        old.unlink()

    common = args.common_args or _profile_common(profile)

    features = _parse_features(args.features)
    if features is None:
        features = FEATURES[args.profile]
    if features is None:
        features = list(NON_PFT_FEATURES)

    unknown = [f for f in features if f not in NON_PFT_FEATURES]
    if unknown:
        parser.error(
            f"unknown or non-ablatable feature(s): {', '.join(unknown)}; "
            f"choose from {', '.join(NON_PFT_FEATURES)}"
        )

    model = args.model or str(profile.get("type", "aelstm"))
    written = []

    if args.study in ("features", "all"):
        written.append(
            _write(
                jobs_dir / "run_baseline.sh",
                _feature_script(None, common, project, data, venv, out_dir, logdir),
            )
        )
        for feat in features:
            name = f"run_ablate_{feat}.sh"
            written.append(
                _write(
                    jobs_dir / name,
                    _feature_script(feat, common, project, data, venv, out_dir, logdir),
                )
            )

    if args.study in ("groups", "all"):
        n = len(GROUPS)
        for lo in range(0, n, args.per_gpu):
            hi = min(lo + args.per_gpu - 1, n - 1)
            name = f"run_groups_{lo}_{hi}.sh"
            written.append(
                _write(
                    jobs_dir / name,
                    _group_script(
                        model,
                        lo,
                        hi,
                        common,
                        project,
                        data,
                        venv,
                        out_dir,
                        logdir,
                    ),
                )
            )

    if args.study in ("combo", "all"):
        name = "run_combo.sh"
        written.append(
            _write(
                jobs_dir / name,
                _combo_script(
                    common,
                    project,
                    data,
                    venv,
                    out_dir,
                    logdir,
                    skip_baseline=args.skip_baseline,
                ),
            )
        )

    _write(
        jobs_dir / "submit_all.sh",
        SUBMIT_ALL.format(
            project=_quote(project),
            logdir=_quote(logdir),
            data_paths=_dataset_paths(data),
        ),
    )

    print(f"Wrote {len(written) + 1} scripts to {jobs_dir}/")
    print("  " + ", ".join(sorted(written + ["submit_all.sh"])))
    print("\nSubmit (or run locally):")
    print(f"  bash {jobs_dir / 'submit_all.sh'}")
    print(f"  PHENONN_LOCAL=1 bash {jobs_dir / 'submit_all.sh'}")
    if args.study in ("features", "all"):
        print("Features:", ", ".join(features))
    if args.study in ("groups", "all"):
        print("Group sweep model:", model, f"({len(GROUPS)} runs)")


if __name__ == "__main__":
    main()
