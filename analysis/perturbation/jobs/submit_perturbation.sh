#!/usr/bin/env bash
# Submit the climate-perturbation study.
#
# One checkpoint:
#   CKPT=runs/<experiment>/checkpoints/run_best_model.pth.tar \
#       PHENONN_SITE=hal \
#       bash analysis/perturbation/jobs/submit_perturbation.sh
#
# Several experiments:
#   MODE=eval EXPERIMENTS="exp_a exp_b" RUNS_ROOT=runs \
#       PHENONN_SITE=hal \
#       bash analysis/perturbation/jobs/submit_perturbation.sh
#
# Dry run:
#   PHENONN_DRY_RUN=1 CKPT=... \
#       bash analysis/perturbation/jobs/submit_perturbation.sh
#
# Local run:
#   PHENONN_LOCAL=1 PHENONN_SITE=local CKPT=... NSITES=100 \
#       bash analysis/perturbation/jobs/submit_perturbation.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

MODE="${MODE:-run}"
case "$MODE" in
    run)
        : "${CKPT:?CKPT is not set: pass CKPT=<checkpoint>}"
        phenonn_submit "$HERE/run_perturbation.sh"
        ;;
    eval)
        : "${EXPERIMENTS:?EXPERIMENTS is not set: pass a space-separated list}"
        phenonn_submit "$HERE/run_perturbation_eval.sh"
        ;;
    *)
        echo "unknown MODE=$MODE (expected run or eval)" >&2
        exit 2
        ;;
esac
