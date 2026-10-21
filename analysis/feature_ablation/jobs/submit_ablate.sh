#!/usr/bin/env bash
# Generate and submit the feature-ablation study.
#
# Usage:
#   PHENONN_SITE=hal       bash analysis/feature_ablation/jobs/submit_ablate.sh
#   PHENONN_SITE=leonardo  bash analysis/feature_ablation/jobs/submit_ablate.sh
#   PROFILE=smoke PHENONN_LOCAL=1 bash .../submit_ablate.sh
#   PHENONN_DRY_RUN=1 bash .../submit_ablate.sh
#   SUBSET=0.25           bash .../submit_ablate.sh
#
# HAL paper runs default to SUBSET=0.25 and PHENONN_MEM=120G.
# Extra arguments are forwarded to gen_jobs.py (e.g. --study all, --features).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

PROFILE="${PROFILE:-paper}"
STUDY="${STUDY:-features}"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    OUT="$OUTPUT_DIR"
elif [[ "$PROFILE" == "smoke" ]]; then
    OUT="runs/feature_ablation_smoke"
else
    OUT="runs/feature_ablation"
fi

SUBSET="${SUBSET:-}"
if [[ -z "$SUBSET" && "$PROFILE" == "paper" && "${PHENONN_SITE:-}" == "hal" ]]; then
    SUBSET=0.25
fi
if [[ -z "${PHENONN_MEM:-}" && "${PHENONN_SITE:-}" == "hal" ]]; then
    export PHENONN_MEM=120G
fi

subset_args=()
if [[ -n "$SUBSET" ]]; then
    subset_args=(--subset "$SUBSET")
fi

"$PHENONN_PYTHON" -m analysis.feature_ablation.gen_jobs \
    --project "$PROJECT" --data "$DATA" --profile "$PROFILE" \
    --study "$STUDY" --output_dir "$OUT" "${subset_args[@]}" "$@"

if [[ ! -f "$OUT/jobs/submit_all.sh" ]]; then
    echo "job generation failed: missing $OUT/jobs/submit_all.sh" >&2
    exit 1
fi
bash "$OUT/jobs/submit_all.sh"
