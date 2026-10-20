#!/usr/bin/env bash
# Generate and submit the fixed-combo model/size sweep (baselines optional).
#
# Usage:
#   PHENONN_SITE=hal       bash analysis/feature_ablation/jobs/submit_combo.sh
#   PHENONN_SITE=leonardo  bash analysis/feature_ablation/jobs/submit_combo.sh
#   PHENONN_LOCAL=1        bash analysis/feature_ablation/jobs/submit_combo.sh
#   PHENONN_DRY_RUN=1      bash analysis/feature_ablation/jobs/submit_combo.sh
#   bash .../submit_combo.sh --skip-baseline   # reuse existing baselines
#
# Submit this only after the per-feature/group study has finished: on a
# single-GPU quota, a newly queued combo job can otherwise preempt it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

PROFILE="${PROFILE:-paper}"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    OUT="$OUTPUT_DIR"
elif [[ "$PROFILE" == "smoke" ]]; then
    OUT="runs/feature_ablation_smoke"
else
    OUT="runs/feature_ablation"
fi

"$PHENONN_PYTHON" -m analysis.feature_ablation.gen_jobs \
    --project "$PROJECT" --data "$DATA" --profile "$PROFILE" \
    --study combo --output_dir "$OUT" "$@"

if [[ ! -f "$OUT/jobs/submit_all.sh" ]]; then
    echo "job generation failed: missing $OUT/jobs/submit_all.sh" >&2
    exit 1
fi
bash "$OUT/jobs/submit_all.sh"
