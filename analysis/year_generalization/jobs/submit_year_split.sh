#!/usr/bin/env bash
# Submit the temporal-generalization study: six jobs, one per split.
#   chrono, rand1, rand2, rand3, rand4, coldhot
#
# coldhot needs year_temp_rank.json in the repository root. Create it with
# rank_years_by_temp.py first. The job fails clearly when the file is missing.
#
#   PHENONN_SITE=hal      bash analysis/year_generalization/jobs/submit_year_split.sh
#   PHENONN_DRY_RUN=1     bash analysis/year_generalization/jobs/submit_year_split.sh
#   PHENONN_LOCAL=1       bash analysis/year_generalization/jobs/submit_year_split.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

SPLITS=(chrono rand1 rand2 rand3 rand4 coldhot)

if [[ "${PHENONN_DRY_RUN:-0}" == "1" ]]; then
    for split in "${SPLITS[@]}"; do
        echo "  GPU: $split"
        if phenonn_is_local; then
            printf '[dry-run] SPLIT=%q bash %q\n' "$split" \
                "$HERE/run_year_split.sh"
        else
            dry_opts=()
            mapfile -t dry_opts < <(phenonn_sbatch_opts)
            printf '[dry-run] sbatch'
            printf ' %q' "${dry_opts[@]}"
            printf ' %q' "--job-name=yg-$split" "--export=ALL,SPLIT=$split"
            printf ' %q\n' "$HERE/run_year_split.sh"
        fi
    done
    exit 0
fi

if phenonn_is_local; then
    for split in "${SPLITS[@]}"; do
        echo "  GPU: $split"
        SPLIT="$split" bash "$HERE/run_year_split.sh"
    done
    echo "== ${#SPLITS[@]} splits completed =="
    exit 0
fi

mapfile -t opts < <(phenonn_sbatch_opts)
for split in "${SPLITS[@]}"; do
    echo "  GPU: $split"
    sbatch "${opts[@]}" \
        --job-name="yg-$split" \
        --export="ALL,SPLIT=$split" \
        "$HERE/run_year_split.sh"
done
echo "== ${#SPLITS[@]} splits submitted =="
