#!/usr/bin/env bash
# Full evaluation of one or more trained runs:
#   1) prediction on the validation split
#   2) the four perturbation scenarios
#   3) comparison of the scenarios
# Ablation and daily-LAI output are replayed automatically from the checkpoint.
#
#   EXPERIMENTS="exp_a exp_b" RUNS_ROOT=runs \
#       bash analysis/perturbation/jobs/submit_perturbation.sh
#
# This file is the job body. Set MODE=eval when using submit_perturbation.sh.
#SBATCH --job-name=perturb_eval
#SBATCH --output=perturb_eval_%j.out
#SBATCH --error=perturb_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
: "${EXPERIMENTS:?EXPERIMENTS is not set: use MODE=eval and list experiment names}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

RUNS_ROOT="${RUNS_ROOT:-runs}"
NSITES="${NSITES:-5000}"

phenonn_require_data \
    "$DATA/era5_pixelset" \
    "$DATA/LAI_pixelset" \
    "$DATA/PFT_pixelset" \
    "$DATA/selected_pixels01_1.nc"

DIRS=(
    --features_dir "$DATA/era5_pixelset"
    --target_dir   "$DATA/LAI_pixelset"
    --pft_dir      "$DATA/PFT_pixelset"
    --parent_map   "$DATA/selected_pixels01_1.nc"
)

find_ckpt() {
    local dir="$1/checkpoints" candidate
    for candidate in run_best_model.pth.tar best_model.pth.tar \
        run_best_model.pth best_model.pth; do
        if [ -f "$dir/$candidate" ]; then
            echo "$dir/$candidate"
            return 0
        fi
    done
    return 1
}

for EXP in $EXPERIMENTS; do
    RUN_DIR="$RUNS_ROOT/$EXP"
    CKPT="$(find_ckpt "$RUN_DIR" || true)"
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "!! no best checkpoint under $RUN_DIR/checkpoints: skipping"
        continue
    fi
    echo "================ $EXP ================"

    # --n_curves 100 keeps the per-site figure below the matplotlib image limit.
    "$PHENONN_PYTHON" -m phenonn.prediction.predict \
        --checkpoint "$CKPT" "${DIRS[@]}" \
        --predict_sites val --n_predict_sites "$NSITES" \
        --n_curves 100 --scatter_years \
        --output_csv "$RUN_DIR/pred_val.csv"

    OUT="$RUNS_ROOT/perturbation/$EXP"
    PCOMMON=(--checkpoint "$CKPT" "${DIRS[@]}"
             --predict_sites val --n_predict_sites "$NSITES" --output_dir "$OUT")
    "$PHENONN_PYTHON" -m analysis.perturbation.drought      "${PCOMMON[@]}"
    "$PHENONN_PYTHON" -m analysis.perturbation.heatwave     "${PCOMMON[@]}"
    "$PHENONN_PYTHON" -m analysis.perturbation.warming_co2  "${PCOMMON[@]}"
    "$PHENONN_PYTHON" -m analysis.perturbation.winter_frost "${PCOMMON[@]}"

    "$PHENONN_PYTHON" -m analysis.perturbation.aggregate_perturbation \
        --output_dir "$OUT"
done

echo "=== Done ==="
