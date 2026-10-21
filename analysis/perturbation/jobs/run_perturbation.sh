#!/usr/bin/env bash
# Run all four weather-perturbation scenarios sequentially on one trained model,
# then aggregate them. If the checkpoint was trained with ablated features, the
# ablation is replayed automatically from the checkpoint.
#
#   CKPT=runs/<exp>/checkpoints/run_best_model.pth.tar \
#       bash analysis/perturbation/jobs/submit_perturbation.sh
#
# This file is the job body. Use submit_perturbation.sh for submission.
#SBATCH --job-name=perturb
#SBATCH --output=perturb_%j.out
#SBATCH --error=perturb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
: "${CKPT:?CKPT is not set: use submit_perturbation.sh or pass CKPT}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

EXP=$(basename "$(dirname "$(dirname "$CKPT")")")
OUT="${OUT:-runs/perturbation/$EXP}"
NSITES="${NSITES:-5000}"

phenonn_require_data \
    "$CKPT" \
    "$DATA/era5_pixelset" \
    "$DATA/LAI_pixelset" \
    "$DATA/PFT_pixelset" \
    "$DATA/selected_pixels01_1.nc"

COMMON=(
    --checkpoint   "$CKPT"
    --features_dir "$DATA/era5_pixelset"
    --target_dir   "$DATA/LAI_pixelset"
    --pft_dir      "$DATA/PFT_pixelset"
    --parent_map   "$DATA/selected_pixels01_1.nc"
    --predict_sites val --n_predict_sites "$NSITES"
    --output_dir   "$OUT"
)

echo "=== Perturbation study: $EXP -> $OUT (n_sites=$NSITES) ==="

"$PHENONN_PYTHON" -m analysis.perturbation.drought      "${COMMON[@]}"
"$PHENONN_PYTHON" -m analysis.perturbation.heatwave     "${COMMON[@]}"
"$PHENONN_PYTHON" -m analysis.perturbation.warming_co2  "${COMMON[@]}"
"$PHENONN_PYTHON" -m analysis.perturbation.winter_frost "${COMMON[@]}"

"$PHENONN_PYTHON" -m analysis.perturbation.aggregate_perturbation \
    --output_dir "$OUT"

echo "=== Done: $OUT ==="
