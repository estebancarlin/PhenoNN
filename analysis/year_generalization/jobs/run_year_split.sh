#!/usr/bin/env bash
# One GPU runs one year split of the temporal-generalization study. The model is
# fixed (aelstm h256/d2, huber, pft_mixing meteo_only); only the train and
# validation years change. Do not submit this file directly: use
# submit_year_split.sh.
#
#SBATCH --job-name=yeargen
#SBATCH --output=runs/year_generalization/logs/yeargen_%x_%j.out
#SBATCH --error=runs/year_generalization/logs/yeargen_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail
: "${SPLIT:?SPLIT is not set (chrono|rand1..4|coldhot): use submit_year_split.sh}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${PHENONN_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init

ALL_YEARS="${ALL_YEARS:-1992-2019}"
YEAR_SUBSET="${YEAR_SUBSET:-0.5}"
TEMP_RANK_JSON="${TEMP_RANK_JSON:-$PROJECT/runs/year_generalization/year_temp_rank.json}"

phenonn_require_data \
    "$DATA/era5_pixelset" \
    "$DATA/LAI_pixelset" \
    "$DATA/PFT_pixelset" \
    "$DATA/selected_pixels005_balanced.nc" \
    "$DATA/selected_pixels01_1.nc" \
    "$DATA/norm_stats_1992_2019.json" \
    "$DATA/CO2_1700_2023_TRENDYv2024.txt"

if [[ "$SPLIT" == "coldhot" ]]; then
    phenonn_require_data "$TEMP_RANK_JSON"
fi

# Fixed model configuration. Keep it identical for every split so Delta R2
# isolates the effect of the year split.
"$PHENONN_PYTHON" -m analysis.year_generalization.run_year_split \
    --split "$SPLIT" --temp_rank_json "$TEMP_RANK_JSON" \
    --all_years "$ALL_YEARS" \
    --features_dir    "$DATA/era5_pixelset" \
    --target_dir      "$DATA/LAI_pixelset" \
    --pft_dir         "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_balanced.nc" \
    --parent_map      "$DATA/selected_pixels01_1.nc" \
    --stats_path      "$DATA/norm_stats_1992_2019.json" \
    --co2_path        "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --val_fraction_of_grid 100 \
    --n_years_per_epoch 30 --n_sites_per_epoch 250000 --n_val_sites 50000 \
    --subset "$YEAR_SUBSET" \
    --type aelstm --hidden_size 256 --num_layers 2 \
    --learning_rate 1e-3 --weight_decay 1e-5 \
    --loss_type huber \
    --pft_mixing --pft_meteo_only \
    --num_epochs 300 --patience 15 \
    --batch_size 64 --amp --num_workers 2 \
    --prefix "" --output_dir "$PROJECT/runs/year_generalization"
