# Year generalization

This study measures how prediction skill changes when the training and
validation years are split in different ways. All splits use the same sites and
the same model configuration, so `Delta R2` isolates the effect of the year
split.

The split definitions are:

| Split | Training years | Validation years |
|---|---|---|
| `chrono` | first 18 years | last 10 years |
| `rand1` to `rand4` | 18 random years, seeds 1001 to 1004 | the other 10 years |
| `coldhot` | 18 coldest years | 10 hottest years |

The default year pool is 1992 to 2019, which gives 28 years. That matches the
18 training and 10 validation years used by the chronological baseline.

## Before you run

Run every command from the repository root after following
[the analysis setup](../README.md#before-you-start).

```bash
cd /path/to/PhenoNN
source .venv/bin/activate
export DATA=/path/to/pixelset1
```

`coldhot` needs `year_temp_rank.json`, which is produced by
`rank_years_by_temp.py`. The ranking script reads the ERA5 files directly and
does not need a checkpoint or a GPU.

## Rank the years

```bash
python -m analysis.year_generalization.rank_years_by_temp \
    --features_dir "$DATA/era5_pixelset" \
    --year_start 1992 --year_end 2019 \
    --out year_temp_rank.json
```

The script computes the mean `Tmean` over every site and day in each ERA5 file,
then writes:

```text
year_temp_rank.json
```

The JSON contains `years_cold_to_hot`, the per-year means, and the variable
name. Run this command before submitting the `coldhot` split.

## Run one split

The split runner forwards every unrecognized flag to the normal
`train_full_ram` parser. Supply the same fixed model configuration for every
split.

```bash
python -m analysis.year_generalization.run_year_split \
    --split rand1 --all_years 1992-2019 \
    --temp_rank_json ./year_temp_rank.json \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_balanced.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --val_fraction_of_grid 100 \
    --n_years_per_epoch 30 --n_sites_per_epoch 250000 --n_val_sites 50000 \
    --subset 0.5 \
    --type aelstm --hidden_size 256 --num_layers 2 \
    --learning_rate 1e-3 --weight_decay 1e-5 \
    --loss_type huber \
    --pft_mixing --pft_meteo_only \
    --num_epochs 300 --patience 15 \
    --batch_size 64 --amp --num_workers 2 \
    --prefix "" --output_dir ./runs/year_generalization
```

The experiment name is set automatically to `yeargen_<split>`. A split that
already has a `config.json` in its run directory is skipped.

Use `--all_years` explicitly when `--train_years` and `--val_years` are not
supplied. If they are supplied, their union is used as the year pool.

## Submit all splits

The job wrapper queues six jobs, one per split:

```bash
export PHENONN_SITE=hal
bash analysis/year_generalization/jobs/submit_year_split.sh
```

Use `PHENONN_SITE=leonardo` for Leonardo or `PHENONN_SITE=local` for a local
run. See [jobs/README.md](jobs/README.md) for dry-run and environment details.

## Read the results

Each split writes under `runs/year_generalization/yeargen_<split>/`. Compare
the validation `R2` values:

```text
Delta R2 = R2(split) - R2(chrono)
```

A negative `Delta R2` means the alternative split is harder than validating on
the future. Random-year splits estimate the cost of the chronological split
when the validation years are not specifically the future. The `coldhot` split
tests whether training on cold years transfers to hot years.

## Notes

- The model configuration must stay fixed across splits. Change only the year
  lists and the experiment name.
- `rank_years_by_temp.py` uses the full pixelset mean, not a site-weighted
  subset.
- The job script uses `analysis/_cluster.sh` for the environment, site presets,
  dry-run mode, and local execution.
