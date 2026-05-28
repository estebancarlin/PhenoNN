# PhenoCam GCC Prediction

Predicts 1 year (for 3 days per monnth, always day 5, 15, 25) LAI (Leaf area index) from 720-day meteorological windows

## Project layout

folders: phenocam/, util/, data/, 

```
project/
├── Phenocam/                    # Existing RTnn code (unchanged)
│   ├── models/
│   │   ├── rnn.py
│   │   ├── fcn.py
│   │   ├── Transformer.py
│   │   └── Transformerbis.py
│   ├── main.py                     # Training pipeline
│   ├── dataset.py                  
│   ├── predict_1year.py
│   └── feature_engineering.py
├── util/                
│   ├── diagnostic.py    
    ├── logger.py          
│   ├── evaluater.py         
│   ├── logger.py          
│   ├── evaluater.py     
│   ├── model_loader.py              
│   └── wrapper.py           
└── data/
    ├── feature.csv              
│   └── taget.csv
── runs/
    ├── 
```
```

## Data format

The target CSV file must have the columns: site_id,date,year,month,day,file,lat_requested,lon_requested,row,col,lat_pixel,lon_pixel,LAI_raw,LAI

the feature CSV file must have the columns: site_id,date,year,month,day,file,lat_requested,lon_requested,row,col,lat_pixel,lon_pixel,pft1_frac,pft2_frac,pft3_frac,pft4_frac,pft5_frac,pft6_frac,pft7_frac,pft8_frac,pft9_frac,pft10_frac,pft11_frac,pft12_frac,pft13_frac,pft14_frac,pft15_frac,tmin,tmax,daylength,prcp,srad,vpd,swe




| Column      | Type    | Description                        |
|-------------|---------|------------------------------------|
| year        | int     | Calendar year                      |
| doy         | int     | Day of year (1-366)                |
| tmin        | float   | Min temperature (°C)               |
| tmax        | float   | Max temperature (°C)               |
| daylength   | float   | Day length (hours)                 |
| vpd         | float   | Vapor pressure deficit             |
| swa         | float   | Soil water availability            |
| radiation   | float   | Solar radiation (W/m²)             |
| snow        | float   | Snow amount                        |
| mat         | float   | Mean annual temperature (static)   |
| map         | float   | Mean annual precipitation (static) |
| gcc_lowess  | float   | Smoothed GCC target                |

Filename format: `{PFT}_{sitename}.csv` (e.g., `DB_asuhighlands.csv`).

## Quick start

### 1. Train (leave-site-out split)

```bash
python -m phenocam.main \
    --data_dir ./data/DB/ \
    --type lstm \
    --hidden_size 128 \
    --num_layers 2 \
    --num_epochs 50 \
    --batch_size 64 \
    --output_dir ./runs \
    --experiment DB_lstm_v1
```

### 2. Train (year-based split)
liste param
    p = argparse.ArgumentParser(description="LAI Prediction")

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing site CSVs ({PFT}_{site}.csv)")
    p.add_argument("--stats_path", type=str, default="",
                    help="Path to precomputed norm_stats.json (computed if empty)")
    p.add_argument("--output_dir", type=str, default="./runs",
                    help="Root output directory for logs, checkpoints, etc.")
    p.add_argument("--experiment", type=str, default="exp01",
                    help="Experiment name (creates sub-folder)")
    p.add_argument("--use_site_features", type=str, default="all",
                    help="Which static site features to use: 'all', 'none', or comma-separated list (e.g. 'lat,lon')")
    p.add_argument("--use_derived_features", type=str, default="all",
                    help="Which derived features to use: 'all', 'none', or comma-separated list (e.g. 'gdd_0,cdd')")
                    
    # Normalization
    p.add_argument("--gcc_norms_csv", type=str, default="",
                    help="Path to per-site GCC min/max CSV for inter-site "
                         "normalization (e.g. gcc_rcc_mins_site_veg.csv)")
    p.add_argument("--residual_csv", type=str, default="",
                    help="Path to predictions.csv from a first model run. "
                         "If provided, the target becomes obs-pred (residual learning).")

    # Split strategy
    p.add_argument("--split_mode", type=str, default="site",
                    choices=["site", "year"],
                    help="'site': leave-site-out. 'year': all sites, split by year.")
    p.add_argument("--val_fraction", type=float, default=0.2,
                    help="Fraction of sites for validation (split_mode=site)")
    p.add_argument("--train_years", type=str, default="",
                    help="Comma-separated training years (split_mode=year)")
    p.add_argument("--val_years", type=str, default="",
                    help="Comma-separated validation years (split_mode=year)")

    # Model
    p.add_argument("--type", type=str, default="lstm",
                    choices=["lstm", "gru", "fcn", "fullyconnected", "transformer", "linear",
                     "linear_perday", "bitransformer", "1year_bitransformer", "1year_lstm"],)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--feed_forward_trans", type=int, default=4)
    p.add_argument("--feed_forward_encoder", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--seq_length", type=int, default=365)
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.)
    p.add_argument("--dropout_trans", type=float, default=0.)

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=2e-3)
    p.add_argument("--loss_type", type=str, default="nmae", 
                    choices=["mse", "huber", "mae",'nmae', 'nmse','wmse', 'logcosh', 'smoothl1', 'gradient'],)
    p.add_argument("--huber_beta", type=float, default=1.0,
                    help="Delta parameter for Huber loss (only used if --loss_type huber)")
    p.add_argument("--gradient_loss_weight", type=float, default=0.5,
                    help="Weight λ for temporal gradient term (only if --loss_type gradient)")
    p.add_argument("--gradient_base_loss", type=str, default="mse",
                    choices=["mse", "huber", "mae"],
                    help="Base loss for gradient-aware loss")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (epochs without improvement)")
    p.add_argument("--stride", type=int, default=7,
                    help="Stride between consecutive training windows (default 7 "
                         "to reduce correlation between samples)")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Max gradient norm for clipping (0 = no clipping)")
    p.add_argument("--sites_per_epoch", type=int, default=0,
                    help="Number of training sites sampled per epoch (0 = all sites). "
                         "Each epoch draws a different random subset for regularisation.")
    p.add_argument("--random_stride", type=int, default=0,
                    help="Number of random samples per site-year per epoch (0 = use fixed stride). "
                         "Each epoch draws a different random subset of days.")
    p.add_argument("--feature_mode", type=str, default="all",
                    choices=["all", "site_only", "meteo_only"],
                    help="Feature selection: 'all' (default), 'site_only' (cyclic+static+PFT "
                         "for climatology model), 'meteo_only' (dynamic+cyclic for anomaly model)")
    p.add_argument("--full_year", action="store_true",
                    help="Predict a full year (365 days) from a 730-day input window. "
                         "Use with --type 1year_bitransformer and 1year_lstm --seq_length 730.")

```bash
python -m phenocam.main \
    --data_dir ./data/DB/ \
    --split_mode year \
    --train_years 2018,2019,2020 \
    --val_years 2021 \
    --type transformer \
    --embed_size 64 \
    --nhead 4 \
    --num_epochs 50 \
    --output_dir ./runs \
    --experiment DB_transformer_v1
```
python -m phenocam.main --data_dir ./data/DB/ --split_mode year --train_years 2007-2021 --val_years 2022-2023 --type transformer --embed_size 128 --nhead 4 --num_epochs 50 --dropout 0.1 --output_dir ./runs --experiment DB_transformer_year_emb128

python -m phenocam.main --data_dir ./data/DB/ --split_mode year --train_years 2005-2021 --val_years 2022-2023 --type lstm --hidden_size 256 --num_layer 2 --num_epochs 100 --output_dir ./runs --experiment DB_lstm_year_hid256_GDD_CDD

### 3. Predict year by year

```bash
python -m phenocam.predict_1year \
    --checkpoint ./runs/DB_lstm_v1/checkpoints/best_model.pth \
    --data_dir ./data/DB/ \
    --predict_years 2020,2021 \
    --output_csv ./runs/DB_lstm_v1/predictions.csv
```
python -m phenocam.predict --checkpoint ./runs/DB_transformer_year_emb128/checkpoints/best_model.pth --data_dir ./data/ --predict_years 2022,2023 --output_csv ./runs/DB_transformer_year_emb128/predictions.csv



## Architecture

```
Input: 365 days × N features
       ┌─────────────────────────────────────────┐
       │ tmin, tmax, daylength, vpd, swa,        │
       │ radiation, snow, sin(doy), cos(doy),    │  N channels
       │ mat, map, PFT_onehot                 │
       └─────────────────────────────────────────┘
                         │
                    RTnn Model
              (LSTM / GRU / FCN / Transformer)
              output: (batch, 1, 365)
                         │
                  SingleDayWrapper
              takes last timestep [:,:,-1]
                         │
                    (batch, 1)
                         │
                  MSE loss vs gcc_lowess
```

## Key design decisions

- **log1p transform** on snow, vpd, swa before z-scoring (heavy right skew)
- **Cyclic encoding** of day-of-year: sin + cos channels
- **Static features** (mat, map) broadcast across all 365 timesteps
- **PFT one-hot** also broadcast, so the model learns PFT-specific seasonality
- **Normalization stats** computed across all training sites, saved as JSON
- **Stride** parameter: use stride=1 for validation, stride=7 for training
  to reduce sample count and memory without losing seasonal coverage
