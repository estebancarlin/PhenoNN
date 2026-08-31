# Paramètres d'entraînement — `phenonn.training.train_full_ram`

Point d'entrée : `python -m phenonn.training.train_full_ram [options]`
(`train_full_ram` est autonome : il charge en RAM une fois le *working-set* des
sites que la boucle échantillonnera, puis fait 0 I/O par epoch).

Un run écrit dans `<output_dir>/<experiment>/` : `checkpoints/{last,best}_model.pth`,
`logs/train.log`, `loss_history.png`, `metric_history.png`, `config.json`.

---

## 1. Données (obligatoires)

| Option | Défaut | Rôle |
|---|---|---|
| `--features_dir` | `""` | Dossier des `ERA5_daily_pixelset_{Y}.nc` (features quotidiennes par site). |
| `--target_dir` | `""` | Dossier des `LAI_dekadal_{Y}.nc` (cible LAI(dekad, site)). |
| `--pft_dir` | `""` | Dossier des `PFTmap_{Y}.nc` (fractions PFT(pft, site)). |
| `--train_years` | `""` | Années d'entraînement. Format `1992-2009` ou `1992,1994,1996`. |
| `--val_years` | `""` | Années de validation, même format. |

## 2. Sélection du pool de sites

Deux sources possibles ; **fournis l'une des deux** :

| Option | Défaut | Rôle |
|---|---|---|
| `--selected_pixels` | `""` | Chemin d'un `selected_pixels.nc` (sous-échantillon pré-tiré, p.ex. 10 % des pixels valides). **Recommandé.** Prioritaire sur `--valid_dir`. |
| `--valid_dir` | `""` | Dossier des `valid_pixels_{Y}.nc` : le pool = union des masques valides sur toutes les années (borné par la bbox row/col si donnée). |

**Mode DISJOINT vs OVERLAP** (via `--val_fraction_of_grid`, voir §4) :
- DISJOINT : le pool est coupé en train/val (pixels différents).
- OVERLAP (`--val_fraction_of_grid 100`) : train et val tirés du **même** pool ;
  la séparation se fait alors par les **années** (`--train_years`/`--val_years`
  doivent être disjointes).

## 3. Normalisation, CO₂, paradigme 0.05°

| Option | Défaut | Rôle |
|---|---|---|
| `--stats_path` | `""` | `norm_stats.json` (log1p sur features log + z-scoring). Sans lui : échelles brutes. |
| `--co2_path` | `""` | LUT CO₂ (`Annee_YYYY=VALEUR` par ligne), diffusé annuellement. Si `--stats_path` est donné, il doit contenir une entrée `co2`. |
| `--no_normalize_lai` | (norm ON) | Désactive le z-scoring **de la cible LAI** (les features restent normalisées). |
| `--parent_map` | `""` | `selected_pixels_01.nc` (via `make_selected_pixels_01`). Fait lire les features depuis l'ERA5 **dédupliqué 0.1°** (`site_id 'E{lat}_{lon}'`) via la cellule parente de chaque site 0.05°. LAI/PFT restent en 0.05°. **Chemin RAM uniquement.** Absent → features cherchées sur le pool de sites lui-même. |
| `--threaded_feature_read` | `False` | EXPÉRIMENTAL : lecture concurrente des fichiers annuels (ThreadPoolExecutor). Plus rapide sur NFS, mais **dangereux** si netCDF4/h5py embarquent un libhdf5 non thread-safe (segfault). Défaut = séquentiel. |

## 4. Échantillonnage par epoch

| Option | Défaut | Rôle |
|---|---|---|
| `--n_sites_per_epoch` | `500` | Nb de sites tirés aléatoirement à chaque epoch. |
| `--n_years_per_epoch` | `3` | Nb d'années tirées à chaque epoch. |
| `--n_val_sites` | `200` | Taille du set de validation fixe (tiré une fois du val_pool). |
| `--val_fraction_of_grid` | `0.1` | Fraction du pool réservée à la validation (mode DISJOINT). **Sentinelle `100` → mode OVERLAP.** |

## 5. Sous-grille (borne le pool `--valid_dir`)

| Option | Défaut | Rôle |
|---|---|---|
| `--row_min` / `--row_max` | `0` / `-1` | Plage d'indices latitude (inclusive). `-1` = toute la grille. |
| `--col_min` / `--col_max` | `0` / `-1` | Plage d'indices longitude. |

## 6. Mode anomalie (optionnel)

| Option | Défaut | Rôle |
|---|---|---|
| `--anomaly_mode` | off | Le modèle prédit l'**anomalie** de LAI vs climatologie (au lieu du LAI absolu). |
| `--clim_years` | `1992-2010` | Années servant à calculer la climatologie par site. |
| `--clim_target_dir` | `""` | Dossier LAI source de la climatologie (défaut : `--target_dir`). |

## 7. Modèle

| Option | Défaut | Rôle |
|---|---|---|
| `--type` | `lstm` | Architecture : `lstm`, `gru`, `transformer`, `transformer_dec`, `bitransformer`, `bitransformer_v2`, `attnlstm`, `aelstm`, `fcn`, `linear`, `linear_perday`. |
| `--pft_mixing` | off | Enveloppe `PFTMixingWrapper` : le modèle sort 15 LAI « purs » (un par PFT), sommés par les fractions PFT. |
| `--pft_meteo_only` | off | Avec `--pft_mixing` : le modèle de base ne voit **que** les canaux météo/cyclique/CO₂ (pas les fractions PFT). Change le nb de canaux d'entrée → **non reprenable** depuis un checkpoint non-meteo_only. |
| `--pft_nonneg` | off | Avec `--pft_mixing` : plancher souple à 0 (physique) de chaque LAI pur par PFT. Change le modèle → réentraînement requis. |
| `--seq_length` | `720` | Longueur de la fenêtre d'entrée (jours). |

**Hyperparamètres par architecture** (ignorés par les modèles qui ne les lisent pas) :

| Option | Défaut | Modèles concernés |
|---|---|---|
| `--hidden_size` | `128` | tous (LSTM/aelstm hidden ; bitransformer_v2 stage-2 ; attnlstm LSTM) |
| `--d_model` | `hidden_size` | bitransformer_v2, attnlstm (dim d'embedding stage-1) |
| `--num_layers` | `2` | tous (bitransformer_v2 stage-2 ; attnlstm LSTM) |
| `--num_layers1` | `2` | bitransformer_v2, attnlstm (blocs transformer stage-1) |
| `--nhead` | `4` | aelstm, bitransformer_v2, attnlstm |
| `--stress_dim` | `8` | bitransformer_v2, attnlstm (dim du bottleneck de stress) |
| `--dropout1` | `0.0` | bitransformer_v2, attnlstm (stage-1) |
| `--dropout2` | `0.0` | aelstm, bitransformer_v2 (stage-2), attnlstm (LSTM inter-couches, num_layers≥2) |
| `--dropout_att` | `0.0` | aelstm |
| `--forward_expansion` | `4` | aelstm |
| `--feed_forward_trans1` | `4` | bitransformer_v2/attnlstm stage-1 (multiplicateur FFN) |
| `--feed_forward_trans2` | `4` | bitransformer_v2 stage-2 |
| `--n_attn_blocks` | `2` | aelstm |
| `--d_layers` | `2` | (aucun de lstm/aelstm/bitransformer_v2/attnlstm) |
| `--embed_size` | `64` | (idem) |

## 8. Optimisation & perte

| Option | Défaut | Rôle |
|---|---|---|
| `--batch_size` | `32` | Taille de batch. |
| `--num_epochs` | `30` | Nb max d'epochs. |
| `--learning_rate` | `1e-3` | Pas d'apprentissage (Adam). |
| `--weight_decay` | `1e-5` | Régularisation L2. |
| `--loss_type` | `huber` | `mse`, `mae`, `huber`, `smoothl1`, `nmse`, `nmae`, `gradient`. |
| `--huber_beta` | `1.0` | Seuil δ de la Huber. |
| `--gradient_loss_weight` | `0.5` | Poids du terme gradient (loss `gradient`). |
| `--peak_penalty_weight` | `0.0` | Pénalité supplémentaire sur les pics de LAI. |
| `--corr_loss_weight` | `0.0` | Poids du terme (1 − Pearson masquée) : forme/phase, invariant d'échelle, anti-amortissement. |
| `--amp_loss_weight` | `0.0` | Poids du terme `|std(pred) − std(cible)|` (amplitude). |
| `--max_grad_norm` | `1.0` | Clipping de la norme du gradient (0 = off). |
| `--patience` | `11` | Early-stopping : epochs sans amélioration val avant arrêt. |
| `--num_workers` | `4` | Workers du DataLoader (mets ≈ `--cpus-per-task`). |
| `--amp` | off | Autocast bf16 (mixed precision) : plus rapide / moins de mémoire GPU (Ampère+). |
| `--compile` | off | `torch.compile` (fusion de kernels). |
| `--no_fused_adam` | (fused ON) | Désactive le kernel Adam fusionné CUDA (activé par défaut sur GPU). |

## 9. Sortie & reprise

| Option | Défaut | Rôle |
|---|---|---|
| `--output_dir` | `runs_final` | Racine des runs. |
| `--experiment` | `exp_big` | Sous-dossier du run. |
| `--seed` | `42` | Graine (numpy + torch + échantillonnage). |
| `--resume` | `""` | Chemin d'un `last_model.pth` : reprend modèle/optim/scheduler/historique. Les args du checkpoint priment, sauf ceux passés explicitement en ligne de commande. |
| `--wandb` | off | Active le suivi Weights & Biases. Nécessite `python -m pip install -e ".[tracking]"`. |
| `--wandb_project` | `phenonn-lai` | Projet W&B. |
| `--wandb_entity` / `--wandb_group` | `""` | Entité et groupe W&B optionnels. |
| `--wandb_tags` | `""` | Tags séparés par des virgules. |
| `--wandb_mode` | `online` | `online`, `offline` ou `disabled`. |

---

## Exemple minimal

```bash
python -m phenonn.training.train_full_ram \
  --features_dir  <era5_pixelset>  --target_dir <LAI_pixelset>  --pft_dir <PFT_pixelset> \
  --selected_pixels <selected_pixels.nc> \
  --stats_path <norm_stats.json> --co2_path <co2.txt> \
  --parent_map <selected_pixels01_1.nc> \          # paradigme 0.05° (sinon omettre)
  --train_years 1992-2009 --val_years 2010-2018 --val_fraction_of_grid 100 \
  --type bitransformer_v2 --pft_mixing --pft_meteo_only \
  --num_epochs 30 --n_sites_per_epoch 1000 --n_val_sites 1000 --n_years_per_epoch 5 \
  --learning_rate 1e-4 --amp \
  --output_dir ./runs/ --experiment exp_big
```

> Toutes les options sont aussi listées par `python -m phenonn.training.train_full_ram --help`.
> (Paquet `LaiNN_final01` : baseline XGBoost via `phenonn.training.xgb_train --help`, jeu de paramètres distinct.)
