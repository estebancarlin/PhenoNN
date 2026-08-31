# Paramètres de prédiction — `phenonn.prediction.predict`

Point d'entrée : `python -m phenonn.prediction.predict --checkpoint <best_model.pth> [options]`

Charge un checkpoint, reconstruit le modèle, lance l'inférence sur les sites ×
années demandés, reconvertit en LAI physique (gère normalisation + mode
anomalie) et écrit un CSV + un fichier de métriques + des figures.

> Variante **0.1° natif** : features et cibles partagent la grille 0.1° et le
> même `site_id` (pas de `--parent_map` ici — c'est propre au paquet 0.05°
> `LaiNN_final`). Baselines associées : `xgb_predict`, `pure_pft_greedy_predict`
> (voir leur `--help`).

> **La plupart des réglages sont restaurés depuis le checkpoint** : `norm_stats`,
> `co2_lut`, `anomaly_mode`/`anomaly_clim`, `pft_mixing`, `normalize_lai`,
> l'architecture et ses hyperparamètres, ainsi que `features_dir/target_dir/
> pft_dir` et `val_years` s'ils y ont été stockés. Tu n'as donc en général qu'à
> donner `--checkpoint` (+ éventuellement les dossiers si tu déplaces les données).

---

## 1. Entrée

| Option | Défaut | Rôle |
|---|---|---|
| `--checkpoint` | **requis** | `best_model.pth` (ou tout snapshot) produit par l'entraînement. |
| `--features_dir` | ← checkpoint | Dossier `ERA5_daily_pixelset_{Y}.nc`. Écrase la valeur du checkpoint si donné. |
| `--target_dir` | ← checkpoint | Dossier `LAI_dekadal_{Y}.nc`. |
| `--pft_dir` | ← checkpoint | Dossier `PFTmap_{Y}.nc`. |

## 2. Sélection des sites

Ordre de priorité : `--selected_pixels` > `--sites` > `--predict_sites`.

| Option | Défaut | Rôle |
|---|---|---|
| `--selected_pixels` | `""` | `selected_pixels*.nc` : prédire uniquement sur ses sites. Prioritaire. |
| `--selection_split` | `""` | Avec `--selected_pixels`, sélectionne `train`, `validation`, `test` ou `buffer` selon les labels de split du fichier. |
| `--sites` | `""` | Liste explicite d'IDs séparés par des virgules. Prioritaire sur `--predict_sites`. |
| `--predict_sites` | `val` | Mode automatique depuis le checkpoint : `val` (val_site_ids), `train` (train_site_ids), `all` (union), `grid` (tous les pixels de la bbox), `test` (grid \ val_site_ids, pour l'overlap). |
| `--n_predict_sites` | `0` | Si > 0, sous-échantillonne aléatoirement ce nb de sites (graine `--seed`). |
| `--row_min`/`--row_max`/`--col_min`/`--col_max` | `-1` | Bbox pour les modes `grid`/`test`. `-1` → reprend la bbox des args d'entraînement. |

## 3. Années

| Option | Défaut | Rôle |
|---|---|---|
| `--predict_years` | `""` | `2015-2018`, `2015,2016` ou `all`. Vide → `val_years` du checkpoint. |

## 4. Exécution

| Option | Défaut | Rôle |
|---|---|---|
| `--batch_size` | `64` | Taille de batch d'inférence. |
| `--seed` | `42` | Graine (sous-échantillonnage sites / courbes). |
| `--output_csv` | `predictions.csv` | Chemin du CSV de sortie (les autres fichiers en dérivent : même préfixe). |

## 5. Figures

| Option | Défaut | Rôle |
|---|---|---|
| `--scatter_years` | off | Un scatter pred-vs-obs **par année**, dans un sous-dossier `scatter_year/`. |
| `--n_curves` | `0` | Nb de courbes par site dans `*_lai_curves_all.png`. `0` = toutes ; sinon sous-ensemble aléatoire. |
| `--pft_min_frac` | `0.05` | Seuil : n'affiche au-dessus de chaque sous-courbe que les PFT de fraction ≥ ce seuil. |

---

## Sorties produites

À partir de `--output_csv <base>.csv` :
- **`<base>.csv`** : colonnes `site_id, year, month, day, doy, lai_pred, lai_obs,
  lai_pred_norm, lai_obs_norm, error`.
- **`<base>_metrics.txt`** : R² **global** (NSE, inclut les écarts de moyenne
  inter-sites), R² **centré** (NSE sur la dynamique intra-site — test honnête),
  **Pearson²** (corrélation, ignore biais/échelle, comparable aux papiers),
  RMSE, et la **distribution du R² par site** (médiane, moyenne, p5, p95, % sites R²>0).
- **`<base>_pred_vs_obs.png`** (+ `_norm`/`_anomaly` selon le mode).
- **`<base>_lai_curves.png`** et **`<base>_lai_curves_all.png`** (courbes par site,
  PFT annotés).
- **`scatter_year/`** si `--scatter_years`.

## Exemple

```bash
python -m phenonn.prediction.predict \
  --checkpoint runs/exp_big/checkpoints/best_model.pth \
  --features_dir <era5_dir> --target_dir <LAI_dir> --pft_dir <PFT_dir> \
  --predict_sites val --predict_years 2010-2018 \
  --output_csv runs/exp_big/predictions.csv --scatter_years
```

> Liste complète : `python -m phenonn.prediction.predict --help`.
