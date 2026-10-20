# Data generation

Scripts that generate the `pixelset/` dataset: a site-indexed collection of
daily meteorological forcing, leaf area index (LAI), and plant functional type
(PFT) composition for **1992–2019**, built for training the PhenoNN.

---

## Coordinate contract

The `selected_pixels` file defines the set of sites. Every subsequent product —
ERA5 meteo, dekadal and daily LAI, PFT fractions, and normalisation statistics —
is indexed by the **same `site_id`** and the **same native 0.05° grid coordinates**
(`lat_idx`, `lon_idx`, `latitude`, `longitude`). This guarantees that a site read
from any file refers to the same geographic cell, and that scattered per-site
reads never require re-extraction or re-gridding.

---

## Two resolutions

| Data | Native resolution | Grid shape | Longitude convention | Role |
|---|---|---|---|---|
| **LAI, PFT** | **0.05°** (~5.5 km) | 3600 × 7200 | −180° … 180° | Site grid (identity, targets) |
| **ERA5-Land meteo** | **0.1°** (~11 km) | 1801 × 3600 | 0° … 360° | Forcing (features) |

One 0.1° cell contains exactly four 0.05° cells. The site grid is the **0.05°**
LAI/PFT grid; each 0.05° site inherits the meteo of its containing 0.1° ERA5
cell, so four consecutive 0.05° sites share identical ERA5 values while each
keeps its own LAI and PFT.

---

## The chain

```text
CO2_1700_2023_TRENDYv2024.txt (external input)
│
valid_pixels_{YYYY}.nc (per-year 0.05° validity masks, from compute_valid_pixels.py)
│
▼
select_pixels.py  ──►  all_pixels_selected.nc         (--fraction 1.0, all land pixels)
│                      selected_pixels005_1.nc        (--fraction ~0.1, valid subset)
│                      selected_pixels01_1.nc         (0.1° parent map)
│
▼
rebalance_pixels.py     (PFT-dominance route)     ──► selected_pixels005_balanced.nc
  or
rebalance_spatial.py    (10° block-density route) ──► selected_pixels005_balanced.nc
                                                    + rebalance_blocks.csv
│
▼
build_daily_dataset_pixelset.py
  ERA5-Land hourly ──► era5_pixelset/ERA5_daily_pixelset_Y.nc
│
▼
build_pixelset_targets.py
  full LAI grid    ──► LAI_pixelset/LAI_dekadal_Y.nc
  full PFT grid    ──► PFT_pixelset/PFTmap_Y.nc
│
▼
interpolate_lai_daily_linear.py
  LAI_pixelset     ──► LAI_daily_linear/LAI_daily_Y.nc
│
▼
(optional) add_daylength.py / add_pheno_thermal.py
  era5_pixelset    ──► era5_pixelset_dl/ or era5_pheno/
│
▼
create_stats_file.py
  era5 + LAI + PFT + CO2 ──► norm_stats_1992_2019.json
│
▼
(training pipeline consumes era5_pixelset/ + LAI_daily_linear/ +
 norm_stats.json + CO2 LUT; PFT_pixelset/ for PFT conditioning)
```

---

## Step 0 — The one external input

**`CO2_1700_2023_TRENDYv2024.txt`**

- An external CO₂ forcing table in `Annee_YYYY=VALUE` format.
- Read **once**, at Step 6, by `create_stats_file.py --co2_path`.

---

## Step 1 — Build the validity masks

**Script:** `compute_valid_pixels.py`

**Inputs:**
- Full-grid `LAI_dekadal_{Y}.nc`
- Full-grid `PFTmap_{Y}.nc`

**What it does:** for each year, evaluates the LAI/PFT validity criteria cell
by cell on the native 0.05° grid and writes a boolean mask. The masks are the
raw material from which the site set is sampled.

**Output:** `valid_pixels_{YYYY}.nc` — per-year 0.05° boolean masks.

> This script is **only needed if you want to rebuild the validity masks** from
> the raw LAI/PFT grids. If you already have the selection files (Step 2), you
> can skip it.

---

## Step 2 — Sample the site set

**Script:** `select_pixels.py`

**Inputs:**
- A directory of `valid_pixels_{YYYY}.nc` (from Step 1)

**What it does:**
1. Opens every `valid_pixels_{YYYY}.nc` and counts, per pixel, how many years
   it was valid.
2. Keeps pixels with `counter >= --min_years` (default `1` = union of all years;
   pass `N` = number of files for the intersection).
3. Samples `--fraction` of the qualifying pixels **without replacement**, with
   a seeded RNG (`--seed`, default `42`).
4. Writes a single selection file with `site_id = "pix_LLLL_OOOOO"`
   (4-digit latitude index, 5-digit longitude index).

**Outputs (depending on the invocation):**

| File | Invocation | Meaning |
|---|---|---|
| `all_pixels_selected.nc` | `--fraction 1.0 --min_years 1` | **All** land pixels valid ≥1 year — the reference distribution |
| `selected_pixels005_1.nc` | `--fraction ~0.1 --min_years 1` on the 0.05° masks | 0.05° valid subset (what everything downstream uses) |
| `selected_pixels01_1.nc` | `--fraction ~0.1` on the 0.1° masks (or via `make_selected_pixels_01.py`) | 0.1° parent map (deduplicated ERA5 route) |

**Output schema:** `site = N_selected`; `lat_idx` (int32), `lon_idx` (int32),
`latitude` (float32), `longitude` (float32), `site_id` (str).
Attrs: `fraction`, `seed`, `min_years`, `n_qualifying`, `n_total`, `n_selected`,
`source`, `valid_files`.

---

## Step 3 — Rebalance the site set

Two rebalancing routes exist. They produce the **same output filename**
(`selected_pixels005_balanced.nc`) but implement **different corrections**. Only
one of them was used for a given dataset; the presence of `rebalance_blocks.csv`
indicates the spatial route.

### Step 3a — PFT-dominance rebalancing

**Script:** `rebalance_pixels.py`

**Inputs:**
- `selected_pixels005_1.nc` (the 0.05° valid subset)
- `PFT_pixelset/PFTmap_2010.nc` (one PFT year; PFT is ~static)

**What it does:**
1. For each site, compute the dominant PFT = argmax of the 15 PFT fractions
   from the single PFT year.
2. Count sites per dominant PFT (the "before" histogram). Tropical broadleaf
   evergreen (PFT 2) is over-represented: ~34 % of pure cells versus ~14 % over
   all land.
3. Decide how many reducible sites (those whose dominant PFT is in
   `--reduce_pft`, default PFT 2) to keep, from either:
   - `--target_total N` → keep roughly N sites in total, or
   - `--target_share 0.15` → reduce PFT 2 until it is 15 % of the pool.
4. Draw a **uniform random subset without replacement** of the reducible sites
   (`--seed 42`, so the draw is exactly reproducible). Non-reducible sites are
   always kept.
5. Write the reduced list.

**Output:** `selected_pixels005_balanced.nc` — same schema as the input, just
fewer rows.

**What is *not* rebuilt:** the underlying ERA5, LAI, and PFT pixel sets. They
were built on the full valid set; the balanced set is a subset of it, so
training just loads fewer sites from the same files. The unreduced 0.1° parent
map remains a superset and can still be used as `--parent_map` during training.

### Step 3b — Spatial (10° block-density) rebalancing

**Script:** `rebalance_spatial.py`

**Inputs:**
- `all_pixels_selected.nc` (the reference distribution — all land pixels)
- `selected_pixels005_1.nc` (the target — valid subset)
- `--block_deg 10` (block size in degrees)

**What it does:** matches the per-block density of the valid subset to the
per-block density of the all-land reference, so the kept sites are not
concentrated in a few 10° blocks. Draws a seeded random subset within each
block to hit `--target_total`.

**Outputs:**
- `selected_pixels005_balanced.nc`
- `rebalance_blocks.csv` — per-block report (before/after counts)

> **Which route was used?** Check for `rebalance_blocks.csv` next to
> `selected_pixels005_balanced.nc`. If it exists → spatial route (3b). If not
> → PFT route (3a).

> **Start point for the data build is therefore:** the selection files from
> Step 2 (or the balanced variant from Step 3) + the raw ERA5-Land hourly files
> + the full-grid LAI and PFT files. The `selected_pixels*.nc` files are the
> coordinate contract.

---

## Step 4 — ERA5 daily pixelset (meteo features)

**Script:** `build_daily_dataset_pixelset.py`

**Inputs:**
- `/bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF` — raw ERA5-Land hourly
  NetCDFs (`t2m, d2m, sp, ssrd, strd, tp`)
- `selected_pixels005_*.nc` (the 0.05° selection)

**What it does:**
1. Maps each 0.05° site to its containing 0.1° ERA5 cell:
   - `lat_era5 = (row05 + 1) // 2`
   - `lon_era5 = ((col05 + 1) // 2) % 3600`, then `+1800` roll
     (−180..180 → 0..360 fix).
2. Opens each hourly variable year, extracts only the selected pixels with
   `isel(latitude=…, longitude=…)` → `(time, site)`.
3. Resamples hourly → daily (`min/max/mean` for temperature; last-step-of-day
   for the cumulative `ssrd/strd/tp`).
4. Derives `VPD_max`, `VPD_mean`, `Rn_mean`, `Rn_tot`, `PET`
   (Tetens + Priestley–Taylor).
5. Drops intermediates (`Tdew_mean`, `sp_mean`, `Rn_mean`).
6. Overwrites `site_id/latitude/longitude/lat_idx/lon_idx` with the **native
   0.05°** values from `selected_pixels`.

**Output:** `era5_pixelset/ERA5_daily_pixelset_{1992…2019}.nc`

Schema: `time=365, site=N`; vars `Tmin, Tmax, Tmean, ssrd_sum, strd_sum,
tp_sum, VPD_max, VPD_mean, Rn_tot, PET`.

> `build_daily_dataset_era5idx.py` is the **sibling** that writes
> `era5_pixelset_01` (0.1° deduplicated cells, raw ERA5 indices, no roll). It
> is **not** the writer of `era5_pixelset/` — that dir comes from
> `build_daily_dataset_pixelset.py` (0.05° sites).

---

## Step 5 — LAI and PFT pixelsets (targets)

**Script:** `build_pixelset_targets.py`

**Inputs:**
- Full-grid `LAI_dekadal_{Y}.nc` (3600×7200) — `--target_dir`
- Full-grid `PFTmap_{Y}.nc` — `--pft_dir`
- `selected_pixels*.nc` (`lat_idx`, `lon_idx`, `site_id`, `latitude`, `longitude`)

**What it does:** one sequential decompression pass of each full grid, then a
NumPy fancy-index `[lat_idx, lon_idx]` → `(dekad, site)` for LAI and
`(pft, site)` for PFT. No scattered `isel` (that thrashes the chunk cache over
NFS). Writes the site coords onto the output.

**Outputs:**
- `LAI_pixelset/LAI_dekadal_{1992…2019}.nc` — `LAI(dekad=36, site)`
- `PFT_pixelset/PFTmap_{1992…2019}.nc` — `pft_frac(pft=15, site)`

---

## Step 6 — Daily LAI by linear interpolation

**Script:** `interpolate_lai_daily_linear.py`

**Input:** `LAI_pixelset/LAI_dekadal_{Y}.nc`

**What it does:** computes the 36 dekad day-of-year anchors (days 5/15/25,
leap-safe), then for each site linearly interpolates to 365 daily values with
`np.interp`. Needs ≥2 finite dekads; otherwise NaN. Edge values are held
constant (no extrapolation).

**Output:** `LAI_daily_linear/LAI_daily_{1992…2019}.nc` — `LAI(day=365, site)`,
carrying the same site coords.

---

## Step 7 — (Optional) extra derived features

Two augmentation scripts exist; **neither output is in `pixelset/`**, but they
are the documented next steps if `add_pheno_features` / `daylength` are enabled
in config:

- `add_daylength.py` — reads `era5_pixelset/ERA5_daily_pixelset_{Y}.nc`, adds
  `daylength(time, site)` from latitude × day-of-year (FAO-56 declination).
  Output → `era5_pixelset_dl/`.
- `add_pheno_thermal.py` — reads `era5_pixelset/`, adds 10 ORCHIDEE thermal
  phenology features (`gdd_0/5/10, ncd_temp, ncd_bor, ngd, botta_threshold,
  botta_forcing, t_rising, t_falling`), hemisphere-aware annual reset, prepends
  Y−1 for the SH mid-year reset. Output → `era5_pheno/`.

Both write to **new** directories; `era5_pixelset/` is the un-augmented Step-4
output.

---

## Step 8 — Normalisation statistics

**Script:** `create_stats_file.py`

**Inputs:**
- `era5_pixelset/ERA5_daily_pixelset_{Y}.nc` — `--features_dir`
- `LAI_pixelset/LAI_dekadal_{Y}.nc` — `--target_dir`
- `PFT_pixelset/PFTmap_{Y}.nc` — `--pft_dir`
- `CO2_1700_2023_TRENDYv2024.txt` — `--co2_path`

**What it does:** scans every year, accumulating online mean/std
(`RunningStats`) for:
- `DYNAMIC_FEATURES` (METEO [+ DERIVED]) — with `log1p` applied first for
  `LOG_TRANSFORM_FEATURES`
- `PFT_COLS` (`pft1_frac..pft15_frac`, pooled over years)
- `LAI` (target)
- `co2` (from the TRENDY LUT, per year in range)

Optional `--n_sites` subsamples sites per file for speed. Output is
byte-identical to the legacy format.

**Output:** `norm_stats_1992_2019.json` — `{feat: {mean, std}}` for every
channel the trainer z-scores.

---

## Step 9 — Diagnostics (not part of the data build)

**Script:** `diagnose_selected_pixels.py` (and `_bis`, `_pft`)

**Inputs:** `selected_pixels*.nc`, `PFT_pixelset/PFTmap_{1992…2019}.nc`

**What it does:** averages PFT fractions over the year range (joined by
`site_id`), renders world maps (dominant PFT, pure cells above threshold,
density per 10° block) and a purity table. Uses one throwaway subprocess per
`.nc` to avoid HDF5 heap corruption on `.venv_ERA5`.

**Outputs (in `analysis/`):**
- `selected_pixels_diag_dominant_pft.png`
- `selected_pixels_diag_pure_gt60.png`
- `selected_pixels_diag_purity_table.{csv,png}`
- `selected_pixels_diag_density_10deg.png`

`pft_diagnostics.py` is a *different* diagnostic: it reads `LAI_pixelset` +
`era5_pixelset` for a single-PFT subset (`selected_pixels_PFT{k}.nc`) and
writes `lai_seasonal.png`, `features_hist.png`, `features_dist_by_year.png`,
`features_seasonal.png`. **These filenames are not in `analysis/`** — so
`analysis/` was produced by `diagnose_selected_pixels*.py`, not
`pft_diagnostics.py`.

---

## The scripts — role in this chain

| File | Step | Produces |
|---|---|---|
| `compute_valid_pixels.py` | 1 | `valid_pixels_{YYYY}.nc` (per-year masks) |
| `select_pixels.py` | 2 | `all_pixels_selected.nc`, `selected_pixels005_1.nc`, `selected_pixels01_1.nc` |
| `rebalance_pixels.py` | 3a | `selected_pixels005_balanced.nc` (PFT route) |
| `rebalance_spatial.py` | 3b | `selected_pixels005_balanced.nc` + `rebalance_blocks.csv` (spatial route) |
| `make_selected_pixels_01.py` | 2 (wrapper) | `selected_pixels01_1.nc` (0.1° parent map), if used as a separate script |
| `build_daily_dataset_pixelset.py` | 4 | `era5_pixelset/ERA5_daily_pixelset_Y.nc` |
| `build_pixelset_targets.py` | 5 | `LAI_pixelset/LAI_dekadal_Y.nc`, `PFT_pixelset/PFTmap_Y.nc` |
| `interpolate_lai_daily_linear.py` | 6 | `LAI_daily_linear/LAI_daily_Y.nc` |
| `add_daylength.py` | 7 (optional) | `era5_pixelset_dl/` |
| `add_pheno_thermal.py` | 7 (optional) | `era5_pheno/` |
| `create_stats_file.py` | 8 | `norm_stats_1992_2019.json` |
| `diagnose_selected_pixels.py` / `_bis.py` | 9 | `analysis/selected_pixels_diag_*` |
| `pft_diagnostics.py` | 9 (separate) | `lai_seasonal.png`, `features_*.png` (not in `analysis/`) |
| `build_daily_dataset_era5idx.py` | sibling of 4 | `era5_pixelset_01/` (0.1° deduplicated route) |

---

## Reproducibility checklist

To rebuild `pixelset/` from scratch you need:

- [ ] The raw LAI and PFT full grids (`LAI_dekadal_{Y}.nc`, `PFTmap_{Y}.nc`)
- [ ] The raw ERA5-Land hourly archive
      (`/bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF`)
- [ ] `CO2_1700_2023_TRENDYv2024.txt`
- [ ] The scripts listed in the table above

Run order:

```
Step 1  compute_valid_pixels.py       → valid_pixels_{YYYY}.nc
Step 2  select_pixels.py              → all_pixels_selected.nc
                                        selected_pixels005_1.nc
                                        selected_pixels01_1.nc
Step 3  rebalance_pixels.py  (or)
        rebalance_spatial.py          → selected_pixels005_balanced.nc
Step 4  build_daily_dataset_pixelset.py   ─┐
Step 5  build_pixelset_targets.py         ─┤ (4 and 5 can run in parallel)
Step 6  interpolate_lai_daily_linear.py   ← depends on Step 5
Step 7  add_daylength.py / add_pheno_thermal.py   (optional)
Step 8  create_stats_file.py              ← depends on 4, 5, and CO2
Step 9  diagnose_selected_pixels.py       (diagnostic only)
```

If you already have the selection files from Step 2 (and optionally the
balanced variant from Step 3), you can start at Step 4.
