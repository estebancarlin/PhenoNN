# LAI Prediction Dataset
# Adapted from RTnn DataPreprocessor for daily phenocam time series.
#
# Input:  365-day window of meteorological features ending on day x
# Output: LAI on day x (single scalar)

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from phenocam.feature_engineering import add_derived_features


# ── Feature definitions ──────────────────────────────────────────────────────

DYNAMIC_FEATURES = [
    "tmin", "tmax", "daylength", "vpd", "prcp", "srad", "swe"
]

DERIVED_FEATURES = ["gdd_0", "gdd_5", "gdd_10", "cdd", "botta_threshold", "botta_forcing", "ncd"]
SITE_FEATURES = ["lat", "lon", "elev"]  # "lat", "lon", "elevation"
STATIC_FEATURES = ["mat", "map"]  # mean annual temp and precip
SOIL_FEATURES = ["clay", "sand", "silt", "ph"] 
SLOPE_FEATURES= ["slope"]  

add_site_features = False
add_GDD_features = True
add_soil_features = False
add_slope_features = False
add_cyclic_features = False

if add_site_features:
    STATIC_FEATURES += SITE_FEATURES
    
if add_GDD_features:
    DYNAMIC_FEATURES += DERIVED_FEATURES

if add_soil_features:
    STATIC_FEATURES += SOIL_FEATURES

if add_slope_features:
    STATIC_FEATURES += SLOPE_FEATURES

CYCLIC_FEATURES = []  
if add_cyclic_features:
    CYCLIC_FEATURES = ["doy_sin", "doy_cos"]

LOG_TRANSFORM_FEATURES = {"swe", "vpd", "prcp", "gdd_0", "gdd_5", "gdd_10", "cdd", "elev"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def extract_pft_and_site(filepath: str) -> Tuple[str, str]:
    """
    Extract PFT code and site name from a filename like 'DB_asuhighlands.csv'.

    Parameters
    ----------
    filepath : str
        Path to site CSV file.
    Returns
    -------
    tuple of (str, str)
        (pft_code, site_name), e.g. ('DB', 'asuhighlands').
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "UNK", basename


def load_site(filepath: str) -> pd.DataFrame:
    """
    Load and clean a single site CSV.

    Sorts by (year, doy), adds cyclic day-of-year features, and forward-fills small meteo gaps.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with all feature and target columns.
    """
    df = pd.read_csv(filepath)
    df = df.sort_values(["year", "doy"]).reset_index(drop=True)

    # Cyclic day-of-year encoding
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    df = add_derived_features(df)
    # Forward-fill small gaps in meteo (typically ≤1 day ) Not sure of why there still are missing meteo values -> look data_loader
    for col in DYNAMIC_FEATURES:
        df[col] = df[col].ffill().bfill()
        
    return df

def load_lai_norms(
    norms_csv: str,
    site_files: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load per-site LAI min, max (from external file).
    Enables inter-site normalization: lai_norm = (lai - min) / (max - min).
    """
    df = pd.read_csv(norms_csv)
    key_col = df.columns[0]

    lai_norms = {}
    for _, row in df.iterrows():
        site_key = row[key_col]
        entry = {"lai_min": float(row["lai_min"])}
        entry["lai_max"] = float(row["lai_max"])
        lai_norms[site_key] = entry

    return lai_norms


# ── Normalization stats ──────────────────────────────────────────────────────


def compute_norm_stats(
    site_files: List[str],
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-feature mean and std across all training sites.

    For features in LOG_TRANSFORM_FEATURES, stats are computed on log1p(x).
    For static features (mat, map), stats are computed per-site (one value
    per site) to avoid inflating variance with repeated rows.

    Parameters
    ----------
    site_files : list of str
        Paths to training site CSVs.
    save_path : str, optional
        If provided, save stats as JSON to this path.

    Returns
    -------
    dict
        {feature_name: {"mean": float, "std": float}} for all features.
    """
    # Accumulate dynamic features from all sites
    dynamic_values = {feat: [] for feat in DYNAMIC_FEATURES + CYCLIC_FEATURES}
    static_values = {feat: [] for feat in STATIC_FEATURES}
    target_values = []

    for f in site_files:
        df = load_site(f)

        for feat in DYNAMIC_FEATURES:
            vals = df[feat].dropna().values
            if feat in LOG_TRANSFORM_FEATURES:
                vals = np.log1p(np.clip(vals, 0, None))
            dynamic_values[feat].append(vals)

        for feat in CYCLIC_FEATURES:
            dynamic_values[feat].append(df[feat].values)

        # Static: take single value per site
        for feat in STATIC_FEATURES:
            val = df[feat].dropna()
            if len(val) > 0:
                static_values[feat].append(val.iloc[0])

        target_values.append(df["LAI"].dropna().values)

    stats = {}

    # Dynamic + cyclic features: global mean/std
    for feat in DYNAMIC_FEATURES + CYCLIC_FEATURES:
        all_vals = np.concatenate(dynamic_values[feat])
        stats[feat] = {"mean": float(np.mean(all_vals)), "std": float(np.std(all_vals))}
        # Avoid zero std
        if stats[feat]["std"] < 1e-8:
            stats[feat]["std"] = 1.0

    # Static features: cross-site mean/std
    for feat in STATIC_FEATURES:
        arr = np.array(static_values[feat])
        stats[feat] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        if stats[feat]["std"] < 1e-8:
            stats[feat]["std"] = 1.0

    # Target stats (for optional normalization)
    all_target = np.concatenate(target_values)
    stats["LAI"] = {
        "mean": float(np.mean(all_target)),
        "std": float(np.std(all_target)),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats


def load_norm_stats(path: str) -> Dict[str, Dict[str, float]]:
    """Load normalization stats from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────


class PhenoCamDataset(Dataset):
    """
    PyTorch dataset for LAI prediction.

    Each sample is a 365-day sliding window of meteorological and site features,
    with the target being LAI on the final day of the window.

    The feature tensor has shape (feature_channels, 365) where feature_channels =
    len(DYNAMIC_FEATURES) + len(CYCLIC_FEATURES) + len(STATIC_FEATURES) + n_pfts.

    Parameters
    ----------
    site_files : list of str
        Paths to site CSV files (e.g. ['DB_asuhighlands.csv', ...]).
    norm_stats : dict
        Normalization statistics from compute_norm_stats().
    seq_length : int
        Window length in days. Default 365.
    target_col : str
        Target column name. Default 'LAI'.
    normalize_target : bool
        Whether to z-score the target. Default False.
    pft_list : list of str, optional
        Ordered list of all PFT codes for one-hot encoding.
        If None, auto-discovered from site_files.
    stride : int
        Step between consecutive windows. Default 1 (every day).
    years : list of int, optional
        If provided, only use windows whose target day falls in these years.
        Use this to split "predict year by year".

    Examples
    --------
    > stats = compute_norm_stats(train_files)
    > train_ds = PhenoCamDataset(train_files, stats, pft_list=['DB', 'EN', 'GR'])
    > features, target = train_ds[0]
    > features.shape
    torch.Size([14, 365])   # 7 meteo + 2 cyclic + 2 static + 3 PFT
    > target.shape
    torch.Size([1])
    """

    def __init__(
        self,
        site_files: List[str],
        norm_stats: Dict[str, Dict[str, float]],
        seq_length: int = 365,
        target_col: str = "LAI",
        normalize_target: bool = True,
        pft_list: Optional[List[str]] = None,
        stride: int = 1,
        lai_norms=None,
        years: Optional[List[int]] = None,
        n_target_days: int = 1,
        residual_csv: Optional[str] = None,
        random_stride: int = 0,
        feature_mode: str = "all",
        full_year: bool = False,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.target_col = target_col
        self.norm_stats = norm_stats
        self.normalize_target = normalize_target
        self.stride = stride
        self.n_target_days = n_target_days
        self.residual_mode = residual_csv is not None
        self.random_stride = random_stride
        self.feature_mode = feature_mode
        self.full_year = full_year
        # For full_year mode: predict last 365 days from a 730-day window
        self.pred_length = 365 if full_year else 1

        # Load residuals from a previous prediction run
        self._residual_lookup: Dict[Tuple[str, int], float] = {}
        if residual_csv is not None:
            res_df = pd.read_csv(residual_csv)
            # residual = observed - predicted (what the first model missed)
            if "lai_obs_norm" in res_df.columns and "lai_pred_norm" in res_df.columns:
                for _, row in res_df.iterrows():
                    key = (str(row["site"]), int(row["day_index"]))
                    self._residual_lookup[key] = float(row["lai_obs_norm"] - row["lai_pred_norm"])
            else:
                for _, row in res_df.iterrows():
                    key = (str(row["site"]), int(row["day_index"]))
                    self._residual_lookup[key] = float(row["lai_obs"] - row["lai_pred"])

            # Debug: which sites are in the residual CSV
            residual_sites = set(str(row["site"]) for _, row in res_df.iterrows())
            print(f"[residual] Loaded {len(self._residual_lookup)} entries "
                  f"from {len(residual_sites)} sites: {sorted(residual_sites)}")

        # Discover PFTs
        if pft_list is None:
            pft_list = sorted(set(extract_pft_and_site(f)[0] for f in site_files))
        self.pft_list = pft_list
        self.pft_to_idx = {pft: i for i, pft in enumerate(pft_list)}
        self.n_pfts = len(pft_list)

        # Feature groups based on mode
        #   "all"       : dynamic + cyclic + static + PFT (default)
        #   "site_only" : cyclic + static + PFT (climatology model)
        #   "meteo_only": dynamic + cyclic (anomaly model)
        self.use_dynamic = feature_mode in ("all", "meteo_only")
        self.use_static = feature_mode in ("all", "site_only")
        self.use_pft = feature_mode in ("all", "site_only")

        # Feature channel count
        self.n_features = len(CYCLIC_FEATURES)
        if self.use_dynamic:
            self.n_features += len(DYNAMIC_FEATURES)
        if self.use_static:
            self.n_features += len(STATIC_FEATURES)
        if self.use_pft:
            self.n_features += self.n_pfts

        # Build all samples: list of (site_idx, target_day_position)
        self.samples: List[Tuple[int, int]] = []
        self.site_data: List[Dict] = []

        for file_idx, filepath in enumerate(site_files):
            pft_code, site_name = extract_pft_and_site(filepath)
            df = load_site(filepath)

            feature_parts = []

            # Dynamic features (meteo + derived)
            if self.use_dynamic:
                features_normed = np.zeros((len(df), len(DYNAMIC_FEATURES)), dtype=np.float32)
                for i, feat in enumerate(DYNAMIC_FEATURES):
                    vals = df[feat].values.copy()
                    if feat in LOG_TRANSFORM_FEATURES:
                        vals = np.log1p(np.clip(vals, 0, None))
                    features_normed[:, i] = (vals - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
                feature_parts.append(features_normed)

            # Cyclic features (always included)
            cyclic_normed = np.zeros((len(df), len(CYCLIC_FEATURES)), dtype=np.float32)
            for i, feat in enumerate(CYCLIC_FEATURES):
                vals = df[feat].values
                cyclic_normed[:, i] = (vals - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
            feature_parts.append(cyclic_normed)

            # Static features
            if self.use_static:
                static_normed = np.zeros((len(df), len(STATIC_FEATURES)), dtype=np.float32)
                for i, feat in enumerate(STATIC_FEATURES):
                    val = df[feat].dropna()
                    val = val.iloc[0] if len(val) > 0 else norm_stats[feat]["mean"]
                    static_normed[:, i] = (val - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
                feature_parts.append(static_normed)

            # PFT one-hot
            if self.use_pft:
                pft_onehot = np.zeros((len(df), self.n_pfts), dtype=np.float32)
                if pft_code in self.pft_to_idx:
                    pft_onehot[:, self.pft_to_idx[pft_code]] = 1.0
                feature_parts.append(pft_onehot)

            # Concatenate selected features: (n_days, n_features)
            all_features = np.concatenate(feature_parts, axis=1)

            # Target
            target_vals = df[target_col].values.astype(np.float32)
            site_key = f"{pft_code}_{site_name}"
            lai_min = None
            lai_max = None

            if lai_norms is not None and site_key in lai_norms:
                lai_min = lai_norms[site_key]["lai_min"]
                lai_max = lai_norms[site_key]["lai_max"]
                denom = lai_max - lai_min
                if denom > 1e-8:
                    target_vals = (target_vals - lai_min) / denom
            elif normalize_target:
                target_vals = (
                    (target_vals - norm_stats["LAI"]["mean"])
                    / norm_stats["LAI"]["std"]
                )

            # Year array for filtering
            year_arr = df["year"].values

            # Store preprocessed data for this site
            self.site_data.append(
                {
                    "features": all_features,  # (n_days, n_features)
                    "target": target_vals,      # (n_days,)
                    "years": year_arr,
                    "pft": pft_code,
                    "site": site_name,
                    "lai_min": lai_min,
                    "lai_max": lai_max,
                }
            )

            # If residual mode, replace targets with residuals from first model
            if self.residual_mode:
                residual_vals = np.full_like(target_vals, np.nan)
                n_matched = 0
                for d in range(len(target_vals)):
                    key = (site_name, d)
                    if key in self._residual_lookup:
                        residual_vals[d] = self._residual_lookup[key]
                        n_matched += 1
                self.site_data[-1]["target"] = residual_vals
                if n_matched == 0:
                    print(f"[residual] WARNING: site '{site_name}' — 0 matches "
                          f"(day_index range 0..{len(target_vals)-1}). "
                          f"Is this site in the predictions CSV?")
                else:
                    print(f"[residual] site '{site_name}': "
                          f"{n_matched}/{len(target_vals)} days matched")

            # Build sample indices: every valid window position
            n_days = len(df)
            target_check = self.site_data[-1]["target"]

            if self.full_year:
                # Full-year mode: one sample per year, positioned at the last day
                # of each year that has 730 days of features and 365 days of valid targets
                for yr in sorted(df["year"].unique()):
                    yr_mask = (year_arr == yr)
                    yr_indices = np.where(yr_mask)[0]
                    if len(yr_indices) == 0:
                        continue
                    d = yr_indices[-1]  # last day of this year
                    # Check enough history for the input window
                    if d < seq_length - 1:
                        continue
                    # Check 365 days of valid targets at the end
                    target_start = d - self.pred_length + 1
                    if target_start < 0:
                        continue
                    target_slice = target_check[target_start : d + 1]
                    if len(target_slice) < self.pred_length or np.any(np.isnan(target_slice)):
                        continue
                    if years is not None and yr not in years:
                        continue
                    self.samples.append((file_idx, d))
            else:
                # Standard mode: sliding window
                effective_stride = stride if random_stride == 0 else 1
                for d in range(seq_length - 1, n_days, effective_stride):
                    if np.isnan(target_check[d]):
                        continue
                    if n_target_days > 1:
                        t_start = d - n_target_days + 1
                        if t_start < seq_length - 1 or np.any(np.isnan(target_check[t_start:d+1])):
                            continue
                    if years is not None and year_arr[d] not in years:
                        continue
                    self.samples.append((file_idx, d))

        # For random_stride: group all valid samples by (site_idx, year),
        # then subsample N per group
        if self.random_stride > 0:
            self._samples_by_site_year: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            for (site_idx, d) in self.samples:
                yr = int(self.site_data[site_idx]["years"][d])
                key = (site_idx, yr)
                self._samples_by_site_year.setdefault(key, []).append((site_idx, d))
            self._all_valid_count = len(self.samples)
            self.resample()  # initial random selection
        else:
            self._samples_by_site_year = None

        # Final check
        if self.residual_mode and len(self.samples) == 0:
            dataset_sites = [extract_pft_and_site(f)[1] for f in site_files]
            residual_sites = set(k[0] for k in self._residual_lookup.keys())
            overlap = set(dataset_sites) & residual_sites
            raise RuntimeError(
                f"Residual mode: 0 valid samples! "
                f"Dataset sites ({len(dataset_sites)}): {sorted(dataset_sites)[:5]}... "
                f"Residual CSV sites ({len(residual_sites)}): {sorted(residual_sites)[:5]}... "
                f"Overlap: {len(overlap)} sites. "
                f"Did you run predict.py with --predict_sites all?"
            )

    @property
    def feature_channels(self) -> int:
        """Number of input feature channels."""
        return self.n_features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            features : shape (feature_channels, seq_length)
            target   : shape (1,) if n_target_days=1,
                        shape (1, n_target_days) if n_target_days>1
        """
        site_idx, day_idx = self.samples[index]
        site = self.site_data[site_idx]

        # Extract window: [day_idx - seq_length + 1, day_idx + 1)
        start = day_idx - self.seq_length + 1
        window = site["features"][start : day_idx + 1]  # (seq_length, n_features)

        # Transpose to (n_features, seq_length) to match RTnn convention
        features = torch.from_numpy(window.T.copy())

        if self.full_year:
            # Target: last pred_length (365) days of LAI in the window
            target_start = day_idx - self.pred_length + 1
            target_seq = site["target"][target_start : day_idx + 1]  # (365,)
            target = torch.from_numpy(target_seq.copy()).unsqueeze(0)  # (1, 365)
        elif self.n_target_days > 1:
            # Target: last n days for gradient loss, e.g. [LAI(t-1), LAI(t)]
            t_start = day_idx - self.n_target_days + 1
            target_days = site["target"][t_start : day_idx + 1]  # (n_target_days,)
            target = torch.from_numpy(target_days.copy()).unsqueeze(0)  # (1, n_target_days)
        else:
            # Target: scalar for the last day
            target = torch.tensor([site["target"][day_idx]], dtype=torch.float32)

        return features, target

    def resample(self):
        """
        Randomly select `random_stride` samples per site-year.

        Call this at the start of each training epoch for fresh random samples.
        No-op if random_stride was not set.
        """
        if self._samples_by_site_year is None:
            return

        rng = np.random.RandomState()
        new_samples = []
        for key, candidates in self._samples_by_site_year.items():
            n = min(self.random_stride, len(candidates))
            chosen = rng.choice(len(candidates), size=n, replace=False)
            for i in chosen:
                new_samples.append(candidates[i])

        self.samples = new_samples

    def get_site_info(self, index: int) -> Dict:
        """
        Get metadata for a sample (useful for evaluation / plotting).

        Returns dict with keys: site, pft, year, doy_index.
        """
        site_idx, day_idx = self.samples[index]
        site = self.site_data[site_idx]
        return {
            "site": site["site"],
            "pft": site["pft"],
            "year": int(site["years"][day_idx]),
            "day_index": day_idx,
            "lai_min": site["lai_min"],
            "lai_max": site["lai_max"],
        }


# ── Convenience: build train/val split by site ──────────────────────────────


def split_sites_by_fraction(
    site_files: List[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split site files into train and validation sets (leave-site-out).

    Parameters
    ----------
    site_files : list of str
        All site CSV paths.
    val_fraction : float
        Fraction of sites to hold out for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (list, list)
        (train_files, val_files)
    """
    rng = np.random.RandomState(seed)
    files = np.array(site_files)
    n_val = max(1, int(len(files) * val_fraction))
    indices = rng.permutation(len(files))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return files[train_idx].tolist(), files[val_idx].tolist()


def split_by_year(
    site_files: List[str],
    norm_stats: Dict,
    train_years: List[int],
    val_years: List[int],
    pft_list: List[str],
    **kwargs,
) -> Tuple["PhenoCamDataset", "PhenoCamDataset"]:
    """
    Create train and validation datasets split by target year.

    All sites are used for both sets, but training windows target
    train_years and validation windows target val_years.

    Parameters
    ----------
    site_files : list of str
        All site CSV paths.
    norm_stats : dict
        Normalization statistics.
    train_years : list of int
        Years to include in training targets.
    val_years : list of int
        Years to include in validation targets.
    pft_list : list of str
        Ordered PFT codes.
    **kwargs
        Additional arguments passed to PhenoCamDataset.

    Returns
    -------
    tuple of (PhenoCamDataset, PhenoCamDataset)
    """
    train_ds = PhenoCamDataset(
        site_files, norm_stats, pft_list=pft_list, years=train_years, **kwargs
    )
    val_ds = PhenoCamDataset(
        site_files, norm_stats, pft_list=pft_list, years=val_years, **kwargs
    )
    return train_ds, val_ds


def split_by_sites_years(
    site_files: List[str],
    norm_stats: Dict,
    train_years: List[int],
    val_years: List[int],
    pft_list: List[str],
    val_fraction: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Tuple["PhenoCamDataset", "PhenoCamDataset"]:
    """
    Create train and validation datasets split by both sites and years.

    Validation set is composed of all years for val_fraction of sites (randomly selected) and of the years in val_years for 
    the remaining sites. The training set is composed of the remaining years for the remaining sites.

    Parameters
    ----------
    site_files : list of str
        All site CSV paths.
    norm_stats : dict
        Normalization statistics.
    train_years : list of int
        Years to include in training targets.
    val_years : list of int
        Years to include in validation targets.
    pft_list : list of str
        Ordered PFT codes.
    val_fraction : float
        Fraction of sites to hold out for validation.
    seed : int
        Random seed for reproducibility.

    """
    return None, None  # Todo