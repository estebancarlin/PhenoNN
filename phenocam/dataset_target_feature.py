# LAI Prediction Dataset — reads from flat feature and target CSVs.
#
# Two flat CSV files replace the per-site CSVs used in dataset.py:
#   - features.csv  : one row per (site_id, date), 365 days/site/year
#                     columns: site_id, date (YYYYMMDD), year, month, day,
#                              pft1_frac..pft15_frac, tmin, tmax, daylength,
#                              prcp, srad, vpd, swe
#   - targets.csv   : one row per (site_id, date), 36 obs/site/year
#                     observation days: 5, 15, 25 of each month
#                     columns: site_id, date, year, month, day, LAI_raw, LAI
#
# Each sample covers one (site_id, year) pair:
#   features : (n_features, seq_length)    — seq_length trailing days of features
#   targets  : (1, 36)                     — 36 LAI observations for that year

import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from phenocam.feature_engineering import add_derived_features


# ── Feature definitions ───────────────────────────────────────────────────────

METEO_FEATURES = ["tmin", "tmax", "daylength", "vpd", "prcp", "srad", "swe"]
PFT_COLS = [f"pft{i}_frac" for i in range(1, 16)]
DERIVED_FEATURES = [
    "gdd_0", "gdd_5", "gdd_10", "cdd", "botta_threshold", "botta_forcing", "ncd"
]
DYNAMIC_FEATURES = METEO_FEATURES + DERIVED_FEATURES   # 14
CYCLIC_FEATURES = ["doy_sin", "doy_cos"]               # 2
ALL_FEATURES = DYNAMIC_FEATURES + CYCLIC_FEATURES + PFT_COLS  # 31

LOG_TRANSFORM_FEATURES = {"swe", "vpd", "prcp", "gdd_0", "gdd_5", "gdd_10", "cdd"}

TARGET_DAYS_OF_MONTH = [5, 15, 25]
N_OBS_PER_YEAR = 36   # 12 months × 3 days


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_site_ids(csv_path: str) -> List[str]:
    """Return sorted list of unique site_ids from a features or targets CSV."""
    df = pd.read_csv(csv_path, usecols=["site_id"])
    return sorted(df["site_id"].unique().tolist())


def split_sites_by_fraction(
    site_ids: List[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split site_ids into (train, val) sets (leave-site-out).

    Returns
    -------
    (train_ids, val_ids)
    """
    rng = np.random.RandomState(seed)
    arr = np.array(site_ids)
    n_val = max(1, int(len(arr) * val_fraction))
    idx = rng.permutation(len(arr))
    return arr[idx[n_val:]].tolist(), arr[idx[:n_val]].tolist()


def _load_features(
    features_csv: str,
    site_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load features CSV, add doy column, optionally filter to site_ids.
    Returns dataframe sorted by (site_id, date).
    """
    df = pd.read_csv(features_csv)
    if site_ids is not None:
        df = df[df["site_id"].isin(site_ids)].copy()
    else:
        df = df.copy()
    dates = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["doy"] = dates.dt.day_of_year.values
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)


def _load_targets(
    target_csv: str,
    site_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load targets CSV, optionally filtering to site_ids."""
    df = pd.read_csv(target_csv)
    if site_ids is not None:
        df = df[df["site_id"].isin(site_ids)].copy()
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)


def _preprocess_site(site_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic and derived ecological features to a single site's daily data.
    Must be called after doy is present and before normalization.
    """
    df = site_df.sort_values(["year", "doy"]).reset_index(drop=True)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df = add_derived_features(df)
    for col in METEO_FEATURES:
        df[col] = df[col].ffill().bfill()
    return df


def _build_feature_matrix(
    site_df: pd.DataFrame,
    norm_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    Normalize and assemble a (n_days, n_features) float32 array.
    Features are ordered: DYNAMIC_FEATURES + CYCLIC_FEATURES + PFT_COLS.
    """
    n = len(site_df)
    parts = []

    dyn = np.zeros((n, len(DYNAMIC_FEATURES)), dtype=np.float32)
    for i, feat in enumerate(DYNAMIC_FEATURES):
        vals = site_df[feat].values.copy().astype(np.float32)
        if feat in LOG_TRANSFORM_FEATURES:
            vals = np.log1p(np.clip(vals, 0, None))
        dyn[:, i] = (vals - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
    parts.append(dyn)

    cyc = np.zeros((n, len(CYCLIC_FEATURES)), dtype=np.float32)
    for i, feat in enumerate(CYCLIC_FEATURES):
        vals = site_df[feat].values.astype(np.float32)
        cyc[:, i] = (vals - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
    parts.append(cyc)

    pft = np.zeros((n, len(PFT_COLS)), dtype=np.float32)
    for i, feat in enumerate(PFT_COLS):
        vals = site_df[feat].values.astype(np.float32)
        pft[:, i] = (vals - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]
    parts.append(pft)

    return np.concatenate(parts, axis=1)


# ── Normalization stats ───────────────────────────────────────────────────────


def compute_norm_stats(
    features_csv: str,
    target_csv: str,
    train_site_ids: List[str],
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-feature mean and std from training sites only.

    Features in LOG_TRANSFORM_FEATURES are log1p-transformed before computing
    stats. PFT fraction columns are included.

    Parameters
    ----------
    features_csv : str
        Path to flat daily features CSV.
    target_csv : str
        Path to sparse LAI targets CSV.
    train_site_ids : list of str
        Site IDs belonging to the training split.
    save_path : str, optional
        If provided, save stats as JSON to this path.

    Returns
    -------
    dict
        {feature_name: {"mean": float, "std": float}} for all 31 features + "LAI".
    """
    feat_df = _load_features(features_csv, site_ids=train_site_ids)
    tgt_df = _load_targets(target_csv, site_ids=train_site_ids)

    processed_parts = []
    for _, grp in feat_df.groupby("site_id"):
        processed_parts.append(_preprocess_site(grp))
    all_df = pd.concat(processed_parts, ignore_index=True)

    stats: Dict[str, Dict[str, float]] = {}

    for feat in DYNAMIC_FEATURES:
        vals = all_df[feat].dropna().values.astype(np.float64)
        if feat in LOG_TRANSFORM_FEATURES:
            vals = np.log1p(np.clip(vals, 0, None))
        stats[feat] = {
            "mean": float(np.mean(vals)),
            "std": float(max(np.std(vals), 1e-8)),
        }

    for feat in CYCLIC_FEATURES:
        vals = all_df[feat].dropna().values.astype(np.float64)
        stats[feat] = {
            "mean": float(np.mean(vals)),
            "std": float(max(np.std(vals), 1e-8)),
        }

    for feat in PFT_COLS:
        vals = all_df[feat].dropna().values.astype(np.float64)
        stats[feat] = {
            "mean": float(np.mean(vals)),
            "std": float(max(np.std(vals), 1e-8)),
        }

    target_vals = tgt_df["LAI"].dropna().values.astype(np.float64)
    stats["LAI"] = {
        "mean": float(np.mean(target_vals)),
        "std": float(max(np.std(target_vals), 1e-8)),
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


# ── Dataset ───────────────────────────────────────────────────────────────────


class LAIDataset(Dataset):
    """
    PyTorch dataset for LAI prediction using flat feature and target CSVs.

    Each sample is one (site_id, target_year) pair. The feature window covers
    seq_length consecutive days ending on the last day of the target year.
    The target is the 36 LAI observations for that year (days 5, 15, 25 of
    every month).

    Feature tensor shape : (n_features, seq_length)   — 31 channels by default
    Target tensor shape  : (1, 36)

    Parameters
    ----------
    features_csv : str
        Path to flat daily features CSV.
    target_csv : str
        Path to sparse LAI targets CSV.
    norm_stats : dict
        Normalization stats from compute_norm_stats().
    site_ids : list of str, optional
        Restrict to these site IDs. Default: all sites in features_csv.
    seq_length : int
        Feature window length in days. Default 720 (≈ 2 years).
    years : list of int, optional
        If provided, only include samples where the target year is in this list.
    normalize_target : bool
        Z-score the LAI target. Default True.
    lai_norms : dict, optional
        Per-site {site_id: {"lai_min": float, "lai_max": float}} for min-max
        normalization. Overrides normalize_target when site_id is present.

    Examples
    --------
    >>> stats = compute_norm_stats("features.csv", "targets.csv", train_ids)
    >>> ds = LAIDataset("features.csv", "targets.csv", stats, site_ids=train_ids)
    >>> feats, targets = ds[0]
    >>> feats.shape
    torch.Size([31, 720])
    >>> targets.shape
    torch.Size([1, 36])
    """

    def __init__(
        self,
        features_csv: str,
        target_csv: str,
        norm_stats: Dict[str, Dict[str, float]],
        site_ids: Optional[List[str]] = None,
        seq_length: int = 720,
        years: Optional[List[int]] = None,
        normalize_target: bool = True,
        lai_norms: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.norm_stats = norm_stats
        self.normalize_target = normalize_target
        self.lai_norms = lai_norms

        feat_df = _load_features(features_csv, site_ids=site_ids)
        tgt_df = _load_targets(target_csv, site_ids=site_ids)

        # Pre-group targets by site_id for fast lookup
        tgt_by_site = {sid: grp for sid, grp in tgt_df.groupby("site_id")}

        # Per-site preprocessed feature matrices, keyed by site_id
        self._site_features: Dict[str, np.ndarray] = {}

        # Each entry: site_id, start_idx, end_idx, targets (36,), year
        self.samples: List[Dict] = []

        for site_id, site_feat in feat_df.groupby("site_id"):
            site_tgt = tgt_by_site.get(site_id)
            if site_tgt is None:
                continue

            site_feat_proc = _preprocess_site(site_feat)
            feat_matrix = _build_feature_matrix(site_feat_proc, norm_stats)
            self._site_features[site_id] = feat_matrix

            feature_years = site_feat_proc["year"].values

            for yr, yr_tgt in site_tgt.groupby("year"):
                if years is not None and yr not in years:
                    continue

                yr_tgt = yr_tgt.sort_values("date")
                if len(yr_tgt) != N_OBS_PER_YEAR:
                    continue

                yr_indices = np.where(feature_years == yr)[0]
                if len(yr_indices) == 0:
                    continue
                last_idx = int(yr_indices[-1])
                start_idx = last_idx - seq_length + 1
                if start_idx < 0:
                    continue

                target_vals = yr_tgt["LAI"].values.astype(np.float32)

                if lai_norms is not None and site_id in lai_norms:
                    lai_min = lai_norms[site_id]["lai_min"]
                    lai_max = lai_norms[site_id]["lai_max"]
                    denom = lai_max - lai_min
                    if denom > 1e-8:
                        target_vals = (target_vals - lai_min) / denom
                elif normalize_target:
                    target_vals = (
                        (target_vals - norm_stats["LAI"]["mean"])
                        / norm_stats["LAI"]["std"]
                    )

                self.samples.append(
                    {
                        "site_id": site_id,
                        "start_idx": start_idx,
                        "end_idx": last_idx + 1,
                        "targets": target_vals,
                        "year": int(yr),
                    }
                )

    @property
    def feature_channels(self) -> int:
        """Number of input feature channels (31 by default)."""
        return len(ALL_FEATURES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        features : (n_features, seq_length)
        targets  : (1, 36)
        """
        s = self.samples[index]
        window = self._site_features[s["site_id"]][s["start_idx"] : s["end_idx"]]
        features = torch.from_numpy(window.T.copy())            # (n_features, seq_length)
        targets = torch.from_numpy(s["targets"]).unsqueeze(0)   # (1, 36)
        return features, targets

    def get_site_info(self, index: int) -> Dict:
        """Return metadata for a sample (useful for evaluation / plotting)."""
        s = self.samples[index]
        return {"site_id": s["site_id"], "year": s["year"]}
