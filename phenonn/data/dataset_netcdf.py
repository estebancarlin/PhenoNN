# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""NetCDF dataset for the selected-site global LAI pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from netCDF4 import Dataset
from torch.utils.data import Dataset as TorchDataset

METEO_FEATURES = [
    "Tmin",
    "Tmax",
    "Tmean",
    "ssrd_sum",
    "strd_sum",
    "tp_sum",
    "VPD_max",
    "VPD_mean",
    "Rn_tot",
    "PET",
    "SMI",
]
DERIVED_FEATURES = [
    "gdd_0",
    "gdd_5",
    "gdd_10",
    "cdd",
    "ncd",
    "botta_threshold",
    "botta_forcing",
]
PFT_FEATURES = [f"pft{index}_frac" for index in range(1, 16)]
CO2_FEATURES = ["co2"]
GLOBAL_ALL_FEATURES = METEO_FEATURES + CO2_FEATURES + PFT_FEATURES
LOG_TRANSFORM_FEATURES = {
    "tp_sum",
    "ssrd_sum",
    "strd_sum",
    "VPD_max",
    "VPD_mean",
    "SMI",
    "gdd_0",
    "gdd_5",
    "gdd_10",
    "cdd",
    "ncd",
}

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2, "buffer": 3}


def _site_ids(dataset):
    return np.asarray(dataset.variables["site_id"][:]).astype(str)


def _non_leap_indices(year):
    n_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    if n_days == 365:
        return np.arange(365, dtype=np.int32)
    return np.concatenate(
        (np.arange(59, dtype=np.int32), np.arange(60, 366, dtype=np.int32))
    )


def _derived_features(tmean):
    output = []
    for threshold in (0.0, 5.0, 10.0):
        output.append(np.cumsum(np.maximum(tmean - threshold, 0.0), axis=1))
    cdd = np.cumsum(np.maximum(5.0 - tmean, 0.0), axis=1)
    ncd = np.cumsum((tmean < 5.0).astype(np.float32), axis=1)
    botta_threshold = 964.0 * np.exp(-0.0058 * ncd) - 12.8
    botta_forcing = output[1] / np.maximum(botta_threshold, 1.0)
    output.extend((cdd, ncd, botta_threshold, botta_forcing))
    return np.stack(output, axis=1).astype(np.float32)


def _read_selected_sites(variable, time_indices, site_indices, site_batch=2048):
    output = np.empty((time_indices.size, site_indices.size), dtype=np.float32)
    n_source_site = variable.shape[1]
    for source_start in range(0, n_source_site, site_batch):
        source_stop = min(n_source_site, source_start + site_batch)
        left = int(np.searchsorted(site_indices, source_start, side="left"))
        right = int(np.searchsorted(site_indices, source_stop, side="left"))
        if left == right:
            continue
        local_sites = site_indices[left:right] - source_start
        slab = np.ma.filled(
            variable[time_indices, source_start:source_stop], np.nan
        ).astype(np.float32)
        output[:, left:right] = slab[:, local_sites]
    return output


class GlobalLAIDataset(TorchDataset):
    """Selected-site NetCDF counterpart of ``BigLAIDataset``.

    Each sample is one ``(site_id, target_year)`` pair. By default the feature
    order follows the historical global pipeline: 11 ERA5-Land variables,
    annual CO2, then 15 ORCHIDEE PFT fractions. Values remain in raw physical
    units and the return contract stays ``(features, targets)``.
    """

    def __init__(
        self,
        era_dir: str,
        target_dir: str,
        selection_path: str,
        years: Sequence[int],
        site_ids: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        include_auxiliary: bool = True,
        seq_length: int = 720,
        min_valid_dekads: int = 26,
        add_pheno_features: bool = False,
        normalize: bool = False,
        norm_stats: Optional[Dict] = None,
        norm_stats_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if seq_length > 730 or seq_length < 365:
            raise ValueError("seq_length must be between 365 and 730 days")
        if split is not None and split not in SPLIT_CODES:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_CODES}")

        self.era_dir = Path(era_dir)
        self.target_dir = Path(target_dir)
        self.selection_path = Path(selection_path)
        self.years = sorted(set(int(year) for year in years))
        self.seq_length = seq_length
        self.add_pheno_features = add_pheno_features
        self.feature_names = (
            METEO_FEATURES
            + (DERIVED_FEATURES if add_pheno_features else [])
            + CO2_FEATURES
            + PFT_FEATURES
        )
        self.normalize = normalize
        if norm_stats is not None and norm_stats_path is not None:
            raise ValueError("Provide either norm_stats or norm_stats_path, not both")
        if norm_stats_path is not None:
            with Path(norm_stats_path).open(encoding="utf-8") as stream:
                norm_stats = json.load(stream)
        if normalize:
            if norm_stats is None:
                raise ValueError(
                    "normalize=True requires norm_stats or norm_stats_path"
                )
            self.norm_stats = norm_stats.get("statistics", norm_stats)
            missing_stats = [
                feature
                for feature in self.feature_names
                if feature not in self.norm_stats
            ]
            if missing_stats:
                raise ValueError(
                    f"Normalization statistics are missing {missing_stats}"
                )
        else:
            self.norm_stats = None

        with Dataset(self.selection_path) as selection:
            canonical_ids = _site_ids(selection)
            selected_split = selection.variables["split"][:]
            primary = selection.variables["is_primary"][:].astype(bool)
            chunks = selection.variables["arco_chunk_id"][:]
            stored_weight = selection.variables["sample_weight"][:].astype(np.float32)

        keep = np.ones(canonical_ids.size, dtype=bool)
        if split is not None:
            keep &= selected_split == SPLIT_CODES[split]
        if not include_auxiliary:
            keep &= primary
        if site_ids is not None:
            keep &= np.isin(canonical_ids, np.asarray(site_ids).astype(str))
        selected_indices = np.flatnonzero(keep).astype(np.int32)
        if not selected_indices.size:
            raise RuntimeError(
                "No sites match the requested IDs, split, and role filters"
            )

        # Re-establish per-chunk auxiliary normalization after ERA-land filtering.
        filtered_weight = stored_weight.copy()
        auxiliary = ~primary
        for chunk in np.unique(chunks[auxiliary]):
            members = auxiliary & (chunks == chunk)
            filtered_weight[members] = 1.0 / members.sum()

        self.site_indices = selected_indices
        self.site_ids = canonical_ids[selected_indices]
        self.site_splits = selected_split[selected_indices].astype(np.int8)
        self.site_primary = primary[selected_indices]
        self.site_weights = filtered_weight[selected_indices]
        self.site_chunks = chunks[selected_indices]
        self._co2 = self._load_co2()
        self._features_by_year: Dict[int, np.ndarray] = {}
        self._pft_by_year: Dict[int, np.ndarray] = {}
        self._targets_by_year: Dict[int, np.ndarray] = {}
        self._target_valid_by_year: Dict[int, np.ndarray] = {}
        self.samples: List[Tuple[int, int]] = []

        feature_years = sorted(set(self.years) | {year - 1 for year in self.years})
        for year in feature_years:
            self._features_by_year[year] = self._load_era_year(year)
        for year in self.years:
            targets, valid = self._load_target_year(year)
            pft, pft_valid = self._load_pft_year(year)
            self._targets_by_year[year] = targets
            self._target_valid_by_year[year] = valid
            self._pft_by_year[year] = pft
            for site_position in range(self.site_ids.size):
                if (
                    valid[site_position].sum() >= min_valid_dekads
                    and pft_valid[site_position]
                ):
                    self.samples.append((site_position, year))

        if not self.samples:
            raise RuntimeError("No valid site-year samples were found")

    @property
    def feature_channels(self) -> int:
        return len(self.feature_names)

    def _load_co2(self):
        path = self.target_dir / "CO2_annual.nc"
        with Dataset(path) as source:
            years = source.variables["year"][:]
            values = source.variables["co2"][:]
        return {int(year): float(value) for year, value in zip(years, values)}

    def _check_site_order(self, source, path):
        source_ids = _site_ids(source)
        if source_ids.size <= int(self.site_indices.max()):
            raise RuntimeError(f"Site dimension is too short in {path}")
        if not np.array_equal(source_ids[self.site_indices], self.site_ids):
            raise RuntimeError(
                f"Site order does not match canonical selection in {path}"
            )

    def _load_era_year(self, year):
        path = self.era_dir / f"ERA5_daily_pixelset_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        indices = _non_leap_indices(year)
        with Dataset(path) as source:
            self._check_site_order(source, path)
            expected = (
                366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
            )
            if len(source.dimensions["time"]) != expected:
                raise RuntimeError(f"Unexpected day count in {path}")
            arrays = []
            for name in METEO_FEATURES:
                values = _read_selected_sites(
                    source.variables[name], indices, self.site_indices
                )
                if np.any(~np.isfinite(values)):
                    raise RuntimeError(f"Non-finite {name} values in {path}")
                arrays.append(values.T)
        meteo = np.stack(arrays, axis=1)
        if self.add_pheno_features:
            derived = _derived_features(meteo[:, METEO_FEATURES.index("Tmean"), :])
            meteo = np.concatenate((meteo, derived), axis=1)
        return meteo

    def _load_target_year(self, year):
        path = self.target_dir / "lai" / f"LAI_dekadal_{year}.nc"
        with Dataset(path) as source:
            self._check_site_order(source, path)
            targets = (
                np.ma.filled(source.variables["LAI"][:, self.site_indices], np.nan)
                .astype(np.float32)
                .T
            )
            valid = (
                source.variables["target_valid"][:, self.site_indices].astype(bool).T
            )
        if not np.array_equal(np.isfinite(targets), valid):
            raise RuntimeError(f"Target mask mismatch in {path}")
        return targets, valid

    def _load_pft_year(self, year):
        path = self.target_dir / "pft" / f"PFTmap_{year}.nc"
        with Dataset(path) as source:
            self._check_site_order(source, path)
            pft = (
                source.variables["pft_frac"][:, self.site_indices].astype(np.float32).T
            )
            valid = source.variables["pft_valid"][self.site_indices].astype(bool)
        return pft, valid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        site_position, year = self.samples[index]
        meteo = np.concatenate(
            (
                self._features_by_year[year - 1][site_position],
                self._features_by_year[year][site_position],
            ),
            axis=1,
        )
        meteo = meteo[:, -self.seq_length :]
        if year not in self._co2:
            raise RuntimeError(f"Missing CO2 value for {year}")
        co2 = np.full((1, self.seq_length), self._co2[year], dtype=np.float32)
        pft = np.broadcast_to(
            self._pft_by_year[year][site_position, :, None],
            (15, self.seq_length),
        )
        history = np.concatenate((meteo, co2, pft), axis=0)
        if self.normalize:
            for channel, feature in enumerate(self.feature_names):
                values = history[channel]
                if feature in LOG_TRANSFORM_FEATURES:
                    values = np.log1p(np.clip(values, 0.0, None))
                statistic = self.norm_stats[feature]
                history[channel] = (values - float(statistic["mean"])) / max(
                    float(statistic["std"]), 1e-8
                )
        features = torch.from_numpy(history.copy())
        targets = torch.from_numpy(self._targets_by_year[year][site_position].copy())
        return features, targets.unsqueeze(0)

    def get_site_info(self, index: int) -> Dict:
        site_position, year = self.samples[index]
        return {
            "site_id": self.site_ids[site_position],
            "year": year,
            "split": int(self.site_splits[site_position]),
            "is_primary": bool(self.site_primary[site_position]),
            "sample_weight": float(self.site_weights[site_position]),
            "arco_chunk_id": int(self.site_chunks[site_position]),
            "target_valid": self._target_valid_by_year[year][site_position].copy(),
        }
