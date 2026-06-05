# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Streaming dataset for very large LAI corpora stored as one CSV file per year.

The dataset is designed for memory-efficient training on large LAI datasets.
Feature and target records are stored in separate yearly CSV files and are
loaded on demand. Only the files required for the current epoch are opened,
and data are streamed in chunks so that rows corresponding to the selected
``site_ids`` are materialized in memory.

This implementation does not perform feature or target normalization.
Skipping normalization avoids the full-dataset scan required to compute
global statistics and allows training directly on raw values.

Storage Layout
--------------
Features:
    {features_dir}/features_{year}.csv

    Contains daily feature records with approximately 365 (or 366) rows
    per site and year.

Targets:
    {target_dir}/target_{year}.csv

    Contains 36 target observations per site and year.

Sample Definition
-----------------
Each sample corresponds to a ``(site_id, target_year)`` pair.

Features:
    Array of shape ``(n_features, seq_length)`` containing the last
    ``seq_length`` days of feature history for the site, ending on
    December 31 of ``target_year``.

Targets:
    Array of shape ``(1, 36)`` containing the 36 observation values
    associated with ``target_year``.

Historical Context
------------------
Constructing a feature sequence requires observations from both the
target year and the preceding year. Therefore, for each target year
``Y``, feature files from years ``Y`` and ``Y - 1`` must be available.

Examples
--------
Instantiate a dataset using epoch-level sampling:

>>> ds = BigLAIDataset(
...     features_dir="data/data_features",
...     target_dir="data/data_targets",
...     years=[2008, 2014],
...     site_ids=sampled_site_ids,
...     seq_length=720,
... )

Notes
-----
A common training workflow is to sample sites and years at the start of
each epoch and instantiate a new dataset using the selected subset. This
keeps memory usage bounded even when working with very large LAI corpora.
"""

import io
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Set, Tuple

from .feature_engineering import add_derived_features


# ── Feature definitions ───────────────────────────────────────────────────────
#
# Re-export the canonical feature lists from dataset_target_feature so the two
# datasets always agree on channel count. The toggles
#   add_pheno_features / add_cyclic_features
# live in dataset_target_feature.py — flip them there to control both
# datasets at once.
from .dataset_flat import (
    METEO_FEATURES,
    ALL_FEATURES,
    N_OBS_PER_YEAR,
)

# File naming conventions in the data folders
FEATURES_FNAME = "feature_{year}.csv"
TARGET_FNAME = "target_{year}.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────


def generate_site_ids_from_range(
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
) -> List[str]:
    """
    Build every `pix_{row:04d}_{col:05d}` id in the given inclusive grid range.

    The caller can pass this list directly to BigLAIDataset; ids that do not
    appear in the CSV files are silently dropped at load time, so the range
    can be a superset of the actually-extracted sites.
    """
    r0, r1 = row_range
    c0, c1 = col_range
    return [
        f"pix_{r:04d}_{c:05d}" for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
    ]


# ─── Byte-offset pixel index ──────────────────────────────────────────────────
#
# CSVs are sorted pixel-major (all rows of pixel A, then all rows of pixel B,
# …). Once we know the byte offset where each pixel's block starts, loading
# the data for an arbitrary set of pixels becomes a sequence of file seeks +
# fixed-length reads — no full file scan.
#
# The index for a CSV is persisted next to it as `{csv}.pixidx.npz` and
# auto-rebuilt when the CSV's mtime is newer than the index.

INDEX_SUFFIX = ".pixidx.npz"


def _index_path(csv_path: str) -> str:
    return csv_path + INDEX_SUFFIX


def _build_pixel_index(
    csv_path: str,
    progress_every_mb: int = 500,
) -> Tuple[np.ndarray, np.ndarray, bytes]:
    """
    Scan `csv_path` once and record:
        site_ids[i] : site_id of the i-th pixel block (str)
        offsets[i]  : start byte of pixel i  (int64)
        offsets[-1] : file size sentinel  →  block_i_size = offsets[i+1] − offsets[i]
        header      : raw bytes of the CSV header (incl. trailing '\\n')

    Notes
    -----
    Reads each line just to extract the leading `site_id` field. Variable
    block sizes (e.g. leap years for features) are handled automatically —
    no assumption on rows-per-pixel.
    """
    site_ids: List[str] = []
    offsets: List[int] = []
    file_size = os.path.getsize(csv_path)
    t0 = time.time()
    last_report = 0

    with open(csv_path, "rb") as f:
        header = f.readline()
        current_sid = b""
        pos = f.tell()
        for line in f:
            sid = line.split(b",", 1)[0]
            if sid != current_sid:
                site_ids.append(sid.decode("utf-8"))
                offsets.append(pos)
                current_sid = sid
            pos += len(line)

            if (
                progress_every_mb
                and pos - last_report > progress_every_mb * 1024 * 1024
            ):
                last_report = pos
                pct = pos * 100.0 / max(file_size, 1)
                print(
                    f"      pixel index  {pct:5.1f}% "
                    f"({pos / 1e9:.2f} / {file_size / 1e9:.2f} GB, "
                    f"{len(site_ids):,} pixels, "
                    f"{time.time() - t0:.0f}s)",
                    flush=True,
                )
        offsets.append(pos)  # EOF sentinel

    return (
        np.array(site_ids),
        np.array(offsets, dtype=np.int64),
        header,
    )


def _save_pixel_index(
    csv_path: str,
    site_ids: np.ndarray,
    offsets: np.ndarray,
    header: bytes,
) -> None:
    np.savez(
        _index_path(csv_path),
        site_ids=site_ids,
        offsets=offsets,
        header=np.frombuffer(header, dtype=np.uint8).copy(),
        csv_mtime=np.array([os.path.getmtime(csv_path)], dtype=np.float64),
    )


def _load_pixel_index(
    csv_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, bytes]]:
    """Load a cached index, or return None if missing / stale."""
    idx_path = _index_path(csv_path)
    if not os.path.exists(idx_path):
        return None
    try:
        data = np.load(idx_path)
        if "csv_mtime" in data.files:
            csv_mtime = os.path.getmtime(csv_path)
            # Allow 1s of tolerance for filesystem clock skew.
            if float(data["csv_mtime"][0]) < csv_mtime - 1.0:
                return None
        header = bytes(data["header"])
        return data["site_ids"], data["offsets"], header
    except Exception:
        return None


def get_pixel_index(
    csv_path: str,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bytes]:
    """
    Return (site_ids, offsets, header), building & caching the index on
    first call. Subsequent calls hit the on-disk cache instantly.
    """
    cached = _load_pixel_index(csv_path)
    if cached is not None:
        return cached
    if verbose:
        print(
            f"    Building pixel index for {os.path.basename(csv_path)} "
            f"({os.path.getsize(csv_path) / 1e9:.2f} GB) — one-time cost",
            flush=True,
        )
    site_ids, offsets, header = _build_pixel_index(csv_path)
    _save_pixel_index(csv_path, site_ids, offsets, header)
    if verbose:
        print(
            f"    Index saved → {_index_path(csv_path)} " f"({len(site_ids):,} pixels)",
            flush=True,
        )
    return site_ids, offsets, header


def _load_year_filtered(
    path: str,
    wanted_sites: Set[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read only the pixel blocks listed in `wanted_sites` from `path`, using
    the byte-offset index. Reads are issued in forward byte order so the
    OS page cache works in our favour.

    No `chunk_size` argument: the per-pixel block size is read directly from
    the index (offsets[i+1] − offsets[i]).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    site_ids, offsets, header = get_pixel_index(path, verbose=verbose)
    site_to_idx = {s: i for i, s in enumerate(site_ids)}

    # Resolve wanted sites to indices, drop unknown ids, sort by byte offset
    matched = [site_to_idx[s] for s in wanted_sites if s in site_to_idx]
    if not matched:
        return pd.DataFrame()
    matched.sort()

    blocks = [header]
    with open(path, "rb") as f:
        for i in matched:
            start = int(offsets[i])
            end = int(offsets[i + 1])
            f.seek(start)
            blocks.append(f.read(end - start))

    return pd.read_csv(io.BytesIO(b"".join(blocks)))


def _preprocess_site(site_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add doy / cyclic encodings and derived ecological features for one site.

    Sorts by (year, doy). Forward-/back-fills small meteo gaps. The output
    contains every ALL_FEATURES column.
    """
    df = site_df.copy()
    dates = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["doy"] = dates.dt.day_of_year.values
    df = df.sort_values(["year", "doy"]).reset_index(drop=True)

    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    df = add_derived_features(df)

    for col in METEO_FEATURES:
        df[col] = df[col].ffill().bfill()

    return df


def _build_feature_matrix(site_proc: pd.DataFrame) -> np.ndarray:
    """
    Build a (n_days, n_features) float32 matrix in canonical ALL_FEATURES order.

    No normalization is applied — values stay on their native scales.
    """
    n = len(site_proc)
    m = np.zeros((n, len(ALL_FEATURES)), dtype=np.float32)
    for i, feat in enumerate(ALL_FEATURES):
        m[:, i] = site_proc[feat].values.astype(np.float32)
    return m


# ── Dataset ───────────────────────────────────────────────────────────────────


class BigLAIDataset(Dataset):
    """
    Streaming, per-epoch dataset for LAI prediction.

    Parameters
    ----------
    features_dir : str
        Folder containing `features_{year}.csv` files.
    target_dir : str
        Folder containing `target_{year}.csv` files.
    years : list of int
        Target years for which (site, year) samples will be produced. For
        each `Y` in this list, the dataset also loads the previous year's
        feature file (`features_{Y-1}.csv`) so the seq_length-day window is
        complete.
    site_ids : list of str
        Sites to keep. Ids that do not appear in the relevant year CSVs are
        silently dropped — pass the full candidate list and let the loader
        filter.
    seq_length : int
        Feature window length in days (default 720 ≈ 2 years).
    normalize : bool
        Must be False for now. Reserved for a future opt-in normalization.

    Notes
    -----
    All required data is loaded into memory at construction time. With ~500
    sites and ~3 years per epoch this stays in the few-hundred-MB range. The
    expectation is that `main_big.py` creates a fresh dataset every epoch.
    """

    def __init__(
        self,
        features_dir: str,
        target_dir: str,
        years: List[int],
        site_ids: List[str],
        seq_length: int = 720,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        if normalize:
            raise NotImplementedError(
                "normalize=True is not implemented yet — the point of "
                "BigLAIDataset is to avoid the full-dataset scan needed by "
                "compute_norm_stats."
            )
        self.seq_length = seq_length

        wanted_sites: Set[str] = set(site_ids)
        target_years = sorted(set(int(y) for y in years))
        feature_years_needed = sorted(
            set(target_years) | set(y - 1 for y in target_years)
        )

        # ── Sanity check the directories upfront ──
        # Dropping a single missing year is fine (e.g. Y-1 outside the
        # extraction range), but a non-existent or empty data folder is
        # almost certainly a typo and we should surface it loudly.
        for label, path in [("features_dir", features_dir), ("target_dir", target_dir)]:
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"{label}={path!r} does not exist or is not a directory"
                )

        # ── Load only the feature year files we need ──
        feat_parts = []
        feat_files_found: List[str] = []
        feat_files_missing: List[str] = []
        for y in feature_years_needed:
            path = os.path.join(features_dir, FEATURES_FNAME.format(year=y))
            if not os.path.exists(path):
                # Year-before may legitimately be missing if it's outside the
                # extraction range. Tracked so the final report shows it.
                feat_files_missing.append(os.path.basename(path))
                continue
            feat_files_found.append(os.path.basename(path))
            df = _load_year_filtered(path, wanted_sites)
            if not df.empty:
                feat_parts.append(df)

        if not feat_parts:
            raise RuntimeError(
                f"No feature data could be loaded.\n"
                f"  features_dir : {features_dir}\n"
                f"  files needed : {[FEATURES_FNAME.format(year=y) for y in feature_years_needed]}\n"
                f"  files found  : {feat_files_found}\n"
                f"  files missing: {feat_files_missing}\n"
                f"  n wanted sites: {len(wanted_sites)}\n"
                f"Either the folder/filename pattern is wrong, or none of the "
                f"wanted sites appear in the existing feature files."
            )

        all_feat_df = pd.concat(feat_parts, ignore_index=True)

        # ── Load only the target year files we need ──
        tgt_parts = []
        tgt_files_found: List[str] = []
        tgt_files_missing: List[str] = []
        for y in target_years:
            path = os.path.join(target_dir, TARGET_FNAME.format(year=y))
            if not os.path.exists(path):
                tgt_files_missing.append(os.path.basename(path))
                continue
            tgt_files_found.append(os.path.basename(path))
            df = _load_year_filtered(path, wanted_sites)
            if not df.empty:
                tgt_parts.append(df)

        if not tgt_parts:
            raise RuntimeError(
                f"No target data could be loaded.\n"
                f"  target_dir   : {target_dir}\n"
                f"  files needed : {[TARGET_FNAME.format(year=y) for y in target_years]}\n"
                f"  files found  : {tgt_files_found}\n"
                f"  files missing: {tgt_files_missing}\n"
                f"  n wanted sites: {len(wanted_sites)}\n"
                f"Either the folder/filename pattern is wrong, or none of the "
                f"wanted sites appear in the existing target files."
            )

        all_tgt_df = pd.concat(tgt_parts, ignore_index=True)

        # ── Per-site preprocessing ──
        # Build the (n_days, 31) raw feature matrix for each site, and an
        # index from year → last-row-position so we can carve the window.
        self._site_features = {}
        site_year_indices: Dict[str, Dict[int, int]] = {}

        for site_id, site_feat in all_feat_df.groupby("site_id"):
            site_proc = _preprocess_site(site_feat)
            self._site_features[site_id] = _build_feature_matrix(site_proc)
            years_arr = site_proc["year"].values
            site_year_indices[site_id] = {
                int(yr): int(np.where(years_arr == yr)[0][-1])
                for yr in np.unique(years_arr)
            }

        # ── Build samples ──
        tgt_by_site = {sid: grp for sid, grp in all_tgt_df.groupby("site_id")}

        self.samples = []
        n_with_nan_target = 0
        n_dropped_feature_nan = 0
        for site_id, year_idx_map in site_year_indices.items():
            tgt_g = tgt_by_site.get(site_id)
            if tgt_g is None:
                continue

            for yr, yr_tgt in tgt_g.groupby("year"):
                yr = int(yr)
                if yr not in target_years:
                    continue
                yr_tgt = yr_tgt.sort_values("date")
                if len(yr_tgt) != N_OBS_PER_YEAR:
                    continue

                last_idx = year_idx_map.get(yr)
                if last_idx is None:
                    continue

                start_idx = last_idx - seq_length + 1
                if start_idx < 0:
                    # Not enough history (missing previous-year file or short
                    # site history) — skip this sample.
                    continue

                # Drop the sample if the feature window contains any NaN —
                # there's no clean way to mask NaN inputs to an LSTM and they
                # would blow up the forward pass.
                window = self._site_features[site_id][start_idx : last_idx + 1]
                if np.any(np.isnan(window)):
                    n_dropped_feature_nan += 1
                    continue

                # NaN in the target is OK: the training loop in main_big.py
                # uses a NaN-aware MSE that masks out positions where target
                # is NaN. We just count them for the report.
                target_vals = yr_tgt["LAI"].values.astype(np.float32)
                if np.any(np.isnan(target_vals)):
                    n_with_nan_target += 1

                self.samples.append(
                    {
                        "site_id": site_id,
                        "year": yr,
                        "start_idx": start_idx,
                        "end_idx": last_idx + 1,
                        "targets": target_vals,
                    }
                )

        if n_with_nan_target or n_dropped_feature_nan:
            print(
                f"    [BigLAIDataset] kept {len(self.samples)} samples "
                f"({n_with_nan_target} contain ≥1 NaN target day, handled by "
                f"NaN-safe loss). Dropped {n_dropped_feature_nan} with NaN "
                f"features."
            )

    # ── Required Dataset methods ──────────────────────────────────────────────

    @property
    def feature_channels(self) -> int:
        return len(ALL_FEATURES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[index]
        window = self._site_features[s["site_id"]][s["start_idx"] : s["end_idx"]]
        features = torch.from_numpy(window.T.copy())  # (31, seq_length)
        targets = torch.from_numpy(s["targets"]).unsqueeze(0)  # (1, 36)
        return features, targets

    def get_site_info(self, index: int) -> Dict:
        s = self.samples[index]
        return {"site_id": s["site_id"], "year": s["year"]}
