#!/usr/bin/env python3
"""Build a spatially balanced primary sample and training-only augmentation."""

import argparse
import datetime as dt
import json
import math
import warnings
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from scripts.build_selected_pixels import (
    GRID_SHAPE,
    era_coordinates,
    expected_dates,
    scan_geov2,
)


SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_TEST = 2
SPLIT_BUFFER = 3
SPLIT_NAMES = ("train", "validation", "test", "buffer")


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_qualifying(mask_dir, start_year, end_year, min_years):
    counter = np.zeros(GRID_SHAPE, dtype=np.uint8)
    for year in range(start_year, end_year + 1):
        path = mask_dir / f"valid_{year}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        counter += np.load(path, mmap_mode="r")
    flat = np.flatnonzero(counter >= min_years)
    rows, columns = np.unravel_index(flat, GRID_SHAPE)
    return flat.astype(np.int32), rows.astype(np.int32), columns.astype(np.int32)


def read_candidate_pft(path, rows, columns):
    fractions = np.empty((rows.size, 15), dtype=np.float32)
    with Dataset(path) as source:
        variable = source.variables["maxvegetfrac"]
        for pft in range(15):
            values = np.ma.filled(variable[0, pft, :, :], 0.0)
            fractions[:, pft] = values[rows, columns]
        latitude = source.variables["lat"][:].astype(np.float32)
        longitude = source.variables["lon"][:].astype(np.float32)
    maximum = np.max(fractions, axis=1)
    dominant = np.where(maximum > 0, np.argmax(fractions, axis=1) + 1, 0).astype(
        np.int8
    )
    mixed = maximum < 0.4
    return fractions, dominant, mixed, latitude, longitude


def coarse_lai(raw, rows, columns, min_valid_native):
    total = np.zeros(rows.size, dtype=np.float32)
    count = np.zeros(rows.size, dtype=np.uint8)
    native_rows = rows.astype(np.int64) * 2
    native_columns = columns.astype(np.int64) * 2
    for row_offset in (0, 1):
        for column_offset in (0, 1):
            values = raw[native_rows + row_offset, native_columns + column_offset]
            valid = (values != 255) & (values != 256)
            total += np.where(valid, values.astype(np.float32) / 30.0, 0.0)
            count += valid
    return np.where(count >= min_valid_native, total / np.maximum(count, 1), np.nan)


def representative_lai(year, geov2_dir, rows, columns, min_valid_native, cache_path):
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["mean"], cached["amplitude"]
    files = scan_geov2(geov2_dir)
    values = np.empty((36, rows.size), dtype=np.float32)
    for index, date in enumerate(expected_dates(year)):
        path = files.get(date)
        if path is None:
            raise RuntimeError(f"Missing GEOV2 file for {date}")
        import gzip
        import h5py

        with gzip.open(path, "rb") as compressed:
            with h5py.File(compressed, "r") as source:
                raw = source["LAI"][:]
        values[index] = coarse_lai(raw, rows, columns, min_valid_native)
        if index == 0 or (index + 1) % 6 == 0:
            print(f"{timestamp()} LAI diagnostics {index + 1}/36", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=0).astype(np.float32)
        amplitude = (
            np.nanpercentile(values, 90, axis=0) - np.nanpercentile(values, 10, axis=0)
        ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mean=mean, amplitude=amplitude)
    return mean, amplitude


def morton_code(rows, columns):
    rows = rows.astype(np.uint32)
    columns = columns.astype(np.uint32)
    output = np.zeros(rows.size, dtype=np.uint64)
    for bit in range(9):
        output |= ((rows >> bit) & 1).astype(np.uint64) << (2 * bit)
        output |= ((columns >> bit) & 1).astype(np.uint64) << (2 * bit + 1)
    return output


def systematic_pps_cells(cell_ids, cell_counts, sample_size, rng, draw):
    cell_rows = cell_ids // 360
    cell_columns = cell_ids % 360
    shift = int(rng.integers(0, 360))
    transformed_columns = (cell_columns + shift) % 360
    transformed_rows = 179 - cell_rows if draw % 2 else cell_rows
    order = np.argsort(morton_code(transformed_rows, transformed_columns))
    probabilities = sample_size * cell_counts[order] / cell_counts.sum()
    if np.any(probabilities >= 1):
        raise RuntimeError("One-primary-per-cell PPS probability reached one")
    thresholds = float(rng.random()) + np.arange(sample_size)
    chosen_positions = np.searchsorted(np.cumsum(probabilities), thresholds)
    return cell_ids[order[chosen_positions]]


def choose_pixels_in_cells(chosen_cells, sorted_cell_ids, sorted_candidates, rng):
    starts = np.searchsorted(sorted_cell_ids, chosen_cells, side="left")
    stops = np.searchsorted(sorted_cell_ids, chosen_cells, side="right")
    offsets = np.asarray(
        [rng.integers(start, stop) for start, stop in zip(starts, stops)],
        dtype=np.int64,
    )
    return sorted_candidates[offsets]


def standardized_balance_score(population, sample):
    population_mean = np.nanmean(population, axis=0)
    population_std = np.nanstd(population, axis=0)
    sample_mean = np.nanmean(sample, axis=0)
    valid = population_std > 1e-6
    return float(
        np.max(
            np.abs(
                (sample_mean[valid] - population_mean[valid]) / population_std[valid]
            )
        )
    )


def select_balanced_primary(
    rows,
    columns,
    fractions,
    lai_mean,
    lai_amplitude,
    sample_size,
    seed,
    draws,
):
    cell_ids = (rows // 10) * 360 + columns // 10
    order = np.argsort(cell_ids, kind="stable")
    sorted_cell_ids = cell_ids[order]
    unique_cells, cell_counts = np.unique(sorted_cell_ids, return_counts=True)
    if sample_size > unique_cells.size:
        raise RuntimeError(
            f"Requested {sample_size} primaries but only {unique_cells.size} "
            "one-degree cells are occupied"
        )
    latitude = 89.95 - rows * 0.1
    longitude = -179.95 + columns * 0.1
    diagnostics = np.column_stack(
        (
            fractions,
            lai_mean,
            lai_amplitude,
            np.sin(np.deg2rad(latitude)),
            np.sin(np.deg2rad(longitude)),
            np.cos(np.deg2rad(longitude)),
        )
    )
    rng = np.random.default_rng(seed)
    best = None
    best_score = math.inf
    for draw in range(draws):
        chosen_cells = systematic_pps_cells(
            unique_cells, cell_counts, sample_size, rng, draw
        )
        selected = choose_pixels_in_cells(chosen_cells, sorted_cell_ids, order, rng)
        score = standardized_balance_score(diagnostics, diagnostics[selected])
        if score < best_score:
            best = selected.copy()
            best_score = score
    return np.sort(best), best_score, unique_cells.size


def spherical_xyz(latitude, longitude):
    latitude = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    cosine = np.cos(latitude)
    return np.column_stack(
        (cosine * np.cos(longitude), cosine * np.sin(longitude), np.sin(latitude))
    )


def assign_cluster_splits(labels, dominant, lai_bin, ratios, seed):
    rng = np.random.default_rng(seed)
    clusters = np.unique(labels)
    pft_count = int(dominant.max()) + 1
    lai_count = int(lai_bin.max()) + 1
    features = np.zeros((clusters.size, 1 + pft_count + lai_count), dtype=np.float64)
    for index, cluster in enumerate(clusters):
        keep = labels == cluster
        features[index, 0] = keep.sum()
        features[index, 1 : 1 + pft_count] = np.bincount(
            dominant[keep], minlength=pft_count
        )
        features[index, 1 + pft_count :] = np.bincount(
            lai_bin[keep], minlength=lai_count
        )
    targets = np.outer(ratios, features.sum(axis=0))
    totals = np.zeros_like(targets)
    assignments = np.full(clusters.size, -1, dtype=np.int8)
    order = np.argsort(features[:, 0])[::-1]
    tied = rng.random(order.size)
    order = order[np.lexsort((tied, -features[order, 0]))]
    scale = np.maximum(targets, 1.0)

    def allocation_score(candidate):
        count_error = np.mean(((candidate[:, 0] - targets[:, 0]) / scale[:, 0]) ** 2)
        balance_error = np.mean(
            ((candidate[:, 1:] - targets[:, 1:]) / scale[:, 1:]) ** 2
        )
        return count_error + 0.25 * balance_error

    for cluster_index in order:
        scores = []
        for split in range(3):
            candidate = totals.copy()
            candidate[split] += features[cluster_index]
            scores.append(allocation_score(candidate))
        chosen = int(np.argmin(scores))
        assignments[cluster_index] = chosen
        totals[chosen] += features[cluster_index]

    while True:
        current_score = allocation_score(totals)
        best = None
        best_score = current_score
        for cluster_index, current in enumerate(assignments):
            for proposed in range(3):
                if proposed == current:
                    continue
                candidate = totals.copy()
                candidate[current] -= features[cluster_index]
                candidate[proposed] += features[cluster_index]
                score = allocation_score(candidate)
                if score < best_score - 1e-12:
                    best = (cluster_index, proposed, candidate)
                    best_score = score
        if best is None:
            break
        cluster_index, proposed, totals = best
        assignments[cluster_index] = proposed
    mapping = {
        int(cluster): int(assignments[index]) for index, cluster in enumerate(clusters)
    }
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int8)


def lai_bins(values):
    finite = np.isfinite(values)
    quantiles = np.nanquantile(values[finite & (values >= 0.25)], [0.25, 0.5, 0.75])
    bins = np.zeros(values.size, dtype=np.int8)
    bins[finite & (values >= 0.25)] = (
        np.digitize(values[finite & (values >= 0.25)], quantiles) + 1
    )
    return bins, quantiles


def chunk_ids(rows, columns):
    era_latitude, era_longitude = era_coordinates(rows, columns)
    lat_idx = np.rint((era_latitude.astype(np.float64) + 90.0) * 10).astype(np.int32)
    lon_idx = np.rint((era_longitude.astype(np.float64) + 179.9) * 10).astype(np.int32)
    return (lat_idx // 4).astype(np.int64) * 450 + lon_idx // 8


def build_augmented_selection(
    primary,
    rows,
    columns,
    primary_split,
    primary_group,
    cluster_model,
    buffer_degrees,
):
    all_chunks = chunk_ids(rows, columns)
    primary_chunks = all_chunks[primary]
    train_chunks = np.unique(primary_chunks[primary_split == SPLIT_TRAIN])
    candidate = np.flatnonzero(np.isin(all_chunks, train_chunks))
    candidate = candidate[~np.isin(candidate, primary)]

    latitude = 89.95 - rows[candidate] * 0.1
    longitude = -179.95 + columns[candidate] * 0.1
    candidate_xyz = spherical_xyz(latitude, longitude)
    candidate_group = cluster_model.predict(candidate_xyz)
    group_to_split = {}
    for group in np.unique(primary_group):
        group_splits = primary_split[primary_group == group]
        active = group_splits[group_splits != SPLIT_BUFFER]
        if active.size:
            group_to_split[int(group)] = int(active[0])
    keep = np.asarray(
        [
            group_to_split.get(int(group), SPLIT_BUFFER) == SPLIT_TRAIN
            for group in candidate_group
        ]
    )

    held_out = primary[
        (primary_split == SPLIT_VALIDATION) | (primary_split == SPLIT_TEST)
    ]
    held_out_xyz = spherical_xyz(
        89.95 - rows[held_out] * 0.1,
        -179.95 + columns[held_out] * 0.1,
    )
    distance, _ = cKDTree(held_out_xyz).query(candidate_xyz, k=1)
    chord_limit = 2 * math.sin(math.radians(buffer_degrees) / 2)
    keep &= distance > chord_limit
    return candidate[keep], candidate_group[keep].astype(np.int16)


def write_selection(
    path,
    selected,
    primary_lookup,
    rows,
    columns,
    pft_latitude,
    pft_longitude,
    fractions,
    dominant,
    mixed,
    lai_mean,
    lai_amplitude,
    lai_bin,
    split,
    group,
    nominal_probability,
    attrs,
):
    selected_rows = rows[selected]
    selected_columns = columns[selected]
    era_latitude, era_longitude = era_coordinates(selected_rows, selected_columns)
    selected_chunks = chunk_ids(selected_rows, selected_columns)
    is_primary = primary_lookup[selected]
    auxiliary_counts = {}
    for chunk in np.unique(selected_chunks[~is_primary]):
        auxiliary_counts[int(chunk)] = int(
            np.sum((selected_chunks == chunk) & ~is_primary)
        )
    weight = np.ones(selected.size, dtype=np.float32)
    for index in np.flatnonzero(~is_primary):
        weight[index] = 1.0 / auxiliary_counts[int(selected_chunks[index])]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("site", selected.size)
        output.createDimension("pft", 15)
        variables = {
            "lat_idx": ("i4", selected_rows),
            "lon_idx": ("i4", selected_columns),
            "latitude": ("f4", pft_latitude[selected_rows]),
            "longitude": ("f4", pft_longitude[selected_columns]),
            "era_latitude": ("f4", era_latitude),
            "era_longitude": ("f4", era_longitude),
            "arco_chunk_id": ("i8", selected_chunks),
            "is_primary": ("i1", is_primary.astype(np.int8)),
            "split": ("i1", split.astype(np.int8)),
            "spatial_group": ("i2", group.astype(np.int16)),
            "dominant_pft": ("i1", dominant[selected]),
            "mixed_pft": ("i1", mixed[selected].astype(np.int8)),
            "lai_bin": ("i1", lai_bin[selected]),
            "lai_mean": ("f4", lai_mean[selected]),
            "lai_amplitude": ("f4", lai_amplitude[selected]),
            "sample_weight": ("f4", weight),
            "inclusion_probability": (
                "f4",
                np.where(is_primary, nominal_probability, np.nan).astype(np.float32),
            ),
        }
        for name, (dtype, values) in variables.items():
            output.createVariable(name, dtype, ("site",), zlib=True)[:] = values
        output.createVariable("pft_fraction", "f4", ("site", "pft"), zlib=True)[:] = (
            fractions[selected]
        )
        site_id = output.createVariable("site_id", str, ("site",))
        site_id[:] = np.asarray(
            [
                f"pix_{row:04d}_{column:05d}"
                for row, column in zip(selected_rows, selected_columns)
            ],
            dtype=object,
        )
        for name, value in attrs.items():
            output.setncattr(name, value)
        output.split_names = ",".join(SPLIT_NAMES)
        output.source = "scripts/build_stratified_selection.py"
    temporary.replace(path)


def count_categories(values, size):
    counts = np.bincount(values, minlength=size)
    return {str(index): int(count) for index, count in enumerate(counts)}


def validate_selection(primary, auxiliary, rows, columns, primary_split, primary_group):
    primary_cells = (rows[primary] // 10) * 360 + columns[primary] // 10
    if np.unique(primary_cells).size != primary.size:
        raise RuntimeError(
            "Primary selection has more than one site in a 1-degree cell"
        )
    if np.intersect1d(primary, auxiliary).size:
        raise RuntimeError("Primary and auxiliary selections overlap")
    for group in np.unique(primary_group):
        active = np.unique(
            primary_split[(primary_group == group) & (primary_split != SPLIT_BUFFER)]
        )
        if active.size > 1:
            raise RuntimeError(f"Spatial group {group} crosses active data splits")
    held_out = primary[
        (primary_split == SPLIT_VALIDATION) | (primary_split == SPLIT_TEST)
    ]
    active_train = primary[primary_split == SPLIT_TRAIN]
    distance, _ = cKDTree(
        spherical_xyz(
            89.95 - rows[held_out] * 0.1,
            -179.95 + columns[held_out] * 0.1,
        )
    ).query(
        spherical_xyz(
            89.95 - rows[active_train] * 0.1,
            -179.95 + columns[active_train] * 0.1,
        ),
        k=1,
    )
    return math.degrees(2 * math.asin(float(distance.min()) / 2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--geov2-dir", type=Path, required=True)
    parser.add_argument("--pft-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1992)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--min-years", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.005)
    parser.add_argument("--diagnostic-year", type=int, default=2000)
    parser.add_argument("--min-valid-native", type=int, default=2)
    parser.add_argument("--balance-draws", type=int, default=100)
    parser.add_argument("--spatial-clusters", type=int, default=30)
    parser.add_argument("--buffer-degrees", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    flat, rows, columns = load_qualifying(
        args.mask_dir, args.start_year, args.end_year, args.min_years
    )
    sample_size = round(flat.size * args.fraction)
    print(f"Qualifying pixels: {flat.size:,}; primary target: {sample_size:,}")
    fractions, dominant, mixed, pft_latitude, pft_longitude = read_candidate_pft(
        args.pft_map, rows, columns
    )
    lai_mean, lai_amplitude = representative_lai(
        args.diagnostic_year,
        args.geov2_dir,
        rows,
        columns,
        args.min_valid_native,
        args.output_dir / f"lai_diagnostics_{args.diagnostic_year}.npz",
    )
    lai_bin, quantiles = lai_bins(lai_mean)
    primary, balance_score, occupied_cells = select_balanced_primary(
        rows,
        columns,
        fractions,
        lai_mean,
        lai_amplitude,
        sample_size,
        args.seed,
        args.balance_draws,
    )
    primary_xyz = spherical_xyz(
        pft_latitude[rows[primary]], pft_longitude[columns[primary]]
    )
    cluster_model = KMeans(
        n_clusters=args.spatial_clusters, random_state=args.seed, n_init=20
    ).fit(primary_xyz)
    primary_group = cluster_model.labels_.astype(np.int16)
    primary_split = assign_cluster_splits(
        primary_group,
        dominant[primary],
        lai_bin[primary],
        np.asarray([0.7, 0.15, 0.15]),
        args.seed,
    )

    held_out = primary[
        (primary_split == SPLIT_VALIDATION) | (primary_split == SPLIT_TEST)
    ]
    train_positions = np.flatnonzero(primary_split == SPLIT_TRAIN)
    distance, _ = cKDTree(
        spherical_xyz(pft_latitude[rows[held_out]], pft_longitude[columns[held_out]])
    ).query(primary_xyz[train_positions], k=1)
    chord_limit = 2 * math.sin(math.radians(args.buffer_degrees) / 2)
    primary_split[train_positions[distance <= chord_limit]] = SPLIT_BUFFER

    auxiliary, auxiliary_group = build_augmented_selection(
        primary,
        rows,
        columns,
        primary_split,
        primary_group,
        cluster_model,
        args.buffer_degrees,
    )
    minimum_split_distance = validate_selection(
        primary,
        auxiliary,
        rows,
        columns,
        primary_split,
        primary_group,
    )
    primary_lookup = np.zeros(flat.size, dtype=bool)
    primary_lookup[primary] = True
    primary_selected = np.sort(primary)
    augmented_selected = np.sort(np.concatenate((primary, auxiliary)))

    split_all = np.full(flat.size, SPLIT_BUFFER, dtype=np.int8)
    group_all = np.full(flat.size, -1, dtype=np.int16)
    split_all[primary] = primary_split
    group_all[primary] = primary_group
    split_all[auxiliary] = SPLIT_TRAIN
    group_all[auxiliary] = auxiliary_group
    attrs = {
        "fraction": float(args.fraction),
        "seed": int(args.seed),
        "n_qualifying": int(flat.size),
        "n_primary": int(primary.size),
        "n_auxiliary": int(auxiliary.size),
        "nominal_primary_probability": float(primary.size / flat.size),
        "occupied_one_degree_cells": int(occupied_cells),
        "balance_max_abs_smd": float(balance_score),
        "diagnostic_year": int(args.diagnostic_year),
        "lai_bin_quantiles": json.dumps(quantiles.tolist()),
        "buffer_degrees": float(args.buffer_degrees),
        "spatial_clusters": int(args.spatial_clusters),
        "minimum_train_to_holdout_degrees": float(minimum_split_distance),
        "population_dominant_pft_counts": json.dumps(
            count_categories(dominant, 16), sort_keys=True
        ),
        "primary_dominant_pft_counts": json.dumps(
            count_categories(dominant[primary], 16), sort_keys=True
        ),
        "population_lai_bin_counts": json.dumps(
            count_categories(lai_bin, 5), sort_keys=True
        ),
        "primary_lai_bin_counts": json.dumps(
            count_categories(lai_bin[primary], 5), sort_keys=True
        ),
    }
    write_selection(
        args.output_dir / "selected_primary_0p5pct.nc",
        primary_selected,
        primary_lookup,
        rows,
        columns,
        pft_latitude,
        pft_longitude,
        fractions,
        dominant,
        mixed,
        lai_mean,
        lai_amplitude,
        lai_bin,
        split_all[primary_selected],
        group_all[primary_selected],
        primary.size / flat.size,
        attrs,
    )
    write_selection(
        args.output_dir / "selected_primary_auxiliary.nc",
        augmented_selected,
        primary_lookup,
        rows,
        columns,
        pft_latitude,
        pft_longitude,
        fractions,
        dominant,
        mixed,
        lai_mean,
        lai_amplitude,
        lai_bin,
        split_all[augmented_selected],
        group_all[augmented_selected],
        primary.size / flat.size,
        attrs,
    )
    split_counts = {
        SPLIT_NAMES[index]: int(np.sum(primary_split == index)) for index in range(4)
    }
    diagnostics = {
        **attrs,
        "primary_split_counts": split_counts,
        "augmented_total": int(augmented_selected.size),
        "primary_arco_chunks": int(
            np.unique(chunk_ids(rows[primary], columns[primary])).size
        ),
        "augmented_arco_chunks": int(
            np.unique(
                chunk_ids(rows[augmented_selected], columns[augmented_selected])
            ).size
        ),
    }
    (args.output_dir / "stratified_selection.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
