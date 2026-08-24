#!/usr/bin/env python3
"""Validate and freeze selected-site ERA5-Land daily artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

from scripts.download_era5_selected import FINAL_VARIABLES


SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_TEST = 2
SPLIT_BUFFER = 3


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def same_values(reference, candidate):
    reference = np.ma.filled(reference, np.nan)
    candidate = np.ma.filled(candidate, np.nan)
    if reference.dtype.kind in "f" or candidate.dtype.kind in "f":
        return np.allclose(reference, candidate, rtol=0.0, atol=1e-6, equal_nan=True)
    return np.array_equal(reference, candidate)


def expected_dates(year):
    return np.arange(
        np.datetime64(f"{year}-01-01"),
        np.datetime64(f"{year + 1}-01-01"),
        dtype="datetime64[D]",
    )


def decoded_dates(time_variable):
    values = num2date(
        time_variable[:],
        units=time_variable.units,
        calendar=getattr(time_variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return np.asarray([np.datetime64(value.strftime("%Y-%m-%d")) for value in values])


def selection_metadata(selection):
    metadata = {}
    for name, variable in selection.variables.items():
        if "site" in variable.dimensions:
            metadata[name] = variable[:]
    return metadata


def validate_selection(selection, errors):
    n_site = len(selection.dimensions["site"])
    required = {
        "site_id",
        "lat_idx",
        "lon_idx",
        "latitude",
        "longitude",
        "era_latitude",
        "era_longitude",
        "arco_chunk_id",
        "is_primary",
        "split",
        "spatial_group",
        "sample_weight",
        "source_site_index",
    }
    missing = sorted(required - set(selection.variables))
    if missing:
        errors.append(f"Selection is missing variables: {missing}")
        return {}

    site_ids = np.asarray(selection.variables["site_id"][:]).astype(str)
    if np.unique(site_ids).size != n_site:
        errors.append("Selection site_id values are not unique")
    source_indices = selection.variables["source_site_index"][:]
    if np.unique(source_indices).size != n_site:
        errors.append("Selection source_site_index values are not unique")

    primary = selection.variables["is_primary"][:].astype(bool)
    split = selection.variables["split"][:]
    weight = selection.variables["sample_weight"][:]
    if np.any(
        ~np.isin(split, [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_BUFFER])
    ):
        errors.append("Selection contains an unknown split code")
    if np.any((~primary) & (split != SPLIT_TRAIN)):
        errors.append("Auxiliary sites occur outside the training split")
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0):
        errors.append("Selection sample weights must be finite and positive")

    spatial_group = selection.variables["spatial_group"][:]
    for group in np.unique(spatial_group):
        active_splits = np.unique(
            split[(spatial_group == group) & (split != SPLIT_BUFFER)]
        )
        if active_splits.size > 1:
            errors.append(
                f"Spatial group {int(group)} crosses active splits "
                f"{active_splits.tolist()}"
            )

    if "pft_fraction" in selection.variables:
        pft_fraction = np.ma.filled(
            selection.variables["pft_fraction"][:], np.nan
        ).astype(np.float32)
        if np.any(~np.isfinite(pft_fraction)):
            errors.append("Selection PFT fractions contain non-finite values")
        if np.any((pft_fraction < 0) | (pft_fraction > 1)):
            errors.append("Selection PFT fractions are outside [0, 1]")
        if np.any(np.sum(pft_fraction, axis=1) <= 0.5):
            errors.append("Selection contains sites with PFT fraction sum <= 0.5")

    chunks = selection.variables["arco_chunk_id"][:]
    auxiliary = ~primary
    chunk_weight_sums = []
    for chunk in np.unique(chunks[auxiliary]):
        keep = auxiliary & (chunks == chunk)
        chunk_weight_sums.append(float(np.sum(weight[keep], dtype=np.float64)))
    chunk_weight_sums = np.asarray(chunk_weight_sums)
    return {
        "n_site": n_site,
        "n_primary": int(primary.sum()),
        "n_auxiliary": int(auxiliary.sum()),
        "split_counts": {str(code): int(np.sum(split == code)) for code in range(4)},
        "auxiliary_chunk_count": int(chunk_weight_sums.size),
        "auxiliary_chunk_weight_sum_min": float(chunk_weight_sums.min()),
        "auxiliary_chunk_weight_sum_max": float(chunk_weight_sums.max()),
        "auxiliary_chunk_weight_sum_mean": float(chunk_weight_sums.mean()),
        "auxiliary_effective_weight": float(
            np.sum(weight[auxiliary], dtype=np.float64)
        ),
        "primary_effective_weight": float(np.sum(weight[primary], dtype=np.float64)),
    }


def validate_metadata(year, daily, metadata, errors):
    for name, reference in metadata.items():
        if name not in daily.variables:
            errors.append(f"{year}: missing selection metadata variable {name}")
            continue
        candidate = daily.variables[name][:]
        if not same_values(reference, candidate):
            errors.append(f"{year}: metadata mismatch for {name}")


def update_statistic(statistic, values):
    finite = np.isfinite(values)
    statistic["count"] += int(values.size)
    statistic["non_finite"] += int(values.size - finite.sum())
    if np.any(finite):
        selected = values[finite].astype(np.float64, copy=False)
        statistic["sum"] += float(selected.sum())
        statistic["minimum"] = min(statistic["minimum"], float(selected.min()))
        statistic["maximum"] = max(statistic["maximum"], float(selected.max()))


def finalize_statistics(statistics):
    output = {}
    for name, statistic in statistics.items():
        finite_count = statistic["count"] - statistic["non_finite"]
        output[name] = {
            "count": statistic["count"],
            "non_finite": statistic["non_finite"],
            "minimum": statistic["minimum"],
            "maximum": statistic["maximum"],
            "mean": statistic["sum"] / finite_count if finite_count else math.nan,
        }
    return output


def validate_year(path, year, metadata, site_batch, errors):
    expected = expected_dates(year)
    statistics = {
        name: {
            "count": 0,
            "non_finite": 0,
            "sum": 0.0,
            "minimum": math.inf,
            "maximum": -math.inf,
        }
        for name in FINAL_VARIABLES
    }
    logical_errors = {
        "Tmin_gt_Tmean": 0,
        "Tmean_gt_Tmax": 0,
        "VPD_mean_gt_VPD_max": 0,
        "negative_precipitation": 0,
        "negative_vpd": 0,
        "negative_pet": 0,
    }
    with Dataset(path) as daily:
        if "site" not in daily.dimensions or len(daily.dimensions["site"]) != len(
            metadata["site_id"]
        ):
            errors.append(f"{year}: site dimension does not match selection")
            return {}, logical_errors
        if "time" not in daily.variables:
            errors.append(f"{year}: missing time variable")
            return {}, logical_errors
        dates = decoded_dates(daily.variables["time"])
        if not np.array_equal(dates, expected):
            errors.append(
                f"{year}: dates are not the complete contiguous calendar year"
            )
        validate_metadata(year, daily, metadata, errors)

        for name, (units, _) in FINAL_VARIABLES.items():
            if name not in daily.variables:
                errors.append(f"{year}: missing feature {name}")
                continue
            variable = daily.variables[name]
            if variable.dimensions != ("time", "site"):
                errors.append(f"{year}: {name} dimensions are {variable.dimensions}")
            if getattr(variable, "units", None) != units:
                errors.append(
                    f"{year}: {name} units are {getattr(variable, 'units', None)!r}, "
                    f"expected {units!r}"
                )

        if any(name not in daily.variables for name in FINAL_VARIABLES):
            return finalize_statistics(statistics), logical_errors

        n_site = len(daily.dimensions["site"])
        for start in range(0, n_site, site_batch):
            stop = min(n_site, start + site_batch)
            values = {
                name: np.ma.filled(daily.variables[name][:, start:stop], np.nan).astype(
                    np.float32
                )
                for name in FINAL_VARIABLES
            }
            for name, array in values.items():
                update_statistic(statistics[name], array)
            logical_errors["Tmin_gt_Tmean"] += int(
                np.sum(values["Tmin"] > values["Tmean"] + 1e-5)
            )
            logical_errors["Tmean_gt_Tmax"] += int(
                np.sum(values["Tmean"] > values["Tmax"] + 1e-5)
            )
            logical_errors["VPD_mean_gt_VPD_max"] += int(
                np.sum(values["VPD_mean"] > values["VPD_max"] + 1e-5)
            )
            logical_errors["negative_precipitation"] += int(
                np.sum(values["tp_sum"] < -1e-6)
            )
            logical_errors["negative_vpd"] += int(
                np.sum((values["VPD_mean"] < -1e-6) | (values["VPD_max"] < -1e-6))
            )
            logical_errors["negative_pet"] += int(np.sum(values["PET"] < -1e-6))

    finalized = finalize_statistics(statistics)
    for name, statistic in finalized.items():
        if statistic["non_finite"]:
            errors.append(
                f"{year}: {name} contains {statistic['non_finite']:,} non-finite values"
            )
    for name, count in logical_errors.items():
        if count:
            errors.append(f"{year}: {name} occurs {count:,} times")
    return finalized, logical_errors


def audit(args):
    errors = []
    if not args.selection.exists():
        raise FileNotFoundError(args.selection)
    with Dataset(args.selection) as selection:
        selection_summary = validate_selection(selection, errors)
        metadata = selection_metadata(selection)
        selection_attributes = {
            name: json_value(selection.getncattr(name)) for name in selection.ncattrs()
        }

    files = []
    yearly_statistics = {}
    for year in range(args.start_year, args.end_year + 1):
        path = args.daily_dir / f"ERA5_daily_pixelset_{year}.nc"
        if not path.exists():
            errors.append(f"Missing daily file: {path}")
            continue
        print(f"Validating {year}: {path.name}", flush=True)
        statistics, logical_errors = validate_year(
            path, year, metadata, args.site_batch, errors
        )
        record = {
            "year": year,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "features": statistics,
            "logical_errors": logical_errors,
        }
        if not args.skip_sha256:
            print(f"Hashing {year}: {path.name}", flush=True)
            record["sha256"] = sha256(path)
        files.append(record)
        yearly_statistics[str(year)] = statistics

    selection_record = {
        "path": str(args.selection.resolve()),
        "size_bytes": args.selection.stat().st_size,
    }
    if not args.skip_sha256:
        selection_record["sha256"] = sha256(args.selection)
    manifest = {
        "status": "passed" if not errors else "failed",
        "selection": selection_record,
        "selection_attributes": selection_attributes,
        "selection_summary": selection_summary,
        "years": [args.start_year, args.end_year],
        "feature_order": list(FINAL_VARIABLES),
        "feature_units": {name: units for name, (units, _) in FINAL_VARIABLES.items()},
        "daily_files": files,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    if errors:
        raise RuntimeError(
            f"ERA5 audit failed with {len(errors)} error(s); see {args.output}"
        )
    print(f"Audit passed: {args.output}", flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--site-batch", type=int, default=2048)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip full-file hashes (use only for development tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    if args.site_batch < 1:
        raise ValueError("--site-batch must be positive")
    audit(args)


if __name__ == "__main__":
    main()
