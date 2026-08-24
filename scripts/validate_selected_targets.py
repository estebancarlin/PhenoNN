#!/usr/bin/env python3
"""Validate selected-site GEOV2 LAI, annual PFT, and CO2 artifacts."""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def site_ids(dataset):
    return np.asarray(dataset.variables["site_id"][:]).astype(str)


def expected_dates(year):
    return np.asarray(
        [
            dt.date(year, month, day).isoformat()
            for month in range(1, 13)
            for day in (5, 15, 25)
        ]
    )


def validate_lai_arrays(values, target_valid, valid_count, min_valid_dekads):
    errors = []
    finite = np.isfinite(values)
    if not np.array_equal(finite, target_valid.astype(bool)):
        errors.append("target_valid does not match finite LAI values")
    if not np.array_equal(finite.sum(axis=0), valid_count):
        errors.append("valid_observation_count does not match target_valid")
    finite_values = values[finite]
    if finite_values.size and np.any((finite_values < 0) | (finite_values > 10)):
        errors.append("LAI values are outside [0, 10]")
    return errors, {
        "valid_values": int(finite.sum()),
        "all_missing_sites": int(np.sum(valid_count == 0)),
        "valid_site_years": int(np.sum(valid_count >= min_valid_dekads)),
        "minimum": float(finite_values.min()) if finite_values.size else None,
        "maximum": float(finite_values.max()) if finite_values.size else None,
    }


def validate_pft_arrays(fractions, pft_valid):
    errors = []
    if np.any(~np.isfinite(fractions)):
        errors.append("PFT fractions contain non-finite values")
    if np.any((fractions < 0) | (fractions > 1)):
        errors.append("PFT fractions are outside [0, 1]")
    totals = fractions.sum(axis=0, dtype=np.float32)
    expected_valid = totals > 0.5
    if not np.array_equal(expected_valid, pft_valid.astype(bool)):
        errors.append("pft_valid does not match PFT fraction sums")
    if np.any(np.abs(totals - 1.0) > 1e-4):
        errors.append("PFT fractions do not sum to one")
    return errors, {
        "valid_sites": int(expected_valid.sum()),
        "minimum_fraction": float(fractions.min()),
        "maximum_fraction": float(fractions.max()),
        "minimum_sum": float(totals.min()),
        "maximum_sum": float(totals.max()),
    }


def audit(args):
    errors = []
    with Dataset(args.selection) as selection:
        reference_ids = site_ids(selection)
    if np.unique(reference_ids).size != reference_ids.size:
        errors.append("Canonical selection site IDs are not unique")

    records = []
    total_valid_site_years = 0
    for year in range(args.start_year, args.end_year + 1):
        lai_path = args.target_dir / "lai" / f"LAI_dekadal_{year}.nc"
        pft_path = args.target_dir / "pft" / f"PFTmap_{year}.nc"
        if not lai_path.exists() or not pft_path.exists():
            errors.append(f"{year}: missing LAI or PFT file")
            continue
        with Dataset(lai_path) as lai:
            if not np.array_equal(site_ids(lai), reference_ids):
                errors.append(f"{year}: LAI site order differs from selection")
            dates = np.asarray(lai.variables["date"][:]).astype(str)
            if not np.array_equal(dates, expected_dates(year)):
                errors.append(f"{year}: LAI dates are not the expected 36 dates")
            values = np.ma.filled(lai.variables["LAI"][:], np.nan)
            target_valid = lai.variables["target_valid"][:]
            valid_count = lai.variables["valid_observation_count"][:]
            lai_errors, lai_summary = validate_lai_arrays(
                values, target_valid, valid_count, args.min_valid_dekads
            )
            errors.extend(f"{year}: {error}" for error in lai_errors)
        with Dataset(pft_path) as pft:
            if not np.array_equal(site_ids(pft), reference_ids):
                errors.append(f"{year}: PFT site order differs from selection")
            pft_numbers = pft.variables["pft"][:]
            if not np.array_equal(pft_numbers, np.arange(1, 16)):
                errors.append(f"{year}: PFT numbering is not 1..15")
            fractions = np.ma.filled(pft.variables["pft_frac"][:], np.nan)
            pft_errors, pft_summary = validate_pft_arrays(
                fractions, pft.variables["pft_valid"][:]
            )
            errors.extend(f"{year}: {error}" for error in pft_errors)
        total_valid_site_years += lai_summary["valid_site_years"]
        records.append({"year": year, "lai": lai_summary, "pft": pft_summary})
        print(
            f"{year}: {lai_summary['valid_site_years']:,} valid LAI site-years, "
            f"{pft_summary['valid_sites']:,} valid PFT sites",
            flush=True,
        )

    co2_path = args.target_dir / "CO2_annual.nc"
    if not co2_path.exists():
        errors.append("Missing CO2_annual.nc")
        co2_summary = {}
    else:
        with Dataset(co2_path) as co2:
            years = co2.variables["year"][:]
            values = co2.variables["co2"][:]
        expected_years = np.arange(args.co2_start_year, args.end_year + 1)
        if not np.array_equal(years, expected_years):
            errors.append("CO2 years do not match required coverage")
        if np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0):
            errors.append("CO2 values must be finite and strictly increasing")
        co2_summary = {
            "years": [int(years[0]), int(years[-1])],
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    manifest = {
        "status": "passed" if not errors else "failed",
        "selection": str(args.selection.resolve()),
        "n_site": int(reference_ids.size),
        "years": [args.start_year, args.end_year],
        "total_valid_site_years": total_valid_site_years,
        "co2": co2_summary,
        "records": records,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    if errors:
        raise RuntimeError(
            f"Target audit failed with {len(errors)} error(s); see {args.output}"
        )
    print(f"Audit passed: {args.output}", flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1992)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--co2-start-year", type=int, default=1991)
    parser.add_argument("--min-valid-dekads", type=int, default=26)
    return parser.parse_args()


def main():
    args = parse_args()
    audit(args)


if __name__ == "__main__":
    main()
