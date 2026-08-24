#!/usr/bin/env python3
"""Build selected-site GEOV2 LAI, annual PFT, and CO2 artifacts."""

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset

from scripts.build_selected_pixels import expected_dates, scan_geov2
from scripts.download_ipsl_forcings import parse_co2


SITE_METADATA = (
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
    "dominant_pft",
    "mixed_pft",
    "lai_bin",
    "lai_mean",
    "lai_amplitude",
    "sample_weight",
    "inclusion_probability",
    "source_site_index",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def selected_sites(path):
    with Dataset(path) as source:
        missing = [name for name in SITE_METADATA if name not in source.variables]
        if missing:
            raise RuntimeError(f"Selection is missing metadata variables: {missing}")
        metadata = {name: source.variables[name][:] for name in SITE_METADATA}
    site_id = np.asarray(metadata["site_id"]).astype(str)
    if np.unique(site_id).size != site_id.size:
        raise RuntimeError("Selection site IDs are not unique")
    metadata["site_id"] = site_id
    return metadata


def copy_site_metadata(output, metadata):
    for name in SITE_METADATA:
        values = metadata[name]
        if name == "site_id":
            variable = output.createVariable(name, str, ("site",))
            variable[:] = np.asarray(values, dtype=object)
            continue
        dtype = values.dtype
        variable = output.createVariable(name, dtype, ("site",), zlib=True)
        variable[:] = values


def extract_lai(path, rows, columns, min_valid_native):
    native_rows = rows.astype(np.int64) * 2
    native_columns = columns.astype(np.int64) * 2
    total = np.zeros(rows.size, dtype=np.float32)
    count = np.zeros(rows.size, dtype=np.uint8)
    with gzip.open(path, "rb") as compressed:
        with h5py.File(compressed, "r") as source:
            if "LAI" not in source:
                raise RuntimeError(f"Missing LAI dataset in {path.name}")
            raw = source["LAI"][:]
    for row_offset in (0, 1):
        for column_offset in (0, 1):
            values = raw[native_rows + row_offset, native_columns + column_offset]
            valid = (values != 255) & (values != 256)
            total += np.where(valid, values.astype(np.float32) / 30.0, 0.0)
            count += valid
    return np.where(count >= min_valid_native, total / np.maximum(count, 1), np.nan)


def dates_for_year(year):
    dates = [dt.date(year, month, day) for month in range(1, 13) for day in (5, 15, 25)]
    return dates


def write_lai_year(
    path,
    year,
    geov2_files,
    metadata,
    min_valid_native,
    min_valid_dekads,
    selection_hash,
):
    rows = metadata["lat_idx"].astype(np.int64)
    columns = metadata["lon_idx"].astype(np.int64)
    values = np.empty((36, rows.size), dtype=np.float32)
    for index, date_string in enumerate(expected_dates(year)):
        source_path = geov2_files.get(date_string)
        if source_path is None:
            raise RuntimeError(f"Missing GEOV2 R01 file for {date_string}")
        values[index] = extract_lai(
            source_path, rows, columns, min_valid_native=min_valid_native
        )
        if index == 0 or (index + 1) % 6 == 0:
            print(f"  LAI {year}: {index + 1}/36", flush=True)

    valid = np.isfinite(values)
    valid_count = valid.sum(axis=0).astype(np.int8)
    dates = dates_for_year(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("dekad", 36)
        output.createDimension("site", rows.size)
        output.createVariable("dekad", "i2", ("dekad",))[:] = np.arange(36)
        output.createVariable("month", "i1", ("dekad",))[:] = [
            date.month for date in dates
        ]
        output.createVariable("day", "i1", ("dekad",))[:] = [date.day for date in dates]
        output.createVariable("doy", "i2", ("dekad",))[:] = [
            date.timetuple().tm_yday for date in dates
        ]
        date_variable = output.createVariable("date", str, ("dekad",))
        date_variable[:] = np.asarray(
            [date.isoformat() for date in dates], dtype=object
        )
        copy_site_metadata(output, metadata)
        lai = output.createVariable(
            "LAI",
            "f4",
            ("dekad", "site"),
            zlib=True,
            complevel=4,
            chunksizes=(36, min(512, rows.size)),
            fill_value=np.float32(np.nan),
        )
        lai[:] = values
        lai.units = "m2 m-2"
        lai.long_name = "Leaf Area Index"
        lai.source = "THEIA GEOV2 R01 AVHRR, 0.05 degree aggregated 2x2"
        output.createVariable("target_valid", "i1", ("dekad", "site"), zlib=True)[:] = (
            valid.astype(np.int8)
        )
        output.createVariable("valid_observation_count", "i1", ("site",), zlib=True)[
            :
        ] = valid_count
        output.createVariable("target_year_valid", "i1", ("site",), zlib=True)[:] = (
            valid_count >= min_valid_dekads
        ).astype(np.int8)
        output.year = year
        output.min_valid_native = min_valid_native
        output.min_valid_dekads = min_valid_dekads
        output.selection_sha256 = selection_hash
        output.source = "scripts/build_selected_targets.py"
    temporary.replace(path)
    return {
        "valid_values": int(valid.sum()),
        "all_missing_sites": int(np.sum(valid_count == 0)),
        "valid_site_years": int(np.sum(valid_count >= min_valid_dekads)),
        "lai_min": float(np.nanmin(values)),
        "lai_max": float(np.nanmax(values)),
    }


def write_pft_year(path, year, source_path, metadata, selection_hash):
    rows = metadata["lat_idx"].astype(np.int64)
    columns = metadata["lon_idx"].astype(np.int64)
    fractions = np.empty((15, rows.size), dtype=np.float32)
    with Dataset(source_path) as source:
        if "maxvegetfrac" not in source.variables:
            raise RuntimeError(f"Missing maxvegetfrac in {source_path.name}")
        variable = source.variables["maxvegetfrac"]
        if variable.shape[1] != 15:
            raise RuntimeError(
                f"Unexpected PFT shape in {source_path.name}: {variable.shape}"
            )
        for pft in range(15):
            layer = np.ma.filled(variable[0, pft, :, :], 0.0)
            fractions[pft] = layer[rows, columns]
    if np.any(~np.isfinite(fractions)):
        raise RuntimeError(f"Non-finite PFT fractions in {source_path.name}")
    if np.any((fractions < 0) | (fractions > 1)):
        raise RuntimeError(f"PFT fractions outside [0, 1] in {source_path.name}")
    total = fractions.sum(axis=0, dtype=np.float32)
    valid = total > 0.5

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("pft", 15)
        output.createDimension("site", rows.size)
        pft = output.createVariable("pft", "i1", ("pft",))
        pft[:] = np.arange(1, 16)
        pft.long_name = "ORCHIDEE plant functional type number"
        copy_site_metadata(output, metadata)
        variable = output.createVariable(
            "pft_frac",
            "f4",
            ("pft", "site"),
            zlib=True,
            complevel=4,
            chunksizes=(15, min(512, rows.size)),
        )
        variable[:] = fractions
        variable.units = "1"
        variable.long_name = "ORCHIDEE plant functional type fraction"
        output.createVariable("vegetated_fraction", "f4", ("site",), zlib=True)[:] = (
            total
        )
        output.createVariable("pft_valid", "i1", ("site",), zlib=True)[:] = (
            valid.astype(np.int8)
        )
        output.year = year
        output.selection_sha256 = selection_hash
        output.source_file = str(source_path.resolve())
        output.source = "scripts/build_selected_targets.py"
    temporary.replace(path)
    return {
        "valid_sites": int(valid.sum()),
        "invalid_sites": int((~valid).sum()),
        "fraction_min": float(fractions.min()),
        "fraction_max": float(fractions.max()),
        "sum_min": float(total.min()),
        "sum_max": float(total.max()),
    }


def write_co2(path, source_path, start_year, end_year):
    source_values = parse_co2(source_path)
    years = np.arange(start_year, end_year + 1, dtype=np.int16)
    values = np.asarray([source_values[int(year)] for year in years], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("year", years.size)
        output.createVariable("year", "i2", ("year",))[:] = years
        co2 = output.createVariable("co2", "f4", ("year",))
        co2[:] = values
        co2.units = "ppm"
        co2.long_name = "Annual atmospheric carbon dioxide concentration"
        output.source_file = str(source_path.resolve())
        output.source = "TRENDYv2024"
    temporary.replace(path)
    return {str(int(year)): float(value) for year, value in zip(years, values)}


def valid_existing(path, expected_variable, year, n_site, selection_hash):
    if not path.exists():
        return False
    try:
        with Dataset(path) as source:
            return (
                expected_variable in source.variables
                and len(source.dimensions["site"]) == n_site
                and int(source.year) == year
                and source.selection_sha256 == selection_hash
            )
    except (OSError, KeyError, AttributeError):
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--geov2-dir", type=Path, required=True)
    parser.add_argument("--pft-dir", type=Path, required=True)
    parser.add_argument("--co2-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1992)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--co2-start-year", type=int, default=1991)
    parser.add_argument("--min-valid-native", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--min-valid-dekads", type=int, default=26)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    metadata = selected_sites(args.selection)
    selection_hash = sha256(args.selection)
    geov2_files = scan_geov2(args.geov2_dir)
    if not geov2_files:
        raise RuntimeError(f"No GEOV2 R01 files found under {args.geov2_dir}")
    records = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"Processing {year}", flush=True)
        lai_path = args.output_dir / "lai" / f"LAI_dekadal_{year}.nc"
        if valid_existing(
            lai_path, "LAI", year, len(metadata["site_id"]), selection_hash
        ):
            lai_status = "skipped"
            lai_summary = {}
        else:
            lai_status = "written"
            lai_summary = write_lai_year(
                lai_path,
                year,
                geov2_files,
                metadata,
                args.min_valid_native,
                args.min_valid_dekads,
                selection_hash,
            )
        pft_source = args.pft_dir / f"PFTmap_{year}.nc"
        if not pft_source.exists():
            raise FileNotFoundError(pft_source)
        pft_path = args.output_dir / "pft" / f"PFTmap_{year}.nc"
        if valid_existing(
            pft_path, "pft_frac", year, len(metadata["site_id"]), selection_hash
        ):
            pft_status = "skipped"
            pft_summary = {}
        else:
            pft_status = "written"
            pft_summary = write_pft_year(
                pft_path, year, pft_source, metadata, selection_hash
            )
        records.append(
            {
                "year": year,
                "lai": {
                    "path": str(lai_path.resolve()),
                    "status": lai_status,
                    "size_bytes": lai_path.stat().st_size,
                    "sha256": sha256(lai_path),
                    **lai_summary,
                },
                "pft": {
                    "path": str(pft_path.resolve()),
                    "status": pft_status,
                    "size_bytes": pft_path.stat().st_size,
                    "sha256": sha256(pft_path),
                    **pft_summary,
                },
            }
        )

    co2_path = args.output_dir / "CO2_annual.nc"
    co2_values = write_co2(co2_path, args.co2_file, args.co2_start_year, args.end_year)
    manifest = {
        "selection": str(args.selection.resolve()),
        "selection_sha256": selection_hash,
        "n_site": len(metadata["site_id"]),
        "years": [args.start_year, args.end_year],
        "min_valid_native": args.min_valid_native,
        "min_valid_dekads": args.min_valid_dekads,
        "co2": {
            "path": str(co2_path.resolve()),
            "sha256": sha256(co2_path),
            "values": co2_values,
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "target_manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
