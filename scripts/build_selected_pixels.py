#!/usr/bin/env python3
"""Build PhenoNN selected pixels from compressed GEOV2 and ORCHIDEE PFT maps."""

import argparse
import datetime as dt
import gzip
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset

NATIVE_SHAPE = (3600, 7200)
GRID_SHAPE = (1800, 3600)
DEKAD_DAYS = (5, 15, 25)
FILENAME_RE = re.compile(r"THEIA_GEOV2_R01_AVHRR_LAI_(\d{8})\.h5\.gz$")


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def expected_dates(year):
    return [
        f"{year:04d}{month:02d}{day:02d}"
        for month in range(1, 13)
        for day in DEKAD_DAYS
    ]


def scan_geov2(directory):
    files = {}
    for path in sorted(directory.glob("THEIA_GEOV2_R01_AVHRR_LAI_*.h5.gz")):
        match = FILENAME_RE.fullmatch(path.name)
        if match:
            files[match.group(1)] = path
    return files


def coarse_valid(raw, min_valid_native=2):
    if raw.shape != NATIVE_SHAPE:
        raise RuntimeError(f"Unexpected GEOV2 shape: {raw.shape} != {NATIVE_SHAPE}")
    counts = np.zeros(GRID_SHAPE, dtype=np.uint8)
    for row_offset in (0, 1):
        for column_offset in (0, 1):
            values = raw[row_offset::2, column_offset::2]
            counts += (values != 255) & (values != 256)
    return counts >= min_valid_native


def read_geov2_valid(path, min_valid_native=2):
    with gzip.open(path, "rb") as compressed, h5py.File(compressed, "r") as source:
        if "LAI" not in source:
            raise RuntimeError(f"Missing LAI dataset in {path.name}")
        raw = source["LAI"][:]
    return coarse_valid(raw, min_valid_native=min_valid_native)


def read_pft_land(path, threshold):
    with Dataset(path) as source:
        if "maxvegetfrac" not in source.variables:
            raise RuntimeError(f"Missing maxvegetfrac in {path.name}")
        variable = source.variables["maxvegetfrac"]
        if variable.shape != (1, 15, *GRID_SHAPE):
            raise RuntimeError(f"Unexpected PFT shape in {path.name}: {variable.shape}")
        total = np.zeros(GRID_SHAPE, dtype=np.float32)
        for pft_index in range(15):
            values = np.ma.filled(variable[0, pft_index, :, :], 0.0)
            total += values.astype(np.float32, copy=False)
        latitude = source.variables["lat"][:].astype(np.float32)
        longitude = source.variables["lon"][:].astype(np.float32)
    return total > threshold, latitude, longitude


def process_year(
    year,
    geov2_files,
    pft_dir,
    cache_dir,
    min_valid_native,
    min_valid_dekads,
    pft_threshold,
):
    destination = cache_dir / f"valid_{year}.npy"
    if destination.exists():
        valid = np.load(destination, mmap_mode="r")
        if valid.shape == GRID_SHAPE and valid.dtype == np.bool_:
            print(f"{timestamp()} skipped cached year {year}", flush=True)
            return destination
        destination.replace(destination.with_suffix(".npy.invalid"))

    dates = expected_dates(year)
    missing = [date for date in dates if date not in geov2_files]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} GEOV2 files for {year}: {missing}")
    pft_path = pft_dir / f"PFTmap_{year}.nc"
    if not pft_path.exists():
        raise FileNotFoundError(pft_path)

    started = time.monotonic()
    valid_dekads = np.zeros(GRID_SHAPE, dtype=np.uint8)
    for index, date in enumerate(dates, start=1):
        valid_dekads += read_geov2_valid(
            geov2_files[date], min_valid_native=min_valid_native
        )
        if index == 1 or index % 6 == 0:
            elapsed = dt.timedelta(seconds=int(time.monotonic() - started))
            print(
                f"{timestamp()} {year}: GEOV2 {index}/36, elapsed={elapsed}",
                flush=True,
            )

    land, _, _ = read_pft_land(pft_path, pft_threshold)
    valid = land & (valid_dekads >= min_valid_dekads)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npy.part")
    with temporary.open("wb") as stream:
        np.save(stream, valid, allow_pickle=False)
    temporary.replace(destination)
    print(
        f"{timestamp()} {year}: {int(valid.sum()):,}/{valid.size:,} valid pixels",
        flush=True,
    )
    return destination


def era_coordinates(lat_idx, lon_idx):
    latitude = 90.0 - lat_idx.astype(np.float64) * 0.1
    longitude_0360 = ((lon_idx.astype(np.int64) + 1800) % 3600) * 0.1
    longitude = np.where(longitude_0360 > 180.0, longitude_0360 - 360.0, longitude_0360)
    return latitude.astype(np.float32), longitude.astype(np.float32)


def write_selected_pixels(
    output_path,
    qualifying,
    pft_latitude,
    pft_longitude,
    fraction,
    min_years,
    seed,
    years,
):
    flat = np.flatnonzero(qualifying)
    sample_count = round(flat.size * fraction)
    if sample_count < 1:
        raise RuntimeError("Selection produced no pixels")
    rng = np.random.default_rng(seed)
    selected_flat = np.sort(rng.choice(flat, size=sample_count, replace=False))
    lat_idx, lon_idx = np.unravel_index(selected_flat, GRID_SHAPE)
    lat_idx = lat_idx.astype(np.int32)
    lon_idx = lon_idx.astype(np.int32)
    era_latitude, era_longitude = era_coordinates(lat_idx, lon_idx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("site", sample_count)
        output.createVariable("lat_idx", "i4", ("site",), zlib=True)[:] = lat_idx
        output.createVariable("lon_idx", "i4", ("site",), zlib=True)[:] = lon_idx
        output.createVariable("latitude", "f4", ("site",), zlib=True)[:] = pft_latitude[
            lat_idx
        ]
        output.createVariable("longitude", "f4", ("site",), zlib=True)[:] = (
            pft_longitude[lon_idx]
        )
        output.createVariable("era_latitude", "f4", ("site",), zlib=True)[:] = (
            era_latitude
        )
        output.createVariable("era_longitude", "f4", ("site",), zlib=True)[:] = (
            era_longitude
        )
        site_id = output.createVariable("site_id", str, ("site",))
        site_id[:] = np.asarray(
            [f"pix_{row:04d}_{column:05d}" for row, column in zip(lat_idx, lon_idx)],
            dtype=object,
        )
        output.fraction = float(fraction)
        output.seed = int(seed)
        output.min_years = int(min_years)
        output.n_qualifying = int(flat.size)
        output.n_selected = int(sample_count)
        output.years = f"{min(years)}-{max(years)}"
        output.source = "scripts/build_selected_pixels.py"
    temporary.replace(output_path)
    return sample_count, flat.size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geov2-dir", type=Path, required=True)
    parser.add_argument("--pft-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="selected_pixels.nc")
    parser.add_argument("--start-year", type=int, default=1992)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--min-valid-native", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--min-valid-dekads", type=int, default=26)
    parser.add_argument("--pft-threshold", type=float, default=0.5)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--min-years", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")
    years = list(range(args.start_year, args.end_year + 1))
    if not 1 <= args.min_years <= len(years):
        raise ValueError("--min-years must be between 1 and the number of years")

    geov2_files = scan_geov2(args.geov2_dir)
    print(f"Indexed {len(geov2_files):,} compressed GEOV2 R01 files", flush=True)
    cache_dir = args.output_dir / "valid_masks"
    paths = []
    for year in years:
        paths.append(
            process_year(
                year,
                geov2_files,
                args.pft_dir,
                cache_dir,
                args.min_valid_native,
                args.min_valid_dekads,
                args.pft_threshold,
            )
        )

    counter = np.zeros(GRID_SHAPE, dtype=np.uint8)
    for path in paths:
        counter += np.load(path, mmap_mode="r")
    qualifying = counter >= args.min_years
    _, pft_latitude, pft_longitude = read_pft_land(
        args.pft_dir / f"PFTmap_{years[0]}.nc", args.pft_threshold
    )
    output_path = args.output_dir / args.output_name
    selected_count, qualifying_count = write_selected_pixels(
        output_path,
        qualifying,
        pft_latitude,
        pft_longitude,
        args.fraction,
        args.min_years,
        args.seed,
        years,
    )
    metadata = {
        "years": years,
        "min_valid_native": args.min_valid_native,
        "min_valid_dekads": args.min_valid_dekads,
        "pft_threshold": args.pft_threshold,
        "fraction": args.fraction,
        "min_years": args.min_years,
        "seed": args.seed,
        "n_qualifying": qualifying_count,
        "n_selected": selected_count,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected {selected_count:,}/{qualifying_count:,} pixels -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
