#!/usr/bin/env python3
"""Download ORCHIDEE 15-PFT maps and TRENDYv2024 CO2 from IPSL."""

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from netCDF4 import Dataset

PFT_START_YEAR = 1992
PFT_END_YEAR = 2019
PFT_BASE_URL = (
    "https://thredds-su.ipsl.fr/thredds/fileServer/igcmg/IGCM/SRF/PFTMAPS/"
    "CMIP6/ESACCI-LC/15PFT.v2023.1/0.1"
)
CO2_FILENAME = "CO2_1700_2023_TRENDYv2024.txt"
CO2_URL = (
    f"https://thredds-su.ipsl.fr/thredds/fileServer/igcmg/IGCM/SRF/CO2/{CO2_FILENAME}"
)
CO2_RE = re.compile(r"^Annee_(\d{4})=([0-9]+(?:\.[0-9]+)?)$")


def open_with_retry(request, attempts=10):
    for attempt in range(attempts):
        try:
            return urlopen(request, timeout=180)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(300, 2**attempt)
        except URLError:
            if attempt == attempts - 1:
                raise
            delay = min(300, 2**attempt)
        time.sleep(delay + random.random())
    raise RuntimeError("unreachable")


def remote_size(url):
    with open_with_retry(Request(url, method="HEAD")) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value is not None else None


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_size(url)
    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return "skipped"

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and offset > expected_size:
        partial.unlink()
        offset = 0

    request = Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with open_with_retry(request) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            partial.unlink(missing_ok=True)
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)

    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected_size}"
        )
    partial.replace(destination)
    return "downloaded"


def validate_pft(path):
    expected_dimensions = {
        "time_counter": 1,
        "veget": 15,
        "lat": 1800,
        "lon": 3600,
    }
    with Dataset(path) as dataset:
        actual_dimensions = {
            name: len(dataset.dimensions[name])
            for name in expected_dimensions
            if name in dataset.dimensions
        }
        if actual_dimensions != expected_dimensions:
            raise RuntimeError(
                f"Unexpected dimensions in {path.name}: {actual_dimensions}"
            )
        if "maxvegetfrac" not in dataset.variables:
            raise RuntimeError(f"Missing maxvegetfrac in {path.name}")
        dimensions = dataset.variables["maxvegetfrac"].dimensions
        if dimensions != ("time_counter", "veget", "lat", "lon"):
            raise RuntimeError(
                f"Unexpected maxvegetfrac dimensions in {path.name}: {dimensions}"
            )


def parse_co2(path):
    values = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = CO2_RE.fullmatch(line.strip())
        if not match:
            raise RuntimeError(f"Invalid CO2 line {line_number}: {line!r}")
        values[int(match.group(1))] = float(match.group(2))
    expected_years = set(range(1700, 2024))
    if set(values) != expected_years:
        missing = sorted(expected_years - set(values))
        extra = sorted(set(values) - expected_years)
        raise RuntimeError(
            f"Invalid CO2 year coverage: missing={missing}, extra={extra}"
        )
    return values


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=PFT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=PFT_END_YEAR)
    parser.add_argument("--skip-co2", action="store_true")
    args = parser.parse_args()

    if not PFT_START_YEAR <= args.start_year <= args.end_year <= PFT_END_YEAR:
        parser.error(
            f"PFT years must satisfy {PFT_START_YEAR} <= start <= end <= {PFT_END_YEAR}"
        )

    records = []
    for year in range(args.start_year, args.end_year + 1):
        filename = f"PFTmap_{year}.nc"
        url = f"{PFT_BASE_URL}/{filename}"
        destination = args.output_dir / "pft" / filename
        status = download(url, destination)
        validate_pft(destination)
        records.append(
            {
                "kind": "pft",
                "year": year,
                "filename": str(destination.relative_to(args.output_dir)),
                "url": url,
                "size": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        print(f"{status}: {filename}", flush=True)

    if not args.skip_co2:
        destination = args.output_dir / "co2" / CO2_FILENAME
        status = download(CO2_URL, destination)
        parse_co2(destination)
        records.append(
            {
                "kind": "co2",
                "filename": str(destination.relative_to(args.output_dir)),
                "url": CO2_URL,
                "size": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        print(f"{status}: {CO2_FILENAME}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(args.output_dir / "manifest.json", records)
    print(f"Complete: {len(records)} files", flush=True)


if __name__ == "__main__":
    main()
