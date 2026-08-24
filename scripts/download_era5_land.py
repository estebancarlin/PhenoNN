#!/usr/bin/env python3
"""Download monthly ERA5-Land hourly NetCDF files through the CDS API."""

import argparse
import calendar
import datetime as dt
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from netCDF4 import Dataset

DATASET = "reanalysis-era5-land"
CDS_FIELD_LIMIT = 12_000
GRID_POINTS = 1801 * 3600
VARIABLES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "tp": "total_precipitation",
}
HOURS = [f"{hour:02d}:00" for hour in range(24)]


@dataclass(frozen=True)
class DownloadTask:
    variable: str
    year: int
    month: int
    area: tuple = ()

    @property
    def filename(self):
        return f"{self.variable}.{self.year}{self.month:02d}.nc"

    @property
    def days(self):
        count = calendar.monthrange(self.year, self.month)[1]
        return [f"{day:02d}" for day in range(1, count + 1)]

    @property
    def field_count(self):
        return len(self.days) * len(HOURS)

    def request(self):
        request = {
            "variable": [VARIABLES[self.variable]],
            "year": [str(self.year)],
            "month": [f"{self.month:02d}"],
            "day": self.days,
            "time": HOURS,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if self.area:
            request["area"] = list(self.area)
        return request


def build_tasks(start_year, end_year, variables, months=range(1, 13), area=()):
    return [
        DownloadTask(variable, year, month, tuple(area))
        for year in range(start_year, end_year + 1)
        for month in months
        for variable in variables
    ]


def expected_grid_shape(area):
    if not area:
        return 1801, 3600
    north, west, south, east = area
    latitude_count = round((north - south) / 0.1) + 1
    longitude_count = round((east - west) / 0.1) + 1
    return latitude_count, longitude_count


def validate_netcdf(path, task):
    expected_time = task.field_count
    expected_latitude, expected_longitude = expected_grid_shape(task.area)
    with Dataset(path) as dataset:
        time_name = "time" if "time" in dataset.dimensions else "valid_time"
        latitude_name = "latitude" if "latitude" in dataset.dimensions else "lat"
        longitude_name = "longitude" if "longitude" in dataset.dimensions else "lon"
        required_dimensions = {time_name, latitude_name, longitude_name}
        if not required_dimensions.issubset(dataset.dimensions):
            raise RuntimeError(f"Missing dimensions in {path.name}")
        actual_shape = (
            len(dataset.dimensions[time_name]),
            len(dataset.dimensions[latitude_name]),
            len(dataset.dimensions[longitude_name]),
        )
        expected_shape = (expected_time, expected_latitude, expected_longitude)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Unexpected dimensions in {path.name}: "
                f"{actual_shape} != {expected_shape}"
            )
        if task.variable not in dataset.variables:
            raise RuntimeError(f"Missing variable {task.variable!r} in {path.name}")
        dimensions = set(dataset.variables[task.variable].dimensions)
        if not required_dimensions.issubset(dimensions):
            raise RuntimeError(
                f"Unexpected {task.variable} dimensions in {path.name}: "
                f"{dataset.variables[task.variable].dimensions}"
            )


def estimate_uncompressed_bytes(tasks):
    total = 0
    for task in tasks:
        latitude_count, longitude_count = expected_grid_shape(task.area)
        total += task.field_count * latitude_count * longitude_count * 4
    return total


def format_bytes(value):
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise RuntimeError("unreachable")


def load_manifest(path):
    if not path.exists():
        return {"dataset": DATASET, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def task_record(task, destination):
    return {
        "task": asdict(task),
        "request": task.request(),
        "path": str(destination),
        "size": destination.stat().st_size,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def download_task(client, task, destination, attempts, retry_delay):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            validate_netcdf(destination, task)
            return "skipped"
        except (OSError, RuntimeError):
            invalid = destination.with_name(destination.name + ".invalid")
            invalid.unlink(missing_ok=True)
            destination.replace(invalid)

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            client.retrieve(DATASET, task.request(), str(partial))
            validate_netcdf(partial, task)
            partial.replace(destination)
            return "downloaded"
        except (Exception, KeyboardInterrupt) as error:
            partial.unlink(missing_ok=True)
            if isinstance(error, KeyboardInterrupt) or attempt == attempts:
                raise
            if "required licences not accepted" in str(error):
                raise RuntimeError(
                    "CDS rejected the request because the ERA5-Land licence "
                    "is not accepted for the API-key account. Accept it at "
                    "https://cds.climate.copernicus.eu/datasets/"
                    "reanalysis-era5-land?tab=download#manage-licences, "
                    "then retry."
                ) from error
            delay = min(15 * 60, retry_delay * 2 ** (attempt - 1))
            print(
                f"{timestamp()} retry {attempt}/{attempts} for {task.filename} "
                f"in {delay}s: {type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument(
        "--variables", nargs="+", choices=VARIABLES, default=list(VARIABLES)
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        choices=range(1, 13),
        default=list(range(1, 13)),
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Optional 0.1-degree-aligned bounding box.",
    )
    parser.add_argument(
        "--allow-global",
        action="store_true",
        help="Acknowledge the very large full-global download.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=60)
    return parser.parse_args(argv)


def validate_args(parser, args):
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.retry_delay < 0:
        parser.error("--retry-delay must not be negative")
    if args.area:
        north, west, south, east = args.area
        if not (-90 <= south <= north <= 90 and -180 <= west <= east <= 180):
            parser.error("--area must be NORTH WEST SOUTH EAST within global bounds")
        for coordinate in args.area:
            if not math.isclose(coordinate * 10, round(coordinate * 10), abs_tol=1e-8):
                parser.error("--area coordinates must align to the 0.1-degree grid")
    elif not args.allow_global and not args.dry_run:
        parser.error("full-global downloads require --allow-global")


def create_client():
    try:
        import cdsapi
    except ImportError as error:
        raise RuntimeError(
            "cdsapi is required; install it with python -m pip install 'cdsapi>=0.7.7'"
        ) from error
    return cdsapi.Client(
        quiet=False,
        progress=True,
        retry_max=10,
        sleep_max=120,
    )


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(add_help=False)
    args = parse_args(argv)
    validate_args(parser, args)
    tasks = build_tasks(
        args.start_year,
        args.end_year,
        args.variables,
        months=args.months,
        area=args.area or (),
    )
    largest_request = max(task.field_count for task in tasks)
    if largest_request > CDS_FIELD_LIMIT:
        raise RuntimeError(
            f"Request has {largest_request} fields; CDS limit is {CDS_FIELD_LIMIT}"
        )

    estimate = estimate_uncompressed_bytes(tasks)
    usage = shutil.disk_usage(args.output_dir.parent)
    print(f"Dataset        : {DATASET}")
    print(f"Years          : {args.start_year}-{args.end_year}")
    print(f"Variables      : {', '.join(args.variables)}")
    print(f"Area           : {args.area or 'global'}")
    print(f"Requests       : {len(tasks):,} (sequential)")
    print(f"Fields/request : <= {largest_request:,} / {CDS_FIELD_LIMIT:,}")
    print(f"Raw-size bound : {format_bytes(estimate)} before compression")
    print(f"Free disk      : {format_bytes(usage.free)}")
    if not args.area and estimate > usage.free:
        print(
            "WARNING: free disk is below the uncompressed global size estimate.",
            file=sys.stderr,
        )

    if args.dry_run:
        print(json.dumps(tasks[0].request(), indent=2))
        print(f"First file: {tasks[0].filename}")
        print(f"Last file : {tasks[-1].filename}")
        return 0

    if client is None:
        client = create_client()
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    started = time.monotonic()
    downloaded = skipped = 0
    for index, task in enumerate(tasks, start=1):
        destination = args.output_dir / task.variable / task.filename
        print(
            f"{timestamp()} [{index}/{len(tasks)}] start {task.filename} "
            f"({task.field_count} fields)",
            flush=True,
        )
        status = download_task(
            client,
            task,
            destination,
            attempts=args.max_attempts,
            retry_delay=args.retry_delay,
        )
        downloaded += status == "downloaded"
        skipped += status == "skipped"
        manifest["files"][task.filename] = task_record(task, destination)
        save_manifest(manifest_path, manifest)
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed else 0
        remaining = (len(tasks) - index) / rate if rate else 0
        print(
            f"{timestamp()} [{index}/{len(tasks)}] {status} {task.filename}; "
            f"downloaded={downloaded}, skipped={skipped}, "
            f"ETA={dt.timedelta(seconds=int(remaining))}",
            flush=True,
        )
    print(f"Complete: downloaded={downloaded}, skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
