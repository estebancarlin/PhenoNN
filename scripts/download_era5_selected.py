#!/usr/bin/env python3
"""Stream ERA5-Land ARCO data for selected pixels and write daily features."""

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

ARCO_GROUPS = {
    "temperature": {
        "url": (
            "https://arco.datastores.ecmwf.int/cadl-arco-geo-007/arco/"
            "reanalysis_era5_land/sfc-2m-temperature/geoChunked.zarr"
        ),
        "variables": {"t2m": "instant", "d2m": "instant"},
    },
    "pressure_precipitation": {
        "url": (
            "https://arco.datastores.ecmwf.int/cadl-arco-geo-009/arco/"
            "reanalysis_era5_land/sfc-pressure-precipitation/geoChunked.zarr"
        ),
        "variables": {"sp": "instant", "tp": "accum"},
    },
    "radiation": {
        "url": (
            "https://arco.datastores.ecmwf.int/cadl-arco-geo-010/arco/"
            "reanalysis_era5_land/sfc-radiation-heat/geoChunked.zarr"
        ),
        "variables": {"ssrd": "accum", "strd": "accum"},
    },
}
STAGING_VARIABLES = {
    "Tmin": ("degC", "Daily minimum 2 m air temperature"),
    "Tmax": ("degC", "Daily maximum 2 m air temperature"),
    "Tmean": ("degC", "Daily mean 2 m air temperature"),
    "Tdew_mean": ("degC", "Daily mean 2 m dew-point temperature"),
    "sp_mean": ("hPa", "Daily mean surface pressure"),
    "ssrd_sum": ("MJ m-2", "Daily downward short-wave radiation"),
    "strd_sum": ("MJ m-2", "Daily downward long-wave radiation"),
    "tp_sum": ("mm", "Daily total precipitation"),
}
FINAL_VARIABLES = {
    "Tmin": STAGING_VARIABLES["Tmin"],
    "Tmax": STAGING_VARIABLES["Tmax"],
    "Tmean": STAGING_VARIABLES["Tmean"],
    "ssrd_sum": STAGING_VARIABLES["ssrd_sum"],
    "strd_sum": STAGING_VARIABLES["strd_sum"],
    "tp_sum": STAGING_VARIABLES["tp_sum"],
    "VPD_max": ("kPa", "Daily maximum vapour pressure deficit"),
    "VPD_mean": ("kPa", "Daily mean vapour pressure deficit"),
    "Rn_tot": ("MJ m-2 day-1", "Daily total net radiation"),
    "PET": ("mm day-1", "Priestley-Taylor potential evapotranspiration"),
    "SMI": ("mm", "Thirty-day precipitation-derived soil-moisture proxy"),
}


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_cds_key():
    path = Path.home() / ".cdsapirc"
    if not path.exists():
        raise RuntimeError(f"Missing CDS configuration: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("key:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    raise RuntimeError(f"Missing key field in {path}")


def selected_to_arco_indices(latitude, longitude):
    latitude = np.asarray(np.ma.filled(latitude, np.nan), dtype=np.float64)
    longitude = np.asarray(np.ma.filled(longitude, np.nan), dtype=np.float64)
    lat_idx = np.rint((latitude + 90.0) * 10.0).astype(np.int32)
    lon_idx = np.rint((longitude + 179.9) * 10.0).astype(np.int32)
    if np.any((lat_idx < 0) | (lat_idx > 1800)):
        raise RuntimeError("Selected ERA latitude is outside the ARCO grid")
    if np.any((lon_idx < 0) | (lon_idx > 3599)):
        raise RuntimeError("Selected ERA longitude is outside the ARCO grid")
    return lat_idx, lon_idx


def spatial_batches(lat_idx, lon_idx, chunks_per_batch):
    chunk_id = (lat_idx // 4).astype(np.int64) * 450 + lon_idx // 8
    unique_chunks = np.unique(chunk_id)
    batches = []
    for start in range(0, unique_chunks.size, chunks_per_batch):
        ids = unique_chunks[start : start + chunks_per_batch]
        site_indices = np.flatnonzero(np.isin(chunk_id, ids)).astype(np.int32)
        batches.append(site_indices)
    return batches, unique_chunks.size


def aggregate_instant(times, values):
    hours = times.astype("datetime64[h]")
    if hours[0].astype("datetime64[h]").astype(object).hour != 0:
        raise RuntimeError("Instantaneous ERA5 chunk is not aligned to full days")
    if np.any(np.diff(hours).astype("timedelta64[h]").astype(np.int64) != 1):
        raise RuntimeError("Instantaneous ERA5 timestamps are not hourly-contiguous")
    complete_hours = hours.size // 24 * 24
    hours = hours[:complete_hours]
    values = values[:complete_hours]
    reshaped = values.reshape(-1, 24, values.shape[1])
    days = hours[::24].astype("datetime64[D]")
    return (
        days,
        np.nanmin(reshaped, axis=1),
        np.nanmax(reshaped, axis=1),
        np.nanmean(reshaped, axis=1),
    )


def aggregate_accum(times, values):
    shifted_days = (times.astype("datetime64[h]") - np.timedelta64(1, "h")).astype(
        "datetime64[D]"
    )
    starts = np.r_[0, np.flatnonzero(shifted_days[1:] != shifted_days[:-1]) + 1]
    days = shifted_days[starts]
    totals = np.add.reduceat(values, starts, axis=0)
    return days, totals


def target_days(start_year, end_year):
    return np.arange(
        np.datetime64(f"{start_year}-01-01"),
        np.datetime64(f"{end_year + 1}-01-01"),
        dtype="datetime64[D]",
    )


def copy_selection_metadata(selected, output, site_indices=None):
    """Copy site-level sampling metadata into an ERA output file."""
    for dimension_name, dimension in selected.dimensions.items():
        if dimension_name != "site" and dimension_name not in output.dimensions:
            output.createDimension(dimension_name, len(dimension))
    for name, source in selected.variables.items():
        if "site" not in source.dimensions or name in output.variables:
            continue
        if source.datatype is str:
            destination = output.createVariable(name, str, source.dimensions)
        else:
            fill_value = (
                source.getncattr("_FillValue")
                if "_FillValue" in source.ncattrs()
                else None
            )
            options = {"zlib": True}
            if fill_value is not None:
                options["fill_value"] = fill_value
            destination = output.createVariable(
                name, source.datatype, source.dimensions, **options
            )
        attributes = {
            attribute: source.getncattr(attribute)
            for attribute in source.ncattrs()
            if attribute != "_FillValue"
        }
        if attributes:
            destination.setncatts(attributes)
        values = source[:]
        if site_indices is not None:
            values = np.take(values, site_indices, axis=source.dimensions.index("site"))
        destination[:] = values
    selection_attributes = {
        f"selection_{name}": selected.getncattr(name) for name in selected.ncattrs()
    }
    if selection_attributes:
        output.setncatts(selection_attributes)


def initialize_staging(path, days, selected_path, overwrite=False):
    if path.exists() and not overwrite:
        return
    path.unlink(missing_ok=True)
    with Dataset(selected_path) as selected:
        n_site = len(selected.dimensions["site"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(path, "w", format="NETCDF4") as output:
            output.createDimension("time", days.size)
            output.createDimension("site", n_site)
            time_var = output.createVariable("time", "i4", ("time",))
            time_var.units = "days since 1970-01-01 00:00:00"
            time_var.calendar = "proleptic_gregorian"
            time_var[:] = (days - np.datetime64("1970-01-01")).astype(np.int32)
            copy_selection_metadata(selected, output)
            for name, (units, long_name) in STAGING_VARIABLES.items():
                variable = output.createVariable(
                    name,
                    "f4",
                    ("time", "site"),
                    zlib=True,
                    complevel=4,
                    chunksizes=(min(366, days.size), min(256, n_site)),
                    fill_value=np.float32(np.nan),
                )
                variable.units = units
                variable.long_name = long_name
            output.title = "ERA5-Land daily staging data for selected PhenoNN pixels"
            output.source = "Copernicus ERA5-Land ARCO geo-chunked stores"


def load_manifest(path):
    if not path.exists():
        return {"completed": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def time_chunk_slices(dataset, start_year, end_year, include_next_midnight):
    chunks = dataset.chunks["time"]
    offsets = np.cumsum((0, *chunks))
    times = dataset.time.values
    lower = np.datetime64(f"{start_year}-01-01T00:00:00")
    upper = np.datetime64(f"{end_year}-12-31T23:00:00")
    if include_next_midnight:
        upper = np.datetime64(f"{end_year + 1}-01-01T00:00:00")
    slices = []
    for chunk_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        left = max(start, int(np.searchsorted(times, lower, side="left")))
        right = min(stop, int(np.searchsorted(times, upper, side="right")))
        if left < right:
            slices.append((chunk_index, slice(left, right)))
    return slices


def write_daily_values(staging, name, days, site_indices, values, all_days, add=False):
    keep = (days >= all_days[0]) & (days <= all_days[-1])
    if not np.any(keep):
        return
    days = days[keep]
    values = values[keep].astype(np.float32, copy=False)
    indices = (days - all_days[0]).astype(np.int64)
    if np.any(np.diff(indices) != 1):
        raise RuntimeError(f"Non-contiguous daily output for {name}")
    day_slice = slice(int(indices[0]), int(indices[-1]) + 1)
    variable = staging.variables[name]
    if add:
        existing = np.ma.filled(variable[day_slice, site_indices], 0.0)
        variable[day_slice, site_indices] = existing + values
    else:
        variable[day_slice, site_indices] = values


def process_loaded_chunk(staging, dataset, kinds, site_indices, all_days):
    times = dataset.time.values
    if "t2m" in dataset:
        days, minimum, maximum, mean = aggregate_instant(times, dataset.t2m.values)
        write_daily_values(
            staging, "Tmin", days, site_indices, minimum - 273.15, all_days
        )
        write_daily_values(
            staging, "Tmax", days, site_indices, maximum - 273.15, all_days
        )
        write_daily_values(
            staging, "Tmean", days, site_indices, mean - 273.15, all_days
        )
    if "d2m" in dataset:
        days, _, _, mean = aggregate_instant(times, dataset.d2m.values)
        write_daily_values(
            staging, "Tdew_mean", days, site_indices, mean - 273.15, all_days
        )
    if "sp" in dataset:
        days, _, _, mean = aggregate_instant(times, dataset.sp.values)
        write_daily_values(
            staging, "sp_mean", days, site_indices, mean / 100.0, all_days
        )
    conversions = {
        "tp": ("tp_sum", 1000.0),
        "ssrd": ("ssrd_sum", 1e-6),
        "strd": ("strd_sum", 1e-6),
    }
    for source_name, (target_name, factor) in conversions.items():
        if source_name in dataset:
            days, totals = aggregate_accum(times, dataset[source_name].values)
            write_daily_values(
                staging,
                target_name,
                days,
                site_indices,
                totals * factor,
                all_days,
                add=True,
            )


def open_arco(group, key):
    return xr.open_zarr(
        ARCO_GROUPS[group]["url"],
        consolidated=True,
        storage_options={"headers": {"Authorization": f"Bearer {key}"}},
    )


def validate_arco_coordinates(dataset, lat_idx, lon_idx, latitude, longitude):
    actual_latitude = dataset.latitude.values[lat_idx]
    actual_longitude = dataset.longitude.values[lon_idx]
    if not np.allclose(actual_latitude, latitude, atol=1e-6):
        raise RuntimeError("Selected latitude does not map exactly to ARCO")
    if not np.allclose(actual_longitude, longitude, atol=1e-6):
        raise RuntimeError("Selected longitude does not map exactly to ARCO")


def download_to_staging(args, selected, all_days):
    key = read_cds_key()
    latitude = selected.variables["era_latitude"][:]
    longitude = selected.variables["era_longitude"][:]
    lat_idx, lon_idx = selected_to_arco_indices(latitude, longitude)
    batches, unique_chunks = spatial_batches(
        lat_idx, lon_idx, args.spatial_chunks_per_batch
    )
    print(f"Sites          : {latitude.size:,}")
    print(f"Spatial chunks : {unique_chunks:,}")
    print(f"Site batches   : {len(batches):,}")

    staging_path = args.output_dir / "era5_daily_staging.nc"
    manifest_path = args.output_dir / "arco_manifest.json"
    initialize_staging(staging_path, all_days, args.selected_pixels)
    manifest = load_manifest(manifest_path)
    completed = set(manifest["completed"])

    with Dataset(staging_path, "a") as staging:
        for group in args.groups:
            source = open_arco(group, key)
            validate_arco_coordinates(source, lat_idx, lon_idx, latitude, longitude)
            kinds = ARCO_GROUPS[group]["variables"]
            include_next = "accum" in kinds.values()
            chunks = time_chunk_slices(
                source, args.start_year, args.end_year, include_next
            )
            for chunk_index, time_slice in chunks:
                for batch_index, site_indices in enumerate(batches):
                    unit = f"{group}:t{chunk_index}:b{batch_index}"
                    if unit in completed:
                        continue
                    print(
                        f"{timestamp()} start {unit}, sites={site_indices.size}",
                        flush=True,
                    )
                    selection = source[list(kinds)].isel(
                        time=time_slice,
                        latitude=xr.DataArray(lat_idx[site_indices], dims="site"),
                        longitude=xr.DataArray(lon_idx[site_indices], dims="site"),
                    )
                    for attempt in range(1, args.max_attempts + 1):
                        try:
                            loaded = selection.load()
                            break
                        except Exception as error:
                            if attempt == args.max_attempts:
                                raise
                            delay = min(900, args.retry_delay * 2 ** (attempt - 1))
                            print(
                                f"{timestamp()} retry {unit} in {delay}s: {error}",
                                flush=True,
                            )
                            time.sleep(delay)
                    process_loaded_chunk(staging, loaded, kinds, site_indices, all_days)
                    staging.sync()
                    completed.add(unit)
                    manifest["completed"] = sorted(completed)
                    save_manifest(manifest_path, manifest)
                    print(f"{timestamp()} complete {unit}", flush=True)
                    loaded.close()
            source.close()


def e_sat(temperature):
    return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))


def calculate_smi(precipitation):
    weights = (30.0 - np.arange(30, dtype=np.float32)) / 30.0 / 31.0
    output = np.empty_like(precipitation, dtype=np.float32)
    for site in range(precipitation.shape[1]):
        output[:, site] = np.convolve(precipitation[:, site], weights, mode="full")[
            : precipitation.shape[0]
        ]
    return output


def derive_features(staging, site_slice):
    values = {
        name: np.ma.filled(staging.variables[name][:, site_slice], np.nan).astype(
            np.float32
        )
        for name in STAGING_VARIABLES
    }
    required = np.stack(list(values.values()))
    if not np.all(np.isfinite(required)):
        count = int(np.size(required) - np.isfinite(required).sum())
        raise RuntimeError(f"Staging data contains {count:,} non-finite values")

    vapour_pressure = e_sat(values["Tdew_mean"])
    vpd_max = np.maximum(e_sat(values["Tmax"]) - vapour_pressure, 0.0)
    vpd_mean = np.maximum(e_sat(values["Tmean"]) - vapour_pressure, 0.0)
    ssrd_average = values["ssrd_sum"] * 1e6 / 86400.0
    strd_average = values["strd_sum"] * 1e6 / 86400.0
    temperature_kelvin = values["Tmean"] + 273.15
    rn_mean = (
        0.77 * ssrd_average + strd_average - 0.97 * 5.67e-8 * temperature_kelvin**4
    )
    rn_total = rn_mean * 0.0864
    delta = 4098.0 * e_sat(values["Tmean"]) / (values["Tmean"] + 237.3) ** 2
    gamma = 0.000665 * (values["sp_mean"] / 10.0)
    pet = np.where(
        rn_total < 0,
        0.0,
        1.26 * (delta / (delta + gamma)) * (rn_total / 2.45),
    )
    return {
        "Tmin": values["Tmin"],
        "Tmax": values["Tmax"],
        "Tmean": values["Tmean"],
        "ssrd_sum": values["ssrd_sum"],
        "strd_sum": values["strd_sum"],
        "tp_sum": values["tp_sum"],
        "VPD_max": vpd_max,
        "VPD_mean": vpd_mean,
        "Rn_tot": rn_total,
        "PET": pet,
        "SMI": calculate_smi(values["tp_sum"]),
    }


def era5_land_site_indices(staging):
    """Return sites present on the ERA5-Land mask and verify mask consistency."""
    masks = []
    for name in STAGING_VARIABLES:
        first_day = np.ma.filled(staging.variables[name][0, :], np.nan)
        masks.append(np.isfinite(first_day))
    reference = masks[0]
    if any(not np.array_equal(reference, mask) for mask in masks[1:]):
        raise RuntimeError("ERA5-Land masks differ between staging variables")
    if not np.any(reference):
        raise RuntimeError("No selected sites are valid on the ERA5-Land mask")
    return np.flatnonzero(reference).astype(np.int32)


def write_filtered_selection(path, selected, site_indices):
    temporary = path.with_suffix(path.suffix + ".part")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("site", site_indices.size)
        copy_selection_metadata(selected, output, site_indices)
        output.createVariable("source_site_index", "i4", ("site",), zlib=True)[:] = (
            site_indices
        )
        output.n_source_sites = len(selected.dimensions["site"])
        output.n_era5_land_valid = int(site_indices.size)
        output.n_era5_land_invalid = int(
            len(selected.dimensions["site"]) - site_indices.size
        )
        output.era5_land_filter = (
            "finite values for every staging variable on 1991-01-01"
        )
        output.source = "scripts/download_era5_selected.py"
    temporary.replace(path)


def initialize_year_file(path, year, selected, site_indices):
    start = dt.date(year, 1, 1)
    end = dt.date(year + 1, 1, 1)
    n_days = (end - start).days
    n_site = site_indices.size
    with Dataset(path, "w", format="NETCDF4") as output:
        output.createDimension("time", n_days)
        output.createDimension("site", n_site)
        time_var = output.createVariable("time", "i4", ("time",))
        time_var.units = f"days since {year}-01-01 00:00:00"
        time_var.calendar = "proleptic_gregorian"
        time_var[:] = np.arange(n_days, dtype=np.int32)
        copy_selection_metadata(selected, output, site_indices)
        output.createVariable("source_site_index", "i4", ("site",), zlib=True)[:] = (
            site_indices
        )
        for name, (units, long_name) in FINAL_VARIABLES.items():
            variable = output.createVariable(
                name,
                "f4",
                ("time", "site"),
                zlib=True,
                complevel=4,
                chunksizes=(n_days, min(256, n_site)),
                fill_value=np.float32(np.nan),
            )
            variable.units = units
            variable.long_name = long_name
        output.title = f"ERA5-Land daily features for selected PhenoNN pixels, {year}"
        output.source = "scripts/download_era5_selected.py"


def write_final_files(args, all_days):
    staging_path = args.output_dir / "era5_daily_staging.nc"
    with Dataset(args.selected_pixels) as selected, Dataset(staging_path) as staging:
        site_indices = era5_land_site_indices(staging)
        n_source_site = len(selected.dimensions["site"])
        n_site = site_indices.size
        n_dropped = n_source_site - n_site
        print(
            f"ERA5-Land mask : {n_site:,}/{n_source_site:,} valid sites; "
            f"dropping {n_dropped:,}",
            flush=True,
        )
        write_filtered_selection(
            args.output_dir / "selected_pixels_era5_valid.nc", selected, site_indices
        )
        outputs = {}
        for year in range(args.start_year, args.end_year + 1):
            path = args.output_dir / "daily" / f"ERA5_daily_pixelset_{year}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            initialize_year_file(path, year, selected, site_indices)
            outputs[year] = Dataset(path, "a")
        try:
            for site_start in range(0, n_site, args.derive_site_batch):
                site_stop = min(n_site, site_start + args.derive_site_batch)
                output_slice = slice(site_start, site_stop)
                source_indices = site_indices[output_slice]
                print(
                    f"{timestamp()} deriving sites {site_start}:{site_stop}",
                    flush=True,
                )
                features = derive_features(staging, source_indices)
                for year, output in outputs.items():
                    keep = all_days.astype("datetime64[Y]").astype(int) + 1970 == year
                    for name, values in features.items():
                        output.variables[name][:, output_slice] = values[keep]
                    output.sync()
        finally:
            for output in outputs.values():
                output.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-pixels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument(
        "--groups", nargs="+", choices=ARCO_GROUPS, default=list(ARCO_GROUPS)
    )
    parser.add_argument("--spatial-chunks-per-batch", type=int, default=32)
    parser.add_argument("--derive-site-batch", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--derive-only",
        action="store_true",
        help="Skip ARCO access and rebuild final files from existing staging data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    all_days = target_days(args.start_year, args.end_year)
    with Dataset(args.selected_pixels) as selected:
        latitude = selected.variables["era_latitude"][:]
        longitude = selected.variables["era_longitude"][:]
        lat_idx, lon_idx = selected_to_arco_indices(latitude, longitude)
        batches, unique_chunks = spatial_batches(
            lat_idx, lon_idx, args.spatial_chunks_per_batch
        )
        n_site = latitude.size
    time_chunks = math.ceil((all_days.size * 24 + 1) / 33792)
    values_per_chunk = 33792 * 4 * 8 * 2
    raw_bytes = unique_chunks * time_chunks * values_per_chunk * 4 * len(args.groups)
    print(f"Selected sites   : {n_site:,}")
    print(f"Unique ARCO tiles: {unique_chunks:,}")
    print(f"Spatial batches  : {len(batches):,}")
    print(f"Approx raw chunks: {raw_bytes / 1024**3:,.1f} GiB before compression")
    print(f"Daily output raw : {all_days.size * n_site * 11 * 4 / 1024**3:,.2f} GiB")
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.derive_only:
        with Dataset(args.selected_pixels) as selected:
            download_to_staging(args, selected, all_days)
    elif not (args.output_dir / "era5_daily_staging.nc").exists():
        raise FileNotFoundError(args.output_dir / "era5_daily_staging.nc")
    write_final_files(args, all_days)
    print(f"Complete: daily files are under {args.output_dir / 'daily'}", flush=True)


if __name__ == "__main__":
    main()
