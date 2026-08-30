"""
Grid alignment helpers shared by every preprocessing script.

The training pipeline targets the ERA5-Land 0.1° grid. We read the canonical
lat / lon vectors from a reference ERA5 file at runtime so the conventions
(decreasing-lat vs increasing-lat, [-180, 180] vs [0, 360], …) are taken
straight from the data instead of being hard-coded.
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr


# ── Reference grid ───────────────────────────────────────────────────────────


def read_reference_grid(ref_nc: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (lat, lon) 1-D float64 arrays from the first NetCDF file given.

    The file is opened with xarray; both `latitude/longitude` and
    `lat/lon` naming conventions are recognised.
    """
    # engine="h5netcdf" (h5py-backed) is used instead of the default
    # netcdf4 engine: the latter's bundled HDF5 in some venvs fails with
    # "HDF error -101" on the ORCHIDEE PFTmap files, while h5netcdf links
    # the system HDF5 (same one ncdump uses) and reads them fine.
    ds = xr.open_dataset(ref_nc, decode_times=False, engine="h5netcdf")
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[lat_name].values.astype(np.float64)
    lon = ds[lon_name].values.astype(np.float64)
    ds.close()
    return lat, lon


# ── ERA5-Land target grid (analytic — no reference file needed) ──────────────

ERA5_RES = 0.1
ERA5_NLAT = 1801  # node-registered: centres on the 0.1° multiples, 90.0 … -90.0
ERA5_NLON = 3600  # -180.0 … 179.9 (.x0) — same physical grid as ERA5's 0..360


def era5_land_grid():
    """Canonical ERA5-Land 0.1° grid, built from its known indexing (no file).

    ERA5-Land is node-registered: latitude runs 90.0 → -90.0 on the 0.1°
    multiples (1801 pts, both poles included), longitude 0.0 → 359.9 (.x0). We
    return longitude in -180..180 (-180.0 → 179.9) — the SAME physical grid —
    so the regridded LAI/PFT stay in the THEIA/ORCHIDEE convention and
    build_daily_dataset_pixelset's +N_LON//2 index roll maps them onto ERA5's
    0..360 exactly. Returns (lat, lon) float64.
    """
    lat = np.round(90.0 - np.arange(ERA5_NLAT) * ERA5_RES, 6)
    lon = np.round(-180.0 + np.arange(ERA5_NLON) * ERA5_RES, 6)
    return lat, lon


# ── Bounding-box slicing ─────────────────────────────────────────────────────


def slice_to_bbox(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    """
    Crop `da` to [lat_min, lat_max] × [lon_min, lon_max].

    Handles both decreasing-lat (ERA5-Land convention) and increasing-lat,
    and both [-180, 180] and [0, 360] longitude conventions, with auto
    antimeridian wrapping for cross-180° boxes.
    """
    # Latitude
    lat_vec = da[lat_name].values
    if float(lat_vec[0]) > float(lat_vec[-1]):
        da = da.sel({lat_name: slice(lat_max, lat_min)})
    else:
        da = da.sel({lat_name: slice(lat_min, lat_max)})

    # Longitude
    lon_vec = da[lon_name].values
    if float(np.nanmax(lon_vec)) > 180.0:
        lo = lon_min % 360
        hi = lon_max % 360
        if lo <= hi:
            da = da.sel({lon_name: slice(lo, hi)})
        else:
            left = da.sel({lon_name: slice(lo, 360)})
            right = da.sel({lon_name: slice(0, hi)})
            da = xr.concat([left, right], dim=lon_name)
    else:
        da = da.sel({lon_name: slice(lon_min, lon_max)})
    return da


# ── Regridding helpers ───────────────────────────────────────────────────────


def coarsen_2x2_nanmean(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
    min_valid: int = 1,
) -> xr.DataArray:
    """
    NaN-aware 2×2 spatial coarsening. The source grid is expected to be twice
    as fine as the target (e.g. THEIA AVHRR 0.05° → ERA5 0.1°). When the
    source has an odd dimension, the boundary is padded with NaN so every
    output cell sees 4 source pixels.

    Parameters
    ----------
    min_valid
        Minimum number of finite source pixels (out of 4) required to keep
        the coarsened cell. Cells with fewer valid pixels are set to NaN.
        - 1 (default permissive)  : keep everything, accept 1-pixel cells
        - 2 (recommended)         : drop cells where only 1 pixel is valid
        - 3 / 4                   : strict, drops more cells in winter
    """
    block = da.coarsen({lat_name: 2, lon_name: 2}, boundary="pad")
    mean_val = block.mean(skipna=True)
    valid_count = (
        (~da.isnull())
        .coarsen(
            {lat_name: 2, lon_name: 2},
            boundary="pad",
        )
        .sum()
    )
    if min_valid > 1:
        mean_val = mean_val.where(valid_count >= min_valid)
    return mean_val


def reindex_to_reference(
    da: xr.DataArray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
    src_lat_name: str = "lat",
    src_lon_name: str = "lon",
    tol: float = 0.06,
) -> xr.DataArray:
    """
    Snap `da` to the reference (lat, lon) coordinates by nearest-neighbour
    matching with a tolerance equal to ~the target cell size (0.06° default
    for the 0.1° ERA5-Land grid).

    Cells with no source within `tol` get NaN — this is the desired
    behaviour at the poles where the source grid (e.g. AVHRR) does not
    extend all the way to ±90°.
    """
    return da.rename({src_lat_name: "latitude", src_lon_name: "longitude"}).reindex(
        latitude=ref_lat, longitude=ref_lon, method="nearest", tolerance=tol
    )


# ── NetCDF write options ─────────────────────────────────────────────────────


def make_encoding(
    ds: xr.Dataset,
    chunks: Optional[dict] = None,
    complevel: int = 4,
) -> dict:
    """
    Build a per-variable encoding dict for `to_netcdf`.

    Defaults to zlib level 4 with float32 storage. Chunks (if given) are
    applied to every data var.
    """
    enc = {}
    for name in ds.data_vars:
        e = {"zlib": True, "complevel": complevel}
        if ds[name].dtype.kind == "f":
            e["dtype"] = "float32"
            e["_FillValue"] = np.float32(np.nan)
        elif ds[name].dtype.kind in ("i", "u"):
            e["dtype"] = str(ds[name].dtype)
        if chunks is not None:
            e["chunksizes"] = tuple(
                chunks.get(d, ds[name].sizes[d]) for d in ds[name].dims
            )
        enc[name] = e
    return enc
