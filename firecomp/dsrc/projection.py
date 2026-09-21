"""Projection utilities for projecting swath data to uniform lon/lat grids.

This module handles projecting satellite swath data (like VNP14 fire masks)
from irregular satellite coordinates to a regular geographic grid.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import netCDF4
import numpy as np
import torch

from firecomp.dsrc.vnp14 import DEG_CELL_SIZE
from osgeo import gdal, osr


def latitude_kernel_size(lat: float, base_kernel: int = 3) -> int:
    """Compute kernel size adjusted for latitude.

    At high latitudes, 1° longitude covers fewer meters than at equator.
    Source pixels (constant ~375m) span more target cells in X direction.
    We increase kernel size to ensure coverage.

    Args:
        lat: Latitude in degrees (center of region)
        base_kernel: Kernel size at equator (default 3)

    Returns:
        Adjusted kernel size (always odd, minimum base_kernel)
    """
    cos_lat = math.cos(math.radians(abs(lat)))
    # At equator cos=1, kernel=base. At 65° cos≈0.42, kernel≈base/0.42≈7
    raw = base_kernel / cos_lat
    # Round up to nearest odd integer
    kernel = int(math.ceil(raw))
    if kernel % 2 == 0:
        kernel += 1
    return max(base_kernel, kernel)


gdal.UseExceptions()

IBAND_NAMES = ["I01", "I02", "I03", "I04", "I05"]


def read_vnp14(vnp14_path: str) -> np.ndarray:
    """Read fire mask from VNP14 file. Returns [H, W] uint8."""
    ds = netCDF4.Dataset(vnp14_path, "r")
    try:
        return np.array(ds.variables["fire mask"][:])
    finally:
        ds.close()


def read_vnp03(vnp03_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read lat/lon from VNP03IMG file. Returns (lat, lon) each [H, W]."""
    ds = netCDF4.Dataset(vnp03_path, "r")
    try:
        geo = ds.groups["geolocation_data"]
        lat = np.array(geo.variables["latitude"][:])
        lon = np.array(geo.variables["longitude"][:])
        return lat, lon
    finally:
        ds.close()


vnp02_no_data_val = 65528


def read_vnp02(
    vnp02_path: str, no_data_val: float = vnp02_no_data_val
) -> np.ndarray | None:
    """Read I-bands (I1-I5) from VNP02IMG file with calibration. Returns [5, H, W] float32 or None if no bands found."""
    ds = netCDF4.Dataset(vnp02_path, "r")
    try:
        obs = ds.groups.get("observation_data", ds)
        bands = []
        for band in IBAND_NAMES:
            if band in obs.variables:
                var = obs.variables[band]
                data = np.array(var[:], dtype=np.float32)
                has_data_mask = data <= var.valid_max
                data = (data - var.add_offset) / var.scale_factor
                data[~has_data_mask] = no_data_val
                bands.append(data)
        if not bands:
            return None
        return np.stack(bands, axis=0)
    finally:
        ds.close()


@dataclass
class Projection:
    """Result of projecting data to a uniform grid."""

    data: np.ndarray  # [C, H, W]
    bounds: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    H: int
    W: int


def compute_projection_coords(
    lat: np.ndarray,
    lon: np.ndarray,
    resolution: float,
) -> tuple[torch.Tensor, int, int, tuple[float, float, float, float]]:
    """Compute projection coordinates mapping source pixels to destination grid.

    Args:
        lat: 2D array of latitudes [src_H, src_W]
        lon: 2D array of longitudes [src_H, src_W]
        resolution: cell size in degrees

    Returns:
        xy_coords: Tensor [src_H, src_W, 2] with (x, y) destination coordinates
        out_W: output grid width
        out_H: output grid height
        bounds: (min_lon, min_lat, max_lon, max_lat)
    """
    # Calculate bounds from data
    min_lon, max_lon = float(lon.min()), float(lon.max())
    min_lat, max_lat = float(lat.min()), float(lat.max())
    bounds = (min_lon, min_lat, max_lon, max_lat)

    # Compute output grid dimensions
    out_H = round((max_lat - min_lat) / resolution)
    out_W = round((max_lon - min_lon) / resolution)

    # Normalize coordinates to [0, 1] then scale to grid size
    lat_t = torch.tensor(lat, dtype=torch.float32)
    lon_t = torch.tensor(lon, dtype=torch.float32)

    x_norm = (lon_t - min_lon) / (max_lon - min_lon)
    y_norm = (lat_t - min_lat) / (max_lat - min_lat)

    x_idx = x_norm * out_W
    # Flip y-axis: lat increases upward, but image y increases downward
    y_idx = (out_H - 1) - (y_norm * out_H)

    xy_coords = torch.stack([x_idx, y_idx], dim=2)

    return xy_coords, out_W, out_H, bounds


def project_patch(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: tuple[float, float, float, float],
    resolution: float = DEG_CELL_SIZE,
    kernel_size: Optional[int] = None,
    no_data_val: float = 0.0,
    method: str = "inv_dist",
) -> Optional[np.ndarray]:
    """Project data to a patch defined by a geographic bounding box.

    This is more efficient than projecting the full image when only a small
    patch is needed (e.g., for a fire event). Uses the C++ project_impl which
    provides accurate interpolation.

    Args:
        data: Source data [H, W] or [C, H, W]
        lat: Latitude array [H, W]
        lon: Longitude array [H, W]
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat)
        resolution: Output cell size in degrees
        kernel_size: Kernel size for interpolation. If None, auto-computed
            based on bbox center latitude to account for longitude compression.
        no_data_val: Value for missing data
        method: Interpolation method ("nearest", "inv_dist", "max")

    Returns:
        Projection with data covering the bbox
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Auto-compute kernel size based on latitude if not specified
    if kernel_size is None:
        center_lat = (min_lat + max_lat) / 2
        kernel_size = latitude_kernel_size(center_lat)

    # Compute output dimensions for this bbox
    out_H = max(1, round((max_lat - min_lat) / resolution))
    out_W = max(1, round((max_lon - min_lon) / resolution))

    # Find source pixels that fall within or near the bbox (with padding for kernel)
    pad_deg = kernel_size * resolution
    in_region = (
        (lon >= min_lon - pad_deg)
        & (lon <= max_lon + pad_deg)
        & (lat >= min_lat - pad_deg)
        & (lat <= max_lat + pad_deg)
    )

    if not np.any(in_region):
        # No source data overlaps the bbox - expected for curved swath edges
        return None

    # Get indices of source pixels in region
    src_rows, src_cols = np.where(in_region)
    if len(src_rows) == 0:
        return None

    # Extract subset of data and coordinates
    row_min, row_max = src_rows.min(), src_rows.max() + 1
    col_min, col_max = src_cols.min(), src_cols.max() + 1

    lat_sub = lat[row_min:row_max, col_min:col_max]
    lon_sub = lon[row_min:row_max, col_min:col_max]

    if data.ndim == 2:
        data_sub = data[row_min:row_max, col_min:col_max]
        data_t = torch.tensor(
            data_sub, dtype=torch.float32 if method == "inv_dist" else torch.uint8
        )
        data_t = data_t.unsqueeze(0)
    else:
        data_sub = data[:, row_min:row_max, col_min:col_max]
        data_t = torch.tensor(
            data_sub, dtype=torch.float32 if method == "inv_dist" else torch.uint8
        )

    # Compute destination coordinates relative to bbox
    lat_t = torch.tensor(lat_sub, dtype=torch.float32)
    lon_t = torch.tensor(lon_sub, dtype=torch.float32)

    x_norm = (lon_t - min_lon) / (max_lon - min_lon)
    y_norm = (lat_t - min_lat) / (max_lat - min_lat)

    x_idx = x_norm * out_W
    # Flip y-axis: lat increases upward, image y increases downward
    y_idx = (out_H - 1) - (y_norm * out_H)

    xy_coords = torch.stack([x_idx, y_idx], dim=2)

    # Use C++ project for accurate interpolation
    from firecomp.cpp import project as cpp_project

    projected = cpp_project(
        data_t, out_W, out_H, xy_coords, method, kernel_size, float(no_data_val)
    )

    return projected.numpy()


def project_patch_uint8(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: tuple[float, float, float, float],
    resolution: float = DEG_CELL_SIZE,
    kernel_size: Optional[int] = None,
    no_data_val: int = 0,
) -> Optional[np.ndarray]:
    """Project uint8 data to a patch using nearest neighbor interpolation.

    Convenience wrapper around project_patch for uint8 data like fire masks.
    Kernel size is auto-computed based on latitude if not specified.
    """
    proj = project_patch(
        data,
        lat,
        lon,
        bbox,
        resolution,
        kernel_size,
        float(no_data_val),
        method="nearest",
    )
    if proj is None:
        return None
    return proj.astype(np.uint8)


def project_patch_float32(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: tuple[float, float, float, float],
    resolution: float = DEG_CELL_SIZE,
    kernel_size: Optional[int] = None,
    no_data_val: float = 0.0,
) -> Optional[np.ndarray]:
    """Project float32 data to a patch using inverse distance weighting.

    Convenience wrapper around project_patch for float32 data like I-bands.
    Kernel size is auto-computed based on latitude if not specified.
    """
    return project_patch(
        data, lat, lon, bbox, resolution, kernel_size, no_data_val, method="inv_dist"
    )


def save_tif(
    projection: Projection,
    output_path: str,
    block_size: int = 128,
    build_overviews: bool = False,
    no_data_val: int | None = None,
    overview_aggregation: Literal["AVERAGE", "NEAREST"] = "NEAREST",
):
    """Save projection to a GeoTIFF file."""
    # GDAL 3.5+ has GDT_Float16, fallback to Float32 for older versions
    gdt_float16 = getattr(gdal, "GDT_Float16", gdal.GDT_Float32)
    dtype_map = {
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("float16"): gdt_float16,
        np.dtype("float32"): gdal.GDT_Float32,
    }
    gdal_dtype = dtype_map.get(projection.data.dtype, gdal.GDT_Float32)

    options = [
        "COMPRESS=ZSTD",
        "ZSTD_LEVEL=3",
        "TILED=YES",
        f"BLOCKXSIZE={block_size}",
        f"BLOCKYSIZE={block_size}",
        "INTERLEAVE=PIXEL",
    ]

    min_lon, min_lat, max_lon, max_lat = projection.bounds
    px_width = (max_lon - min_lon) / projection.W
    px_height = (max_lat - min_lat) / projection.H

    # Write all C, H, W bands
    data = projection.data
    if data.ndim == 2:
        data = data[None, ...]  # (1, H, W)
    num_bands = data.shape[0]

    dst = gdal.GetDriverByName("GTiff").Create(
        output_path, projection.W, projection.H, num_bands, gdal_dtype, options=options
    )
    dst.SetGeoTransform((min_lon, px_width, 0.0, max_lat, 0.0, -px_height))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetSpatialRef(srs)

    for c in range(num_bands):
        band = dst.GetRasterBand(c + 1)
        band.WriteArray(data[c])
        if no_data_val is not None:
            band.SetNoDataValue(float(no_data_val))

    dst.FlushCache()

    if build_overviews:
        overviews = [2, 4, 8, 16, 32, 64]
        dst.BuildOverviews(overview_aggregation, overviews)

    dst = None


def debug_values(data: np.ndarray):
    """
    Print a histogram of value ranges and their occurrences in the data.

    Args:
        data: np.ndarray, can be any shape.
    """
    flat = data.flatten()
    flat = flat.astype(np.float32)

    # For float, use 10 bins by default over the data range
    min_val, max_val = np.nanmin(flat), np.nanmax(flat)
    bins = np.linspace(min_val, max_val, 11)
    hist, edges = np.histogram(flat, bins=bins)
    print(f"Value range: {min_val} to {max_val}")
    for i in range(len(hist)):
        print(f"{edges[i]:.3f} - {edges[i+1]:.3f}: {hist[i]}")
    num_nans = np.count_nonzero(np.isnan(flat))
    if num_nans > 0:
        print(f"NaN: {num_nans}")

