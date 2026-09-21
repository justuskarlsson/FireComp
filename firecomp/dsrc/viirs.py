"""Shared VIIRS utilities for working with granule prefixes and file paths.

VIIRS products (VNP14IMG, VNP03IMG, VNP02IMG) share a common granule prefix format:
    A{YYYYDDD}.{HHMM}.{VVV}

Example: VNP14IMG.A2018166.0900.002.2021082193312.nc
         Prefix:  A2018166.0900.002

This module provides utilities for extracting prefixes, matching files,
and building prefix-to-result mappings for earthaccess queries.
"""

import math
import os
from glob import glob
from typing import Any, Optional

import netCDF4

from firecomp import config
from firecomp.dsrc.projection import read_vnp03


# Valid VIIRS product short names
VIIRS_PRODUCTS = {"VNP14IMG", "VNP03IMG", "VNP02IMG"}


def viirs_granule_prefix(filename: str) -> Optional[str]:
    """Extract granule prefix from any VIIRS filename.

    Args:
        filename: VIIRS filename (can be full path or just filename)
            e.g., "VNP14IMG.A2018166.0900.002.2021082193312.nc"

    Returns:
        Granule prefix (e.g., "A2018166.0900.002") or None if invalid format.
        This prefix is shared across VNP14IMG, VNP03IMG, VNP02IMG for the same observation.
    """
    parts = os.path.basename(filename).split(".")
    if len(parts) >= 4:
        # Format: PRODUCT.A{YYYYDDD}.{HHMM}.{VVV}.{TIMESTAMP}.nc
        return f"{parts[1]}.{parts[2]}.{parts[3]}"
    return None


def match_granule_prefix(filename: str, prefixes: set[str]) -> bool:
    """Check if a filename matches any of the granule prefixes.

    Args:
        filename: VIIRS filename to check
        prefixes: Set of granule prefixes to match against

    Returns:
        True if the filename's prefix matches any in the set.
    """
    prefix = viirs_granule_prefix(filename)
    return prefix in prefixes if prefix else False


def result_filename(result) -> str:
    """Extract filename from an earthaccess search result.

    Args:
        result: Earthaccess search result object

    Returns:
        The filename (basename) of the data file.
    """
    url = result.data_links()[0]
    return os.path.basename(url)


def results_to_prefix_map(results: list) -> dict[str, Any]:
    """Build a mapping from granule prefix to earthaccess result.

    Args:
        results: List of earthaccess search results

    Returns:
        Dict mapping granule prefix (e.g., "A2018166.0900.002") to the
        corresponding earthaccess result object.
    """
    by_prefix: dict[str, Any] = {}
    for r in results:
        fname = result_filename(r)
        prefix = viirs_granule_prefix(fname)
        if prefix:
            by_prefix[prefix] = r
    return by_prefix


def prefix_to_date_str(prefix: str) -> str:
    """Convert granule prefix to ISO date string.

    Args:
        prefix: Granule prefix (e.g., "A2018166.0900.002")

    Returns:
        ISO date string (e.g., "2018-06-15")
    """
    from datetime import datetime, timedelta

    # prefix format: A{YYYY}{DDD}.{HHMM}.{VVV}
    year = int(prefix[1:5])
    doy = int(prefix[5:8])
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return dt.strftime("%Y-%m-%d")


def group_prefixes_by_date(prefixes: set[str]) -> dict[str, set[str]]:
    """Group granule prefixes by their date.

    Args:
        prefixes: Set of granule prefixes

    Returns:
        Dict mapping date string to set of prefixes for that date.
    """
    by_date: dict[str, set[str]] = {}
    for prefix in prefixes:
        date_str = prefix_to_date_str(prefix)
        if date_str not in by_date:
            by_date[date_str] = set()
        by_date[date_str].add(prefix)
    return by_date


def group_vnp14_by_date(vnp14_paths: list[str]) -> dict[str, list[str]]:
    """Group VNP14 paths by their date."""
    by_date: dict[str, list[str]] = {}
    for path in vnp14_paths:
        date_str = prefix_to_date_str(viirs_granule_prefix(path))
        if date_str not in by_date:
            by_date[date_str] = []
        by_date[date_str].append(path)
    return by_date


def extract_vnp14_metadata(
    vnp14_path: str,
) -> Optional[tuple[float, float, float, float]]:
    """Extract bounding box from VNP14 file metadata.

    Returns (min_lon, min_lat, max_lon, max_lat) or None if not available.
    """
    try:
        ds = netCDF4.Dataset(vnp14_path, "r")
        try:
            # VNP14 has these attributes (see VNP14_Attributes in vnp14.py)
            west = float(ds.WestBoundingCoordinate)
            east = float(ds.EastBoundingCoordinate)
            south = float(ds.SouthBoundingCoordinate)
            north = float(ds.NorthBoundingCoordinate)
            bbox = (west, south, east, north)
            return bbox
        finally:
            ds.close()
    except Exception:
        return None


def extract_vnp14_gring(
    vnp14_path: str,
) -> Optional[tuple[list[float], list[float]]]:
    """Extract GRing polygon from VNP14 file metadata.

    The GRing defines the actual swath boundary as a 4-point polygon,
    which is more accurate than the axis-aligned bounding box for polar passes.

    Returns (lons, lats) where each is a list of 4 corner coordinates,
    or None if not available.
    """
    try:
        ds = netCDF4.Dataset(vnp14_path, "r")
        try:
            lons = list(ds.GRingPointLongitude)
            lats = list(ds.GRingPointLatitude)
            return (lons, lats)
        finally:
            ds.close()
    except Exception:
        return None


def _crosses_dateline(lons: list[float]) -> bool:
    """Check if a polygon crosses the dateline (has both very positive and very negative longitudes)."""
    min_lon = min(lons)
    max_lon = max(lons)
    # If the span is > 180°, it likely crosses the dateline
    return (max_lon - min_lon) > 180


def _normalize_lon(lon: float, offset: float = 0) -> float:
    """Normalize longitude to [offset, offset+360) range."""
    return ((lon - offset) % 360) + offset


def point_in_polygon(
    px: float, py: float, poly_x: list[float], poly_y: list[float]
) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.

    Args:
        px, py: Point coordinates
        poly_x, poly_y: Lists of polygon vertex coordinates

    Returns:
        True if point is inside the polygon.
    """
    n = len(poly_x)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]

        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def bbox_intersects_gring(
    bbox: tuple[float, float, float, float],
    gring_lons: list[float],
    gring_lats: list[float],
) -> bool:
    """Check if a bounding box intersects with a GRing polygon.

    Handles dateline-crossing polygons by normalizing longitudes.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        gring_lons: List of polygon longitude vertices
        gring_lats: List of polygon latitude vertices

    Returns:
        True if there is any intersection.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Handle dateline-crossing polygons by normalizing to [0, 360) range
    if _crosses_dateline(gring_lons):
        # Normalize polygon lons to [0, 360)
        gring_lons = [_normalize_lon(lon, 0) for lon in gring_lons]
        # Normalize bbox lons to [0, 360)
        min_lon = _normalize_lon(min_lon, 0)
        max_lon = _normalize_lon(max_lon, 0)
        # If bbox itself crosses dateline after normalization, handle specially
        if min_lon > max_lon:
            # Bbox crosses dateline - check both sides
            # This is rare for fire bboxes, but handle it anyway
            return _bbox_intersects_gring_impl(
                (min_lon, min_lat, 360.0, max_lat), gring_lons, gring_lats
            ) or _bbox_intersects_gring_impl(
                (0.0, min_lat, max_lon, max_lat), gring_lons, gring_lats
            )

    return _bbox_intersects_gring_impl(
        (min_lon, min_lat, max_lon, max_lat), gring_lons, gring_lats
    )


def _bbox_intersects_gring_impl(
    bbox: tuple[float, float, float, float],
    gring_lons: list[float],
    gring_lats: list[float],
) -> bool:
    """Implementation of bbox-polygon intersection check."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # Check if bbox center is inside polygon
    cx = (min_lon + max_lon) / 2
    cy = (min_lat + max_lat) / 2
    if point_in_polygon(cx, cy, gring_lons, gring_lats):
        return True

    # Check if any bbox corner is inside polygon
    corners = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]
    for px, py in corners:
        if point_in_polygon(px, py, gring_lons, gring_lats):
            return True

    # Check if any polygon vertex is inside bbox
    for lon, lat in zip(gring_lons, gring_lats):
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True

    return False


def viirs_is_day(prefix: str, lon: float, lat: float) -> bool:
    """Check if a VIIRS observation is during daytime at the given location.

    Uses solar position calculation based on:
    - UTC time from prefix
    - Longitude for local solar time
    - Latitude and day of year for daylight hours

    Conservative: errs on the side of returning False to avoid nighttime images.

    Args:
        prefix: Granule prefix (e.g., "A2018166.0900.002")
        lon: Longitude in degrees (-180 to 180)
        lat: Latitude in degrees (-90 to 90)

    Returns:
        True if the observation is likely during good daylight hours.
    """
    # Parse UTC time and day of year from prefix
    # Format: A{YYYY}{DDD}.{HHMM}.{VVV}
    try:
        year = int(prefix[1:5])
        doy = int(prefix[5:8])  # Day of year (1-366)
        time_str = prefix.split(".")[1]  # HHMM
        utc_hour = int(time_str[:2])
        utc_minute = int(time_str[2:])
    except (ValueError, IndexError):
        return False  # Can't parse, be conservative

    # Calculate local solar time
    # Local solar time ≈ UTC + (longitude / 15) hours
    utc_decimal = utc_hour + utc_minute / 60.0
    local_solar_time = (utc_decimal + lon / 15.0) % 24.0

    # Calculate approximate daylight hours based on latitude and day of year
    # Using simplified solar declination model
    #
    # Solar declination angle (degrees):
    # δ ≈ -23.45 * cos(360/365 * (doy + 10))
    #
    # Day length depends on latitude and declination

    # Solar declination (simplified)
    declination = -23.45 * math.cos(math.radians(360.0 / 365.0 * (doy + 10)))

    # Hour angle at sunrise/sunset
    # cos(ω) = -tan(lat) * tan(δ)
    lat_rad = math.radians(lat)
    decl_rad = math.radians(declination)

    cos_hour_angle = -math.tan(lat_rad) * math.tan(decl_rad)

    # Clamp to [-1, 1] for polar regions (midnight sun / polar night)
    if cos_hour_angle < -1:
        # Midnight sun - always day (but be conservative, use wide window)
        sunrise = 2.0
        sunset = 22.0
    elif cos_hour_angle > 1:
        # Polar night - never day
        return False
    else:
        # Normal case: calculate sunrise/sunset
        hour_angle = math.degrees(math.acos(cos_hour_angle))
        # Hour angle in hours (15 degrees = 1 hour)
        half_day_length = hour_angle / 15.0

        # Solar noon is at 12:00 local solar time
        sunrise = 12.0 - half_day_length
        sunset = 12.0 + half_day_length

    # Add margin to be conservative (1.5 hours after sunrise, 1.5 hours before sunset)
    # This ensures we're well within daylight with good solar illumination
    margin = 1.5
    safe_sunrise = sunrise + margin
    safe_sunset = sunset - margin

    # Check if local solar time is within safe daylight window
    return safe_sunrise <= local_solar_time <= safe_sunset


def patch_mpp_xy(lon, lat):
    h_dlon = lon[0, 0] - lon[-1, 0]
    h_dlat = lat[0, 0] - lat[-1, 0]
    w_dlon = lon[0, 0] - lon[0, -1]
    w_dlat = lat[0, 0] - lat[0, -1]
    # Calculate meters per degree at the midpoint latitude
    import numpy as np

    R = 6371000  # Mean Earth radius in meters

    # Compute the mean latitude for metric conversions
    mean_lat = np.mean([lat[0, 0], lat[-1, 0], lat[0, -1], lat[-1, -1]])

    # Length of one degree latitude in meters (approximately constant)
    meters_per_deg_lat = (2 * np.pi * R) / 360.0

    # Length of one degree longitude in meters (depends on latitude)
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(mean_lat))

    # Compute delta in meters using degree differences
    h_dm = math.hypot(h_dlon * meters_per_deg_lon, h_dlat * meters_per_deg_lat)
    w_dm = math.hypot(w_dlon * meters_per_deg_lon, w_dlat * meters_per_deg_lat)
    mpp_y = h_dm / lon.shape[0]
    mpp_x = w_dm / lon.shape[1]
    return mpp_x, mpp_y


def test_vnp03_mpp(path, only_print_x=False, num_patches=8):

    lat, lon = read_vnp03(path)
    import numpy as np

    patches = num_patches
    y, x = lat.shape
    y_edges = np.linspace(0, y, patches + 1, dtype=int)
    x_edges = np.linspace(0, x, patches + 1, dtype=int)

    mpp_grid = np.zeros((patches, patches, 2))  # (patch_i, patch_j, [mpp_y, mpp_x])

    for i in range(patches):
        for j in range(patches):
            y0, y1 = y_edges[i], y_edges[i + 1]
            x0, x1 = x_edges[j], x_edges[j + 1]
            lat_patch = lat[y0:y1, x0:x1]
            lon_patch = lon[y0:y1, x0:x1]
            mpp_x, mpp_y = patch_mpp_xy(lon_patch, lat_patch)
            mpp_grid[i, j, 0] = mpp_y
            mpp_grid[i, j, 1] = mpp_x

    if only_print_x:
        # print("Meters-per-pixel (X) for each patch:")
        # for i in range(patches):
        #     row_mpp = []
        #     for j in range(patches):
        #         row_mpp.append(f"{mpp_grid[i, j, 1]:.0f}")
        #     print("  ".join(row_mpp))
        median_val = np.median(mpp_grid[:, :, 1])
        percentile_25_val = np.percentile(mpp_grid[:, :, 1], 25)
        print(f"Shape: {lat.shape}")
        print(f"Median: {median_val:.0f}")
        print(f"25th percentile (lowest quartile): {percentile_25_val:.0f}")
    else:
        print("Meters-per-pixel (Y, X) for each patch:")
        for i in range(patches):
            row_mpp = []
            for j in range(patches):
                row_mpp.append(f"({mpp_grid[i, j, 0]:.1f}, {mpp_grid[i, j, 1]:.1f})")
            print("  ".join(row_mpp))


if __name__ == "__main__":
    paths = [
        "data/vnp03img/VNP03IMG.A2018170.0054.002.2021082222009.nc",
        "data/vnp03img/VNP03IMG.A2018170.0724.002.2021082222006.nc",
        "data/vnp03img/VNP03IMG.A2018170.0900.002.2021082222007.nc",
        "data/vnp03img/VNP03IMG.A2018170.0906.002.2021082222006.nc",
    ]
    for path in paths:
        test_vnp03_mpp(path, only_print_x=True, num_patches=50)
