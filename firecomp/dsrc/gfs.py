"""
GFS (Global Forecast System) Weather Forecast Data Source

Downloads and provides access to GFS weather forecasts from Google Earth Engine.
Forecasts are aggregated to daily values for up to 7 days ahead.

Key features:
- Server-side aggregation in GEE (efficient)
- Global coverage at 0.25° resolution
- int16 encoding for compact storage (~16 MB per day)
- No grid splitting needed (fits within GEE's 32 MB limit)
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from osgeo import gdal, gdalconst
from pqdm.threads import pqdm

from firecomp.config import config
from firecomp.dsrc.dsrc import DSRC
import urllib3

gdal.UseExceptions()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Band configuration: name -> (min, max) for int16 scaling
# Scaling: int16_val = (val - min) / (max - min) * 32767
# Reverse: val = (int16_val / 32767) * (max - min) + min
#
# Essential variables for wildfire prediction only:
# - temp_max: fuel drying
# - rh_min: critical fire weather indicator
# - u_wind_max, v_wind_max: spread rate and direction
# - precip_sum: fuel wetting
GFS_BAND_CONFIG = {
    "temp_max": (-80, 60),  # °C - Daily max temperature
    "rh_min": (0, 100),  # % - Min relative humidity (fire-critical)
    "u_wind_max": (-100, 100),  # m/s - Max eastward wind component
    "v_wind_max": (-100, 100),  # m/s - Max northward wind component
    "precip_sum": (0, 500),  # mm - Daily precipitation sum
}

GFS_BANDS = list(GFS_BAND_CONFIG.keys())


def _init_ee():
    """Initialize Earth Engine."""
    import ee

    ee.Initialize(project="ee-karlssonjustus")


def gfs_aggregate_daily(init_date: str, forecast_day: int = 1):
    """
    Aggregate GFS forecasts to daily values for a single forecast day.

    Args:
        init_date: Initialization date 'YYYY-MM-DD' (uses 00Z run)
        forecast_day: Which forecast day (1 = first day ahead, etc.)

    Returns:
        Aggregated ee.Image with 10 bands
    """
    import ee

    init_dt = datetime.strptime(init_date, "%Y-%m-%d")

    # GFS collection
    gfs = ee.ImageCollection("NOAA/GFS0P25")

    # Filter to this initialization (00Z run)
    init_millis = ee.Date(init_date).millis()
    init_images = gfs.filter(ee.Filter.eq("creation_time", init_millis))

    # Forecast hours for this day
    # For forecast_day N, we want hours covering target_date = init_date + N
    # Hour 0 = init_date 00:00 UTC, Hour 24 = init_date+1 00:00 UTC, etc.
    # So for day 1 targeting init_date+1: hours 24-47 (00:00-23:00 UTC on target)
    # This aligns with ERA5 daily (00:00-23:59 UTC) and fire dates (UTC day)
    hour_start = forecast_day * 24
    hour_end = (forecast_day + 1) * 24 - 1

    day_forecasts = init_images.filter(
        ee.Filter.And(
            ee.Filter.gte("forecast_hours", hour_start),
            ee.Filter.lte("forecast_hours", hour_end),
        )
    )

    # Temperature (already in Celsius in GEE)
    temp = day_forecasts.select("temperature_2m_above_ground")
    temp_max = temp.max().rename("temp_max")

    # Relative humidity - minimum is most fire-relevant
    rh = day_forecasts.select("relative_humidity_2m_above_ground")
    rh_min = rh.min().rename("rh_min")

    # Wind: max absolute value preserving sign for each component
    u_wind = day_forecasts.select("u_component_of_wind_10m_above_ground")
    v_wind = day_forecasts.select("v_component_of_wind_10m_above_ground")

    def max_abs_preserve_sign(collection):
        coll_max = collection.max()
        coll_min = collection.min()
        abs_max = coll_max.abs()
        abs_min = coll_min.abs()
        return coll_max.where(abs_min.gt(abs_max), coll_min)

    u_wind_max = max_abs_preserve_sign(u_wind).rename("u_wind_max")
    v_wind_max = max_abs_preserve_sign(v_wind).rename("v_wind_max")

    # Precipitation: sum of 6-hourly bucket endpoints
    # GFS precip is cumulative over 6-hour windows ending at hours 6, 12, 18, 24, 30, ...
    # To cover full UTC day (hours 24-48 for day 1), we need buckets at +6, +12, +18, +24
    # The +24 bucket (hour 48 for day 1) covers the final 6 hours including 00:00 UTC next day
    precip_hours = [hour_start + h for h in [6, 12, 18, 24]]
    precip_images = init_images.filter(
        ee.Filter.inList("forecast_hours", precip_hours)
    ).select("total_precipitation_surface")
    precip_sum = precip_images.sum().rename("precip_sum")

    # Combine into single image (5 essential bands only)
    target_date = init_dt + timedelta(days=forecast_day)
    daily = ee.Image.cat(
        [
            temp_max,
            rh_min,
            u_wind_max,
            v_wind_max,
            precip_sum,
        ]
    ).set(
        {
            "init_date": init_date,
            "forecast_day": forecast_day,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "system:time_start": ee.Date(target_date).millis(),
        }
    )

    return daily


def _to_int16(image):
    """Convert float bands to int16 with linear scaling."""
    import ee

    band_names = GFS_BANDS
    mins = [GFS_BAND_CONFIG[b][0] for b in band_names]
    maxs = [GFS_BAND_CONFIG[b][1] for b in band_names]

    min_img = ee.Image.constant(mins).rename(band_names)
    range_img = ee.Image.constant([maxs[i] - mins[i] for i in range(len(mins))]).rename(
        band_names
    )

    scaled = image.subtract(min_img).divide(range_img).multiply(32767).round().toInt16()
    return scaled


def download_gfs_day(
    init_date: str, forecast_day: int, output_dir: str
) -> Optional[str]:
    """
    Download a single aggregated GFS forecast day.

    Args:
        init_date: Initialization date 'YYYY-MM-DD'
        forecast_day: Which forecast day (1-7)
        output_dir: Where to save the file

    Returns:
        Path to downloaded file, or None if failed
    """
    import ee
    import requests

    os.makedirs(output_dir, exist_ok=True)

    filename = f"gfs_{init_date}_day{forecast_day:02d}.tif"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return filepath

    # Generate aggregated image
    daily = gfs_aggregate_daily(init_date, forecast_day)
    daily_int16 = _to_int16(daily)

    # Download globally
    region = ee.Geometry.Rectangle([-180, -90, 180, 90], None, False)
    params = {
        "name": filename.replace(".tif", ""),
        "crs": "EPSG:4326",
        "format": "GeoTIFF",
        "scale": 27830,  # ~0.25 degrees
        "region": region.getInfo()["coordinates"],
    }

    try:
        url = daily_int16.getDownloadURL(params)
        r = requests.get(url, stream=True, verify=False, timeout=300)
        if r.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(r.content)
            return filepath
    except Exception as e:
        print(f"Error downloading {init_date} day {forecast_day}: {e}")

    return None


def gfs_download_range(
    start_date: str,
    end_date: str,
    forecast_days: int = 7,
    output_dir: Optional[str] = None,
    n_jobs: int = 20,
):
    """
    Download GFS forecasts for a date range.

    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        forecast_days: Number of forecast days per init (default 7)
        output_dir: Output directory (default: config.data_dir/gfs)
        n_jobs: Parallel download jobs

    Returns:
        List of downloaded file paths
    """
    _init_ee()

    if output_dir is None:
        output_dir = os.path.join(config.data_dir, "gfs")

    # Build task list
    tasks = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        init_str = current.strftime("%Y-%m-%d")
        for day in range(1, forecast_days + 1):
            tasks.append((init_str, day, output_dir))
        current += timedelta(days=1)

    print(
        f"Downloading {len(tasks)} files ({len(tasks) // forecast_days} days × {forecast_days} forecast days)"
    )

    results = pqdm(
        tasks,
        download_gfs_day,
        n_jobs=n_jobs,
        argument_type="args",
        exception_behaviour="immediate",
        desc="Downloading GFS",
    )

    return [r for r in results if r is not None]


class GFSForecast(DSRC):
    """
    GFS Weather Forecast Data Source.

    Provides forecast weather data for up to 7 days ahead.
    Data is aggregated to daily values from hourly GFS forecasts.

    Usage:
        gfs = GFSForecast(forecast_days=1)  # 1-day forecast for next_day task
        gfs7 = GFSForecast(forecast_days=7)  # 7-day forecast

        # Standard DSRC interface - returns flattened forecast
        patch = gfs.get_sample(bbox, t, x_size, y_size)
        # Returns shape (forecast_days * 5, y_size, x_size)
    """

    name = "gfs_forecast"
    no_data = 0.0  # Use 0 for missing data (matches no_data_transform pattern)

    def __init__(self, forecast_days: int = 7):
        self.forecast_days = forecast_days
        self.data_dir = os.path.join(config.data_dir, "gfs")
        self.channel_names = []
        for day in range(1, self.forecast_days + 1):
            for band in GFS_BANDS:
                self.channel_names.append(f"day{day}_{band}")
        super().__init__()

    def _decode_int16(self, arr: np.ndarray, band_idx: int) -> np.ndarray:
        """Convert int16 back to float32."""
        band_name = GFS_BANDS[band_idx]
        bmin, bmax = GFS_BAND_CONFIG[band_name]
        return (arr.astype(np.float32) / 32767.0) * (bmax - bmin) + bmin

    def _get_file_path(self, init_date: datetime, forecast_day: int) -> str:
        """Get path to GFS file for given init date and forecast day."""
        date_str = init_date.strftime("%Y-%m-%d")
        return os.path.join(self.data_dir, f"gfs_{date_str}_day{forecast_day:02d}.tif")

    def _get_day(
        self,
        bbox: list[float],
        init_date: datetime,
        forecast_day: int,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """
        Get forecast data for a specific day.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            init_date: Forecast initialization date
            forecast_day: Which forecast day (1-7)
            x_size: Output width in pixels
            y_size: Output height in pixels

        Returns:
            Array of shape (5, y_size, x_size) with decoded float32 values,
            or None if data not available
        """
        filepath = self._get_file_path(init_date, forecast_day)
        if not os.path.exists(filepath):
            return None

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ds = gdal.Open(filepath)
        if ds is None:
            return None

        xoff, yoff, src_xsize, src_ysize = self._get_src_patch(bbox, ds)

        # Read as int16
        patch_int16 = np.zeros((len(GFS_BANDS), y_size, x_size), dtype=np.int16)
        ds.ReadAsArray(
            xoff,
            yoff,
            src_xsize,
            src_ysize,
            buf_obj=patch_int16,
            resample_alg=gdalconst.GRIORA_Bilinear,
        )

        # Decode each band
        patch = np.zeros((len(GFS_BANDS), y_size, x_size), dtype=np.float32)
        for i in range(len(GFS_BANDS)):
            patch[i] = self._decode_int16(patch_int16[i], i)

        return patch

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """
        Standard DSRC interface - get flattened forecast for all days.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            t: Forecast initialization date
            x_size: Output width in pixels
            y_size: Output height in pixels

        Returns:
            Array of shape (forecast_days * 5, y_size, x_size) with decoded float32 values,
            or None if any day is missing.

            Channel order: day1_temp_max, day1_rh_min, ..., day2_temp_max, day2_rh_min, ...
        """
        sequence = []
        for day in range(1, self.forecast_days + 1):
            patch = self._get_day(bbox, t, day, x_size, y_size)
            if patch is None:
                return None
            sequence.append(patch)

        # Stack and flatten: (days, 5, H, W) -> (days * 5, H, W)
        stacked = np.stack(sequence, axis=0)
        return stacked.reshape(-1, y_size, x_size)

    def get_forecast_sequence(
        self,
        bbox: list[float],
        init_date: datetime,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """
        Get all forecast days as a sequence.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            init_date: Forecast initialization date
            x_size: Output width in pixels
            y_size: Output height in pixels

        Returns:
            Array of shape (forecast_days, 5, y_size, x_size),
            or None if any day is missing
        """
        sequence = []
        for day in range(1, self.forecast_days + 1):
            patch = self._get_day(bbox, init_date, day, x_size, y_size)
            if patch is None:
                return None
            sequence.append(patch)

        return np.stack(sequence, axis=0)

    # Backwards compatibility alias
    def get_forecast_flattened(
        self,
        bbox: list[float],
        init_date: datetime,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """Alias for get_sample() - kept for backwards compatibility."""
        return self.get_sample(bbox, init_date, x_size, y_size)

    def _get_native_resolution(self, t: datetime) -> Optional[tuple[float, float]]:
        """Get native pixel size (dx, dy) in degrees from the GeoTIFF."""
        filepath = self._get_file_path(t, 1)  # Use day 1 file
        if not os.path.exists(filepath):
            return None
        
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ds = gdal.Open(filepath)
        if ds is None:
            return None
        
        gt = ds.GetGeoTransform()
        # gt[1] = pixel width, gt[5] = pixel height (negative)
        dx = abs(gt[1])
        dy = abs(gt[5])
        ds = None
        return dx, dy

def fix_missing_gfs(data_dir: Optional[str] = None, n_jobs: int = 20):
    """
    Scan for missing GFS files and download only those from EE.
    """
    import re

    if data_dir is None:
        data_dir = os.path.join(config.data_dir, "gfs")

    # Scan existing files
    files = os.listdir(data_dir)
    existing = set()
    dates = set()
    for f in files:
        m = re.match(r"gfs_(\d{4}-\d{2}-\d{2})_day(\d+)\.tif", f)
        if m:
            existing.add((m.group(1), int(m.group(2))))
            dates.add(m.group(1))

    sorted_dates = sorted(dates)
    start = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
    end = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")

    # Find all missing (date, day) pairs
    missing = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        for day in range(1, 8):
            if (ds, day) not in existing:
                missing.append((ds, day, data_dir))
        d += timedelta(days=1)

    if not missing:
        print("No missing files!")
        return

    print(f"Found {len(missing)} missing files to download from EE")
    for init_str, day, _ in missing:
        print(f"  gfs_{init_str}_day{day:02d}.tif")

    _init_ee()

    results = pqdm(
        missing,
        download_gfs_day,
        n_jobs=n_jobs,
        argument_type="args",
        exception_behaviour="immediate",
        desc="Downloading missing GFS",
    )

    downloaded = [r for r in results if r is not None]
    failed = len(missing) - len(downloaded)
    print(f"\nDownloaded: {len(downloaded)}/{len(missing)}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Quick test download
        _init_ee()
        output_dir = os.path.expanduser("~/data/gfs_test")
        path = download_gfs_day("2018-06-15", 1, output_dir)
        print(f"Downloaded: {path}")

    elif "--download" in sys.argv:
        # Full download
        # Usage: python -m firecomp.dsrc.gfs --download 2018-01-01 2024-12-31
        if len(sys.argv) >= 4:
            start = sys.argv[sys.argv.index("--download") + 1]
            end = sys.argv[sys.argv.index("--download") + 2]
            gfs_download_range(start, end)
        else:
            print("Usage: python -m firecomp.dsrc.gfs --download START_DATE END_DATE")

    elif "--fix" in sys.argv:
        fix_missing_gfs()
