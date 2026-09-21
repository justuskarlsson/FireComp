"""WeatherNext 2 forecast data source.

Provides 7-day forecasts from Google's WeatherNext 2 model via Earth Engine.
Used as a complement to ERA5 current-day weather data.

Channels per day:
- forecast_precip_sum: daily precipitation sum (m)
- forecast_wind_speed_max: daily max wind speed (m/s)
- forecast_mslp_min: daily min sea-level pressure (Pa) - optional

With 7 days and 2-3 channels per day, total channels = 14-21.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from typing import Optional

import numpy as np
from osgeo import gdal, gdalconst

from firecomp.config import config
from firecomp.dsrc.dsrc import DSRC, create_tif, deg_per_px_from_m_equator
from firecomp.dsrc.earth_engine import DownloadOptions, download_image, init_ee
from firecomp.helpers import pipeline


WEATHERNEXT2_COLLECTION = (
    "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
)

# Dataset native resolution ~0.25 degrees (about 27.8 km at equator).
WEATHERNEXT2_SCALE_M = 27_830


@dataclass
class WeatherNext2Config:
    include_pressure: bool = True
    ensemble_member: str = "0"  # Use member 0 for deterministic forecast
    forecast_days: int = 7  # Number of days to forecast


class WeatherNext2Forecast(DSRC):
    """Multi-day WeatherNext 2 forecast fields.

    Stores one GeoTIFF per initialization date with all forecast days stacked.
    Shape: (days * channels_per_day, H, W)

    Channels per day: precip_sum, wind_speed_max, [mslp_min]
    """

    name = "weathernext2_forecast"
    no_data = 0.0

    def __init__(self, cfg: WeatherNext2Config | None = None):
        self.cfg = cfg or WeatherNext2Config()

        # Channels per day (set before super().__init__ in case parent accesses channel_names)
        self._day_channels = ["precip_sum", "wind_speed_max"]
        if self.cfg.include_pressure:
            self._day_channels.append("mslp_min")

        # Full channel names: day0_precip_sum, day0_wind_speed_max, day1_precip_sum, ...
        channel_names = []
        for d in range(self.cfg.forecast_days):
            for ch in self._day_channels:
                channel_names.append(f"day{d}_{ch}")

        super().__init__(channel_names=channel_names)
        self.data_dir = os.path.join(config.data_dir, "weathernext2_forecast")

    @property
    def channels_per_day(self) -> int:
        return len(self._day_channels)

    def _forecast_path(self, init_date: datetime) -> str:
        return os.path.join(self.data_dir, f"{init_date:%Y-%m-%d}.tif")

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """Get forecast data initialized at date t."""
        path = self._forecast_path(t)
        if not os.path.exists(path):
            return None
        ds: gdal.Dataset = gdal.Open(path)
        if ds is None:
            return None
        xoff, yoff, src_xsize, src_ysize = self._get_src_patch(bbox, ds)
        n_channels = len(self.channel_names)
        patch = np.zeros((n_channels, y_size, x_size), dtype=np.float32)
        ds.ReadAsArray(
            xoff,
            yoff,
            src_xsize,
            src_ysize,
            band_list=list(range(1, n_channels + 1)),
            buf_obj=patch,
            resample_alg=gdalconst.GRIORA_Bilinear,
        )
        return patch

    @pipeline
    def download_forecast(
        self,
        init_date: str,
        *,
        bbox: Optional[list[float]] = None,
        scale_m: int = WEATHERNEXT2_SCALE_M,
    ):
        """Download multi-day forecast starting from init_date.

        Args:
            init_date: Initialization date in YYYY-MM-DD format
            bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
            scale_m: Output resolution in meters
        """
        import ee

        init_ee()
        t0 = datetime.strptime(init_date, "%Y-%m-%d")
        dst_path = self._forecast_path(t0)
        os.makedirs(self.data_dir, exist_ok=True)

        if os.path.exists(dst_path):
            print(f"Already exists: {dst_path}")
            return dst_path

        all_bands = []

        for day_offset in range(self.cfg.forecast_days):
            # Forecast lead hours for this day (hours 6, 12, 18, 24 for day 0, etc.)
            lead_start = day_offset * 24 + 6
            lead_end = (day_offset + 1) * 24
            lead_hours = list(range(lead_start, lead_end + 1, 6))

            # Filter collection for this init date and lead hours
            start = ee.Date(t0.strftime("%Y-%m-%dT00:00:00Z"))
            end = start.advance(1, "day")

            coll = (
                ee.ImageCollection(WEATHERNEXT2_COLLECTION)
                .filter(ee.Filter.date(start, end))
                .filter(ee.Filter.eq("ensemble_member", self.cfg.ensemble_member))
                .filter(ee.Filter.inList("forecast_hour", lead_hours))
            )

            # Precipitation sum
            precip = coll.select("total_precipitation_6hr").sum().rename(
                f"day{day_offset}_precip_sum"
            )

            # Wind speed max
            def wind_speed(img):
                u = img.select("10m_u_component_of_wind")
                v = img.select("10m_v_component_of_wind")
                return u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed")

            wind_max = coll.map(wind_speed).max().rename(
                f"day{day_offset}_wind_speed_max"
            )

            all_bands.extend([precip, wind_max])

            if self.cfg.include_pressure:
                mslp_min = coll.select("mean_sea_level_pressure").min().rename(
                    f"day{day_offset}_mslp_min"
                )
                all_bands.append(mslp_min)

        image = ee.Image.cat(all_bands)

        rect = ee.Geometry.Rectangle(bbox) if bbox is not None else None
        download_image(
            image,
            dst_path,
            rect=rect,
            scale=scale_m,
            options=DownloadOptions(crs="EPSG:4326"),
        )
        print(f"Downloaded: {dst_path}")
        return dst_path


def test_download_single_day(
    date: str = "2024-01-15",
    bbox: list[float] = [-10.0, 35.0, 5.0, 45.0],  # Small region over Spain
):
    """Test downloading a single day's forecast for a small region.

    This verifies:
    1. EE authentication works
    2. Collection filtering works
    3. Aggregation logic (sum, max, min) works
    4. Download and GeoTIFF creation works

    Run with: python -m firecomp.dsrc.weathernext2 test
    """
    import ee

    print(f"Testing WeatherNext 2 download for {date}")
    print(f"Bbox: {bbox}")

    cfg = WeatherNext2Config(include_pressure=True, forecast_days=2)
    dsrc = WeatherNext2Forecast(cfg)

    print(f"Channels: {dsrc.channel_names}")
    print(f"Channels per day: {dsrc.channels_per_day}")

    # Download
    path = dsrc.download_forecast(date, bbox=bbox)
    print(f"Downloaded to: {path}")

    # Verify output
    ds: gdal.Dataset = gdal.Open(path)
    if ds is None:
        print("ERROR: Failed to open downloaded file")
        return

    print(f"Bands: {ds.RasterCount}")
    print(f"Size: {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"GeoTransform: {ds.GetGeoTransform()}")

    # Read and print stats
    arr = ds.ReadAsArray()
    print(f"Shape: {arr.shape}")
    for i, name in enumerate(dsrc.channel_names):
        band = arr[i]
        print(f"  {name}: min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}")

    ds = None
    print("Test passed!")


def test_collection_info(date: str = "2024-01-15"):
    """Print info about what's available in the collection for a date.

    Run with: python -m firecomp.dsrc.weathernext2 info
    """
    import ee

    init_ee()

    t = datetime.strptime(date, "%Y-%m-%d")
    start = ee.Date(t.strftime("%Y-%m-%dT00:00:00Z"))
    end = start.advance(1, "day")

    coll = ee.ImageCollection(WEATHERNEXT2_COLLECTION).filter(
        ee.Filter.date(start, end)
    )

    count = coll.size().getInfo()
    print(f"Images for {date}: {count}")

    if count == 0:
        print("No images found. WeatherNext 2 may not have data for this date.")
        return

    # Get first image info
    first = ee.Image(coll.first())
    props = first.getInfo()["properties"]

    print(f"Properties of first image:")
    for k, v in props.items():
        print(f"  {k}: {v}")

    print(f"\nBand names: {first.bandNames().getInfo()}")

    # Check ensemble members and forecast hours
    members = coll.aggregate_array("ensemble_member").distinct().getInfo()
    hours = coll.aggregate_array("forecast_hour").distinct().sort().getInfo()
    print(f"\nEnsemble members: {members}")
    print(f"Forecast hours: {hours}")


if __name__ == "__main__":
    import fire

    fire.Fire({
        "test": test_download_single_day,
        "info": test_collection_info,
    })
