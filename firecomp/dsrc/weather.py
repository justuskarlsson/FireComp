from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
import math
import os
from typing import Optional
import h5py
from tqdm import tqdm

from firecomp.config import config
from firecomp.dsrc import dsrc
from firecomp.dsrc.dsrc import DSRC, create_tif, pipeline, TIF_CREATE_OPTIONS
from firecomp.dsrc.earth_engine import (
    ee_image_get_date_str,
    ee_to_list,
    download_image,
    init_ee,
)
from pqdm.threads import pqdm
from datetime import date, datetime
from osgeo import gdal, gdalconst, osr
import numpy as np

gdal.UseExceptions()


@dataclass
class Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    padding: float

    def extent(self, xi, yi):
        return [
            max(-180, -180 + xi * self.dx - self.padding),
            max(-90, -90 + yi * self.dy - self.padding),
            min(180, -180 + (xi + 1) * self.dx + self.padding),
            min(90, -90 + (yi + 1) * self.dy + self.padding),
        ]

    @staticmethod
    def from_nx(nx, padding=1.0):
        return Grid(nx, nx // 2, 360 / nx, 360 / nx, padding)

    def find_cell(self, lon, lat):
        xi = int((lon + 180) / self.dx)
        yi = int((lat + 90) / self.dy)

        return xi, yi


era5_grid = Grid.from_nx(8)
era5_bands = [
    # Moisture stress
    "volumetric_soil_water_layer_1_min",
    "volumetric_soil_water_layer_1_max",
    "volumetric_soil_water_layer_2_min",
    "evaporation_from_vegetation_transpiration_sum",
    "total_evaporation_sum",
    "potential_evaporation_sum",
    "skin_reservoir_content_min",
    "skin_reservoir_content_max",
    # Atmospheric dryness
    "dewpoint_temperature_2m_min",
    "dewpoint_temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_max",
    "surface_latent_heat_flux_sum",
    "surface_sensible_heat_flux_sum",
    # Heating & radiation
    "skin_temperature_max",
    "soil_temperature_level_1_max",
    "surface_net_solar_radiation_sum",
    # Hydrology
    "runoff_sum",
    "surface_runoff_sum",
    "snow_depth_min",
    "snow_cover_min",
    # Wind / spread potential
    "u_component_of_wind_10m_max",
    "v_component_of_wind_10m_max",
]


def weather_download(
    start_date: str,
    end_date: str,
    grid: Grid,
    name: str,
    scale: int,
    collection_name: str,
    bands: Optional[list[str]] = None,
):
    import ee

    init_ee()
    dst_dir = os.path.join(config.data_dir, name)
    os.makedirs(dst_dir, exist_ok=True)
    collection = ee.ImageCollection(collection_name)
    collection = collection.filterDate(start_date, end_date)
    if bands is not None:
        collection = collection.select(bands)

    images = ee_to_list(collection)
    print(len(images), "images ready to download")

    def get_args(image):
        args = []
        date_str = ee_image_get_date_str(image)
        for xi in range(grid.nx):
            for yi in range(grid.ny):
                rect = ee.Geometry.Rectangle(grid.extent(xi, yi))
                path = dst_dir + f"/{date_str}_{xi}_{yi}.tif"
                if os.path.exists(path):
                    continue
                args.append((image, path, rect, scale))
        return args

    argss = pqdm(images, get_args, n_jobs=20, desc="Getting args")
    args = []
    for args_list in argss:
        args.extend(args_list)
    paths = pqdm(
        args, download_image, n_jobs=20, argument_type="args", desc="Downloading images"
    )

    return paths


def era5_download(start_date: str, end_date: str):
    return weather_download(
        start_date,
        end_date,
        era5_grid,
        "era5",
        11132,
        "ECMWF/ERA5_LAND/DAILY_AGGR",
        era5_bands,
    )


def get_series(lon, lat, start_date, end_date, grid: Grid, name: str):
    xi, yi = grid.find_cell(lon, lat)
    print(xi, yi)
    paths = glob(os.path.join(config.data_dir, name, f"*{xi}_{yi}.tif"))
    paths.sort()
    args = []
    for path in paths:
        fn = os.path.basename(path)
        date_str = fn.split("_")[0]
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt < start_date or dt > end_date:
            continue
        args.append(path)

    def get_value(path):
        # Suppress GDAL warnings about TIFFReadDirectory color channels mismatch
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ds: gdal.Dataset = gdal.Open(path)
        gt = ds.GetGeoTransform()
        px = int(round((lon - gt[0]) / gt[1]))
        py = int(round((lat - gt[3]) / gt[5]))
        px = max(0, min(px, ds.RasterXSize - 1))
        py = max(0, min(py, ds.RasterYSize - 1))
        data = ds.ReadAsArray(px, py, 1, 1)
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        return data[:, 0, 0].tolist()

    values = pqdm(
        args, get_value, n_jobs=10, desc=f"Getting {len(args)} weather values"
    )
    # T x B
    values = np.array(values).T  # B x T
    return values


def test_get_series(name: str = "era5"):
    # Sahara Desert: approx lon=13, lat=23
    vals = get_series(
        13, 23, datetime(2024, 7, 1), datetime(2024, 11, 1), era5_grid, name
    )
    print(vals)
    # from matplotlib import pyplot as plt
    # plt.plot(vals[0])
    # plt.show()
    # plt.savefig("test.png")


class ERA5(DSRC):
    name = "era5"
    channel_names = era5_bands
    no_data = 0.0

    def __init__(self):
        super().__init__()
        self.grid = era5_grid
        self.data_dir = os.path.join(config.data_dir, "era5")

    def get_sample(self, bbox: list[float], t: datetime, size: int):
        date_str = t.strftime("%Y-%m-%d")
        path = os.path.join(self.data_dir, f"{date_str}.tif")
        ds: gdal.Dataset = gdal.Open(path)

        xoff, yoff, xsize, ysize = self._get_src_patch(bbox, ds)
        patch = np.zeros((len(self.channel_names), size, size), dtype=np.float32)
        ds.ReadAsArray(
            xoff,
            yoff,
            xsize,
            ysize,
            buf_obj=patch,
            resample_alg=gdalconst.GRIORA_Bilinear,
        )
        return patch

    def get_merged_paths_by_date(self):
        all_paths = os.listdir(self.data_dir)
        all_paths.sort()
        paths = []
        for fn in all_paths:
            if "_" not in fn:
                path = os.path.join(self.data_dir, fn)
                paths.append(path)
        by_date = dict()
        for path in paths:
            date_str = os.path.basename(path).replace(".tif", "")
            day = datetime.strptime(date_str, "%Y-%m-%d")
            by_date[day].append(path)
        return by_date

    @pipeline
    def merge_spatial(self):
        paths = glob(os.path.join(self.data_dir, "*_*_*.tif"))
        paths.sort()
        by_date = defaultdict(list)
        for path in paths:
            fn = os.path.basename(path)
            date_str, xi, yi = fn.split("_")
            by_date[date_str].append(path)

        def merge(date_str, paths):
            out = os.path.join(self.data_dir, f"{date_str}.tif")
            px_deg = 0.100000457428185
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            ds = create_tif(
                out,
                [-180.0, -90.0, 180.0, 90.0],
                px_deg,
                self.no_data,
                len(self.channel_names),
            )
            ds.FlushCache()
            ds = None
            dst_ds: gdal.Dataset = gdal.Open(out, gdal.GA_Update)
            wopts = gdal.WarpOptions(
                resampleAlg=gdalconst.GRA_NearestNeighbour,
                dstNodata=self.no_data,
                warpOptions=[
                    "UNIFIED_SRC_NODATA=YES",
                    "SKIP_NOSOURCE=YES",
                ],
            )
            for path in paths:
                src_ds: gdal.Dataset = gdal.Open(path)
                gdal.Warp(dst_ds, src_ds, options=wopts)
                src_ds = None
            dst_ds.FlushCache()
            dst_ds = None

        print(f"starting merges {len(by_date)}")
        args = list(by_date.items())
        pqdm(args, merge, n_jobs=20, desc="Merging", argument_type="args")
        # merge("2015-02-14", by_date["2015-02-14"])


def compute_vpd(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Compute Vapor Pressure Deficit in kPa.

    VPD = saturation vapor pressure - actual vapor pressure
    Higher VPD = drier air = more evaporative demand on vegetation

    Args:
        temp_c: Air temperature in Celsius
        dewpoint_c: Dewpoint temperature in Celsius

    Returns:
        VPD in kPa (typically 0-5 kPa range)
    """
    # Clamp temperatures to physically reasonable range to avoid overflow in exp
    # Valid range: -90°C (coldest recorded) to 60°C (hottest recorded + margin)
    temp_c = np.clip(temp_c, -90, 60)
    dewpoint_c = np.clip(dewpoint_c, -90, 60)

    # Saturation vapor pressure (Tetens formula)
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    # Actual vapor pressure from dewpoint
    ea = 0.6108 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))
    return np.maximum(es - ea, 0.0)  # VPD cannot be negative


def compute_ratio_anomaly(
    current: np.ndarray,
    past: np.ndarray,
    min_denominator: float = 0.05,
    clip_range: tuple[float, float] = (0.2, 3.0),
) -> np.ndarray:
    """Compute ratio-based anomaly: current / past.

    Args:
        current: Current value array
        past: Reference (e.g., 1-year-ago) value array
        min_denominator: Minimum denominator to avoid division by zero
        clip_range: (min, max) to clip extreme ratios

    Returns:
        Ratio array where:
        - 1.0 = same as reference
        - <1.0 = below normal (e.g., 0.5 = 50% of normal)
        - >1.0 = above normal (e.g., 1.5 = 150% of normal)
    """
    safe_past = np.maximum(np.abs(past), min_denominator)
    ratio = current / safe_past
    return np.clip(ratio, clip_range[0], clip_range[1])


# Band indices in era5_bands for quick lookup
ERA5_BAND_IDX = {name: i for i, name in enumerate(era5_bands)}


class ERA5Fire(DSRC):
    """Processed ERA5 weather inputs optimized for fire spread prediction.

    Computes derived variables and anomalies from raw ERA5 data:
    - VPD (vapor pressure deficit) - atmospheric dryness
    - Soil moisture ratio anomaly (current / 1yr ago) - drought indicator
    - Wind magnitude and direction (sin/cos encoded)
    """

    name = "era5_fire"
    channel_names = [
        "vpd",  # Vapor pressure deficit (kPa)
        "soil_moisture",  # Volumetric soil water (avg of layer 1 min/max)
        "soil_moisture_ratio",  # current / 1yr ago (1.0 = normal)
        "wind_magnitude",  # sqrt(u^2 + v^2) m/s
        "wind_sin",  # sin(wind_direction) for directional encoding
        "wind_cos",  # cos(wind_direction) for directional encoding
    ]
    no_data = 0.0

    def __init__(self):
        super().__init__()
        self.grid = era5_grid
        self.data_dir = os.path.join(config.data_dir, "era5")
        self._era5 = ERA5()

    def _get_tif_path(self, t: datetime) -> str:
        return os.path.join(self.data_dir, f"{t.strftime('%Y-%m-%d')}.tif")

    def _load_era5_day(
        self,
        t: datetime,
        bbox: list[float],
        x_size: int,
        y_size: int,
        band_list: list[str],
    ) -> Optional[np.ndarray]:
        """Load raw ERA5 data for a specific day."""
        path = self._get_tif_path(t)
        if not os.path.exists(path):
            return None

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ds: gdal.Dataset = gdal.Open(path)
        if ds is None:
            return None

        xoff, yoff, src_xsize, src_ysize = self._get_src_patch(bbox, ds)
        patch = np.zeros((len(band_list), y_size, x_size), dtype=np.float32)
        ds.ReadAsArray(
            xoff,
            yoff,
            src_xsize,
            src_ysize,
            band_list=band_list,
            buf_obj=patch,
            resample_alg=gdalconst.GRIORA_Bilinear,
        )
        return patch

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        x_size: int,
        y_size: int,
        past_offset_days: int = 365,
    ) -> Optional[np.ndarray]:
        from datetime import timedelta

        # Load current day
        src_bands = [
            "temperature_2m_max",
            "dewpoint_temperature_2m_max",
            "volumetric_soil_water_layer_1_min",
            "volumetric_soil_water_layer_1_max",
            "u_component_of_wind_10m_max",
            "v_component_of_wind_10m_max",
        ]

        current = self._load_era5_day(
            t, bbox, x_size, y_size, [ERA5_BAND_IDX[name] + 1 for name in src_bands]
        )
        if current is None:
            return None
        (temp_max, dewpoint_max, soil_min, soil_max, u_wind, v_wind) = current

        # Load reference day (1 year ago by default)
        past_date = t - timedelta(days=past_offset_days)
        src_bands_prev = [
            "volumetric_soil_water_layer_1_min",
            "volumetric_soil_water_layer_1_max",
        ]

        past = self._load_era5_day(
            past_date,
            bbox,
            x_size,
            y_size,
            [ERA5_BAND_IDX[name] + 1 for name in src_bands_prev],
        )
        if past is None:
            # Fall back to current if no historical data
            print(f"Warning: No historical data found for {past_date}, using current")
            past_soil_min = soil_min
            past_soil_max = soil_max
        else:
            past_soil_min, past_soil_max = past

        # Compute derived variables
        # 1. VPD (convert Kelvin to Celsius if needed - ERA5 uses Kelvin)
        temp_c = temp_max - 273.15 if temp_max.mean() > 100 else temp_max
        dewpoint_c = (
            dewpoint_max - 273.15 if dewpoint_max.mean() > 100 else dewpoint_max
        )
        vpd = compute_vpd(temp_c, dewpoint_c)

        # 2. Soil moisture (average of min/max)
        soil_moisture = (soil_min + soil_max) / 2.0
        past_soil_moisture = (past_soil_min + past_soil_max) / 2.0

        # 3. Soil moisture ratio anomaly
        soil_ratio = compute_ratio_anomaly(soil_moisture, past_soil_moisture)

        # 4. Wind magnitude and direction
        wind_magnitude = np.sqrt(u_wind**2 + v_wind**2)
        wind_direction = np.arctan2(v_wind, u_wind)  # radians
        wind_sin = np.sin(wind_direction)
        wind_cos = np.cos(wind_direction)

        # Stack all channels: (6, H, W)
        result = np.stack(
            [vpd, soil_moisture, soil_ratio, wind_magnitude, wind_sin, wind_cos],
            axis=0,
        ).astype(np.float32)

        return result


class ERA5FirePercentile(DSRC):
    """ERA5 weather inputs using percentile-based drought anomaly.

    Requires pre-computed climatology from build_global_climatology().
    Uses percentile rank (0-100) instead of ratio for soil moisture anomaly.

    Channels:
    - vpd: Vapor pressure deficit (kPa)
    - soil_moisture: Current volumetric soil water
    - soil_percentile: Percentile rank (0=driest on record, 100=wettest)
    - wind_magnitude, wind_sin, wind_cos: Wind speed and direction
    """

    name = "era5_fire_pct"
    channel_names = [
        "vpd",
        "soil_moisture",
        "soil_percentile",  # 0-100 percentile rank
        "wind_magnitude",
        "wind_sin",
        "wind_cos",
    ]
    no_data = 0.0

    def __init__(self, climatology_dir: Optional[str] = None):
        super().__init__()
        self.grid = era5_grid
        self.data_dir = os.path.join(config.data_dir, "era5")
        if climatology_dir is None:
            climatology_dir = os.path.join(config.data_dir, "era5_climatology")
        self.climatology = SoilMoistureClimatology(climatology_dir)
        self._era5_fire = ERA5Fire()

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """Get processed weather sample with percentile-based drought anomaly.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            t: Current date
            x_size: Output patch width (pixels)
            y_size: Output patch height (pixels)

        Returns:
            Array of shape (6, y_size, x_size) with channels:
            [vpd, soil_moisture, soil_percentile, wind_mag, wind_sin, wind_cos]
        """
        # Get base ERA5Fire sample (has ratio-based anomaly)
        base = self._era5_fire.get_sample(bbox, t, x_size, y_size)
        if base is None:
            return None

        # Replace ratio anomaly (channel 2) with percentile rank
        soil_moisture = base[1]  # Current soil moisture
        doy = t.timetuple().tm_yday

        soil_percentile = self.climatology.get_percentile(
            bbox, doy, soil_moisture, x_size, y_size
        )
        if soil_percentile is None:
            # Fall back to ratio if climatology unavailable
            return base

        # Normalize percentile to 0-1 range for model input
        soil_percentile = soil_percentile / 100.0

        result = np.stack(
            [
                base[0],  # vpd
                base[1],  # soil_moisture
                soil_percentile,  # percentile (0-1)
                base[3],  # wind_magnitude
                base[4],  # wind_sin
                base[5],  # wind_cos
            ],
            axis=0,
        ).astype(np.float32)

        return result


def compute_percentile_rank(
    value: np.ndarray,
    historical_values: np.ndarray,
) -> np.ndarray:
    """Compute percentile rank of value against historical distribution.

    Complexity: O(N × H × W) where N = number of historical samples
    Memory: O(N × H × W) - needs all samples in memory

    Args:
        value: Current value array (H, W)
        historical_values: Historical values array (N, H, W) for same day-of-year

    Returns:
        Percentile rank array (H, W) in range [0, 100]
        where 0 = driest on record, 100 = wettest on record
    """
    # Count how many historical values are <= current value
    # Shape: (N, H, W) compared to (1, H, W) -> (N, H, W) -> sum over N -> (H, W)
    n_below = np.sum(historical_values <= value[np.newaxis, ...], axis=0)
    n_total = historical_values.shape[0]
    percentile = (n_below / n_total) * 100.0
    return percentile.astype(np.float32)


# =============================================================================
# Climatology Building - Optimized for IO
# =============================================================================
#
# Strategy: Pre-compute global climatology statistics ONCE, then query cheaply.
#
# Two-phase approach:
#   Phase 1 (offline, run once): build_global_climatology()
#     - Reads each global TIF once
#     - Computes percentile thresholds for each DOY at each pixel
#     - Stores result as 366 TIFs (one per DOY)
#     - IO: O(D) where D = total days of data (~4000 files read once)
#
#   Phase 2 (runtime, per fire): query_percentile()
#     - Single windowed read from pre-computed percentile TIF
#     - IO: O(1) file open per query
# =============================================================================


class _ERA5Cache:
    """LRU-style cache for ERA5 band reads. Avoids re-reading overlapping files."""

    def __init__(self, max_size_mb: int = 3000):
        self.cache: dict[str, np.ndarray] = {}
        self.access_order: list[str] = []  # Most recent at end
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        self.hits = 0
        self.misses = 0

    def get(self, path: str, band_idx: int) -> Optional[np.ndarray]:
        """Get array from cache or read from disk."""
        if path in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(path)
            self.access_order.append(path)
            self.hits += 1
            return self.cache[path]

        # Cache miss - read from disk
        self.misses += 1
        if not os.path.exists(path):
            return None

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ds = gdal.Open(path)
        if ds is None:
            return None

        band = ds.GetRasterBand(band_idx + 1)
        arr = band.ReadAsArray().astype(np.float32)
        ds = None

        # Add to cache
        arr_size_mb = arr.nbytes / (1024 * 1024)
        self._evict_if_needed(arr_size_mb)
        self.cache[path] = arr
        self.access_order.append(path)
        self.current_size_mb += arr_size_mb

        return arr

    def _evict_if_needed(self, new_size_mb: float):
        """Evict oldest entries if adding new_size_mb would exceed max."""
        while (
            self.current_size_mb + new_size_mb > self.max_size_mb and self.access_order
        ):
            oldest_path = self.access_order.pop(0)
            if oldest_path in self.cache:
                evicted = self.cache.pop(oldest_path)
                self.current_size_mb -= evicted.nbytes / (1024 * 1024)

    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache: {self.hits}/{total} hits ({hit_rate:.1f}%), {self.current_size_mb:.0f}MB used"


def build_global_climatology(
    years: list[int],
    output_dir: str,
    window_days: int = 15,
    percentiles: list[int] = [10, 25, 50, 75, 90],
    cache_size_mb: int = 12000,
):
    """Pre-compute global soil moisture climatology for all DOYs.

    Run this ONCE offline. Creates 366 TIF files with percentile thresholds.
    Uses LRU cache to avoid re-reading overlapping files between consecutive DOYs.

    RAM Requirements:
        - Cache: cache_size_mb (default 12GB)
        - Stack operation: ~10GB temporary (for 13 years × 31 days window)
        - Recommended total: 32GB+ for full 13-year run
        - With 32GB: ~30-40 min runtime
        - With 128GB: can increase cache_size_mb to 100000 for ~15-20 min runtime

    Args:
        years: Years to include (e.g., [2012, ..., 2024])
        output_dir: Where to save climatology TIFs
        window_days: Smoothing window (±days around each DOY)
        percentiles: Which percentiles to pre-compute
        cache_size_mb: Max cache size in MB (default 12000 = 12GB, needs ~32GB total RAM)
    """
    from datetime import timedelta

    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(config.data_dir, "era5")
    soil_idx = ERA5_BAND_IDX["volumetric_soil_water_layer_1_min"]

    # Get dimensions from first available file
    sample_path = None
    for year in years:
        p = os.path.join(data_dir, f"{year}-06-15.tif")
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError("No ERA5 files found")

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    sample_ds = gdal.Open(sample_path)
    width, height = sample_ds.RasterXSize, sample_ds.RasterYSize
    gt = sample_ds.GetGeoTransform()
    srs = sample_ds.GetSpatialRef()
    sample_ds = None

    # Initialize cache
    cache = _ERA5Cache(max_size_mb=cache_size_mb)

    # Process each DOY
    for doy in tqdm(range(1, 367), desc="Building climatology"):
        # Collect all values for this DOY (with window)
        all_values = []

        for year in years:
            try:
                center_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            except ValueError:
                continue

            for offset in range(-window_days, window_days + 1):
                target_date = center_date + timedelta(days=offset)
                date_str = target_date.strftime("%Y-%m-%d")
                path = os.path.join(data_dir, f"{date_str}.tif")

                arr = cache.get(path, soil_idx)
                if arr is not None:
                    all_values.append(arr)

        if len(all_values) < 5:
            print(f"  DOY {doy}: insufficient data ({len(all_values)} samples)")
            continue

        # Stack and compute percentiles: (N, H, W) -> (P, H, W)
        stacked = np.stack(all_values, axis=0)  # (N, H, W)
        pct_values = np.percentile(stacked, percentiles, axis=0)  # (P, H, W)

        # Save as TIF
        out_path = os.path.join(output_dir, f"climatology_doy_{doy:03d}.tif")
        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(
            out_path,
            width,
            height,
            len(percentiles),
            gdal.GDT_Float32,
            options=TIF_CREATE_OPTIONS,
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetSpatialRef(srs)
        for i, pct in enumerate(percentiles):
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(pct_values[i])
            band.SetDescription(f"p{pct}")
        out_ds.FlushCache()
        out_ds = None

    print(cache.stats())
    print(f"Climatology saved to {output_dir}")


class SoilMoistureClimatology:
    """Query pre-computed soil moisture climatology.

    Usage:
        clim = SoilMoistureClimatology("/path/to/climatology")
        percentile = clim.get_percentile(bbox, doy, current_value)

    IO per query: O(1) - single windowed read from pre-computed TIF
    """

    def __init__(self, climatology_dir: str):
        self.climatology_dir = climatology_dir
        self.percentiles = [10, 25, 50, 75, 90]
        self._cache = {}  # DOY -> opened dataset (LRU would be better)

    def _get_ds(self, doy: int) -> Optional[gdal.Dataset]:
        """Get (cached) dataset for a DOY."""
        if doy not in self._cache:
            path = os.path.join(self.climatology_dir, f"climatology_doy_{doy:03d}.tif")
            if not os.path.exists(path):
                return None
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            self._cache[doy] = gdal.Open(path)
        return self._cache[doy]

    def get_percentile(
        self,
        bbox: list[float],
        doy: int,
        current_value: np.ndarray,
        x_size: int,
        y_size: int,
    ) -> Optional[np.ndarray]:
        """Get percentile rank for current soil moisture value.

        IO: Single windowed read from climatology TIF.
        Complexity: O(P × H × W) where P = number of percentiles (5)

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            doy: Day of year (1-366)
            current_value: Current soil moisture array (H, W)
            x_size: Output patch width (pixels)
            y_size: Output patch height (pixels)

        Returns:
            Percentile rank array (H, W) in range [0, 100]
        """
        ds = self._get_ds(doy)
        if ds is None:
            return None

        # Single windowed read for all percentile bands
        era5 = ERA5()
        xoff, yoff, xsize, ysize = era5._get_src_patch(bbox, ds)
        patch = np.zeros((len(self.percentiles), y_size, x_size), dtype=np.float32)
        ds.ReadAsArray(
            xoff,
            yoff,
            xsize,
            ysize,
            buf_obj=patch,
            resample_alg=gdalconst.GRIORA_Bilinear,
        )

        # Interpolate percentile from thresholds
        # patch shape: (5, H, W) for percentiles [10, 25, 50, 75, 90]
        # Find where current_value falls in the distribution
        result = np.zeros_like(current_value)

        for i, (pct_lo, pct_hi) in enumerate(
            zip([0] + self.percentiles[:-1], self.percentiles)
        ):
            if i == 0:
                # Below p10
                mask = current_value <= patch[0]
                result[mask] = pct_hi * (
                    current_value[mask] / np.maximum(patch[0][mask], 0.01)
                )
            else:
                # Between percentiles
                mask = (current_value > patch[i - 1]) & (current_value <= patch[i])
                if mask.any():
                    frac = (current_value[mask] - patch[i - 1][mask]) / np.maximum(
                        patch[i][mask] - patch[i - 1][mask], 0.001
                    )
                    result[mask] = pct_lo + frac * (pct_hi - pct_lo)

        # Above p90
        mask = current_value > patch[-1]
        result[mask] = 90 + 10 * np.minimum(
            (current_value[mask] - patch[-1][mask]) / np.maximum(patch[-1][mask], 0.01),
            1.0,
        )

        return np.clip(result, 0, 100).astype(np.float32)

    def close(self):
        """Close cached datasets."""
        for ds in self._cache.values():
            ds = None
        self._cache.clear()


if __name__ == "__main__":

    # ======== TEST ========
    # output_dir = os.path.join(config.data_dir, "era5_climatology_test")
    # os.makedirs(output_dir, exist_ok=True)
    # # Test with 3 DOYs and 3 years (should take ~30 seconds)
    # build_global_climatology(
    #     years=[2022, 2023, 2024],
    #     output_dir=output_dir,
    #     window_days=7,  # Smaller window = fewer reads
    # )
    # print(f"Climatology saved to {output_dir}")
    # ======== PROD ========
    # Run with: conda activate firecomp && python -m firecomp.dsrc.weather --prod
    import sys

    if "--prod" in sys.argv:
        output_dir = os.path.join(config.data_dir, "era5_climatology")
        os.makedirs(output_dir, exist_ok=True)
        build_global_climatology(
            years=list(range(2012, 2025)),
            output_dir=output_dir,
        )
        print(f"Production climatology saved to {output_dir}")
