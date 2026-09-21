# Data Sources and Storage

## Overview

FireComp uses satellite imagery and weather data to predict wildfire behavior. All external data sources inherit from the `DSRC` base class (`firecomp/dsrc/dsrc.py`), which provides a common `get_sample(bbox, t, size)` interface and HDF5 storage utilities (Zstd compression at clevel 3 with shuffle).

Data sources are registered as lazy properties in `firecomp/dsrc/all_dsrc.py` via the `_LazyDSRC` singleton. This avoids opening files that may not exist on every machine. Access them through `from firecomp.dsrc.all_dsrc import dsrc` and then `dsrc.era5`, `dsrc.ae_embeddings`, etc.

---

## VIIRS Products

VIIRS (Visible Infrared Imaging Radiometer Suite) provides the fire detection ground truth and optional raw terrain data. Two products are used:

### VNP14IMG -- Active Fire Detection

- **Module**: `firecomp/dsrc/vnp14.py` -- class `VNP14`
- **Description**: Active fire detection at 375m I-band resolution. Each granule contains per-pixel fire confidence, lat/lon, view geometry, and a full-swath `fire_mask` (6464 x 6400 uint8).
- **Use**: Ground truth labels. The `fire_mask` variable also provides cloud, water, and quality masks used to build loss masks.
- **Key data structures**:
  - `Fires` (dataclass/Store): Top-level container with nested `Meta`, `Images`, `Stats`, `Pixels`, `Projection`, `Polygon` sub-dataclasses.
  - `Fires.Stats`: Per-fire statistics including bounding box, temporal extent, `avg_xy_neighbors`, `ignition_ratio`, and `t_ratio` (used for wildfire vs tame classification).
  - `Fires.Projection`: Rasterized (x, y, t, component, ignition) arrays, indexed by fire ID or time step.
- **Fire search**: The `fire_search()` method uses a C++ BFS extension (`firecomp/cpp`) to cluster individual fire pixels into fire events based on spatio-temporal proximity. The resulting components define fire IDs used throughout the pipeline.
- **Derived DSRCs** (in `firecomp/dsrc/vnp14_dsrc.py`):
  - `FiresDaily` -- daily fire masks for a given (x, y, size)
  - `FiresDailyNextDay` -- next-day fire masks (shifts by +1 day)
  - `FiresAccum` -- accumulated fire time (progression history)

### VNP03IMG -- Geolocation

- **Module**: `firecomp/dsrc/vnp03img.py` -- class `VNP03IMG`
- **Description**: Geolocation data for VIIRS 375m I-band pixels. Provides lat/lon arrays (6464 x 6400) needed to map fire detections and sensor data to geographic coordinates.
- **Use**: Coordinate lookup during preprocessing (projecting VNP14 fire pixels onto the patch grid).
- **Fire mask values** (`FireMaskValue` enum): NOT_PROCESSED_NO_DATA (0), NOT_PROCESSED_BOWTIE (1), UNUSED (2), WATER (3), CLOUD (4), LAND (5), UNCLASSIFIED (6), FIRE_LOW (7), FIRE_NOMINAL (8), FIRE_HIGH (9).
- **Loss mask**: `LOSS_MASK_VALUES = {0, 1, 2, 4, 6}` -- pixels excluded from loss computation (no data, bowtie, unused, cloud, unclassified).

---

## Alpha Earth Embeddings

- **Module**: `firecomp/dsrc/ae_embedding.py` -- class `AEEmbeddings`
- **Source**: Google Earth Engine collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (derived from Sentinel-2 imagery)
- **Native resolution**: 10m, downloaded at 300m via EE pyramiding for VIIRS alignment
- **Channels**: 64 embedding dimensions. Two variants:
  - **Raw bands** (e.g., bands 1-5): `AEEmbeddings(band_list=[1, 2, 3, 4, 5])` -- stored as uint8 via linear quantization (range [-0.6, 0.6] mapped to [0, 254], 255 = nodata). Conversion functions: `ae_float_to_uint8()`, `ae_uint8_to_float()`.
  - **PCA-reduced** (e.g., 5 components): `AEEmbeddings(pca_bands=5)` -- stored as float32. PCA computed on-the-fly from uint8 tiles with online covariance accumulation.
- **Temporal**: Static snapshots. Multiple years available (2017-2021) via `year` parameter. Default is 2017.
- **Pipeline**: `download_all()` -> `find_groups()` (longitude-based tiling) -> `warp_groups()` (reproject to EPSG:4326) -> `translate_to_vrt_to_tif()` (build global GeoTIFF).
- **Registry entries** in `all_dsrc.py`: `ae_embeddings`, `ae_embeddings_64`, `ae_embeddings_12345`, `ae_embeddings_pca_5`, `ae_y17_64` through `ae_y21_64`, `ae_y20_pca_5`, `ae_y21_pca_5`, `ae_y21_12345`.

---

## ERA5 Weather

- **Module**: `firecomp/dsrc/weather.py` -- classes `ERA5`, `ERA5Fire`, `ERA5FirePercentile`
- **Source**: ECMWF ERA5-Land Daily Aggregated (`ECMWF/ERA5_LAND/DAILY_AGGR`) via Google Earth Engine
- **Native resolution**: 0.1 degrees (~11km)
- **Inference resolution**: upsampled to 256x256 at 300m via bilinear interpolation

### ERA5 (raw)

Class `ERA5` provides 23 raw bands covering:
- Moisture stress: `volumetric_soil_water_layer_1_min`, `_max`, `volumetric_soil_water_layer_2_min`, evaporation/transpiration sums, `skin_reservoir_content_min`, `_max`
- Atmospheric dryness: `dewpoint_temperature_2m_min`, `_max`, `temperature_2m_min`, `_max`, surface heat fluxes
- Heating & radiation: `skin_temperature_max`, `soil_temperature_level_1_max`, `surface_net_solar_radiation_sum`
- Hydrology: `runoff_sum`, `surface_runoff_sum`, `snow_depth_min`, `snow_cover_min`
- Wind: `u_component_of_wind_10m_max`, `v_component_of_wind_10m_max`

### ERA5Fire (derived)

Class `ERA5Fire` computes 6 fire-relevant derived channels:
1. `vpd` -- vapor pressure deficit (kPa), computed from temperature and dewpoint via Tetens formula (`compute_vpd()`)
2. `soil_moisture` -- average of layer 1 min/max
3. `soil_moisture_ratio` -- current / 1 year ago (1.0 = normal), via `compute_ratio_anomaly()`
4. `wind_magnitude` -- sqrt(u^2 + v^2)
5. `wind_sin` -- sin(wind direction) for circular encoding
6. `wind_cos` -- cos(wind direction) for circular encoding

### ERA5FirePercentile (percentile-based)

Class `ERA5FirePercentile` replaces the ratio-based soil moisture anomaly with a percentile rank (0-100, where 0 = driest on record). Requires pre-computed climatology built by `build_global_climatology()`:
- Builds 366 GeoTIFFs (one per day-of-year) with percentile thresholds [10, 25, 50, 75, 90]
- Uses LRU cache (`_ERA5Cache`) for efficient IO during construction
- Runtime query: single windowed read per fire sample, O(1) file opens via `SoilMoistureClimatology`

---

## GFS Weather Forecasts

- **Module**: `firecomp/dsrc/gfs.py` -- class `GFSForecast`
- **Source**: NOAA GFS (Global Forecast System) 0.25-degree (`NOAA/GFS0P25`) via Google Earth Engine
- **Native resolution**: 0.25 degrees (~28km)
- **Forecast horizon**: Up to 7 days ahead, aggregated from hourly forecasts to daily values server-side in GEE (`gfs_aggregate_daily()`)
- **Bands per day** (5 channels): `temp_max` (daily max temperature, C), `rh_min` (min relative humidity, %), `u_wind_max` (max eastward wind, m/s), `v_wind_max` (max northward wind, m/s), `precip_sum` (daily precipitation, mm)
- **Storage**: int16 encoding with linear scaling for compact downloads (~16 MB per day globally). Decoding via `_decode_int16()`.
- **Interface**: `get_sample()` returns shape `(forecast_days * 5, y_size, x_size)` flattened, or `get_forecast_sequence()` returns `(forecast_days, 5, y_size, x_size)`.
- **Registry entries**: `gfs_forecast` (7-day), `gfs_1day` (1-day, used by the next-day task), `gfs_7day` (7-day).

---

## WeatherNext 2 Forecasts

- **Module**: `firecomp/dsrc/weathernext2.py` -- class `WeatherNext2Forecast`
- **Source**: Google WeatherNext 2 model (`projects/gcp-public-data-weathernext/assets/weathernext_2_0_0`) via Earth Engine
- **Native resolution**: ~0.25 degrees
- **Channels per day**: `precip_sum`, `wind_speed_max`, optionally `mslp_min` (sea-level pressure)
- **Forecast horizon**: configurable, default 7 days (14-21 total channels)

---

## Canopy Height

- **Module**: `firecomp/dsrc/canopy_height.py` -- classes `CanopyHeightMeta`, `CanopyHeightDate`
- **Source**: Meta/WRI High Resolution Canopy Height Maps (`projects/sat-io/open-datasets/facebook/meta-canopy-height`) via Earth Engine
- **Native resolution**: 1m, aggregated to 300m at download time
- **Channels**: 1 (height in meters, uint8)
- **Temporal filtering**: `CanopyHeightDate` provides observation dates (years since 2000) so the pipeline can exclude post-fire canopy measurements via `date_predates_sample()`.

---

## Region-Based Sampling

The world is divided into 9 regions defined in `data/WildfireRegions.kml`:

| Region | Coverage |
|--------|----------|
| Europe | Mediterranean to Scandinavia |
| Desert | Sahara and Middle East |
| Africa | Sub-Saharan Africa |
| North Asia | Siberia, Mongolia, Northern China |
| South Asia | India, Southeast Asia, Southern China |
| Oceania | Australia, Indonesia, Pacific Islands |
| North NA | Canada, Alaska |
| Central NA | United States, Mexico, Caribbean |
| South America | All of South America |

**Purpose**:
- Balance the dataset across regions (avoid bias toward high-fire-count regions like Canada/Siberia)
- Enable cross-region transfer studies (train on single region vs global)

Strategic region-based subset selection reduces data volume from ~1 million to ~25,000 image files (~2%), making the project computationally feasible.

---
