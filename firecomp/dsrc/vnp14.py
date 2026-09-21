from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from glob import glob
import logging
import math
import os
import random
import shutil
import time
from typing import Annotated, Optional, Type, TypeVar, TypedDict
import typing
from matplotlib import pyplot as plt
import psutil
from tqdm import tqdm

from firecomp.config import config

# Suppress verbose logging from urllib3 and earthaccess
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("earthaccess").setLevel(logging.WARNING)

import numpy as np
import netCDF4
from firecomp.db import Store
from firecomp.helpers import dump_json, load_json, log_time
import h5py

from osgeo import gdal

gdal.UseExceptions()


class Const:
    days_per_batch = 100
    is_test = False
    only_fire = True
    start_date = date(2019, 10, 26)
    end_date = date(2019, 11, 4)


METER_PER_DEG = 111_320
DEG_CELL_SIZE = 375 / METER_PER_DEG  # 0.003369


def grid_geod_to_pixel(x: np.ndarray, y: np.ndarray):
    x = (x + 180.0) / DEG_CELL_SIZE
    y = (y + 90.0) / DEG_CELL_SIZE
    return x, y


def grid_pixel_to_geod(x: np.ndarray, y: np.ndarray):
    x = x * DEG_CELL_SIZE - 180.0
    y = y * DEG_CELL_SIZE - 90.0
    return x, y


@dataclass
class Fires(Store):
    @dataclass
    class Meta:
        start_date: np.datetime64
        end_date: np.datetime64

    # I
    @dataclass
    class Images:
        id: np.ndarray[np.int32]
        num_fire_pixels: np.ndarray[np.int32]
        file_name: list[str]
        start_time: np.ndarray[np.datetime64]

    # F
    @dataclass
    class Stats:
        id: np.ndarray[np.int32]
        start_date: np.ndarray[np.datetime64]
        end_date: np.ndarray[np.datetime64]
        num_fire: np.ndarray[np.int32]
        min_x: np.ndarray[np.float32]
        min_y: np.ndarray[np.float32]
        max_x: np.ndarray[np.float32]
        max_y: np.ndarray[np.float32]
        country: np.ndarray[np.int32]
        avg_xy_neighbors: np.ndarray[np.float32]
        ignition_ratio: np.ndarray[np.float32]
        t_ratio: np.ndarray[np.float32]

    # P
    @dataclass
    class Pixels:
        cls: np.ndarray[np.int8]
        x: np.ndarray[np.float32]
        y: np.ndarray[np.float32]
        width: np.ndarray[np.float32]
        height: np.ndarray[np.float32]
        date: np.ndarray[np.datetime64]
        image_id: np.ndarray[np.int32]
        fire_id: np.ndarray[np.int32]
        ignition: np.ndarray[np.bool]

    @dataclass
    class Projection:
        x: np.ndarray[np.int32]
        y: np.ndarray[np.int32]
        t: np.ndarray[np.int32]
        component: np.ndarray[np.int32]
        ignition: np.ndarray[np.bool]

    @dataclass
    class Polygon:
        fire_id: np.ndarray[np.int32]
        polygon: np.ndarray[np.float32]
        offset: np.ndarray[np.int64]

    meta: Meta
    images: Images
    stats: Stats
    # >= 10, 100, 1000, 10000
    stats_10: Stats
    stats_100: Stats
    stats_1000: Stats
    stats_10000: Stats
    pixels: Pixels
    projection: Projection
    projection_by_fire: dict[str, Projection]
    projection_by_t: dict[str, Projection]
    # polygon_x: dict[str, Polygon]

    @staticmethod
    def filter_fires(
        stats: Stats,
        very_narrow: float = 0.7,
        xy_threshold: float = 17.0,
        ignite_threshold: float = 1.0,
    ):
        too_narrow = stats.t_ratio > very_narrow
        too_sparse = ((stats.ignition_ratio * 100) > ignite_threshold) & (
            stats.avg_xy_neighbors < xy_threshold
        )
        invalid = too_narrow | too_sparse
        return ~invalid

    @staticmethod
    def get_polygons(polygon: Polygon, fire_ids_mask: np.ndarray[np.bool]):
        idxs = fire_ids_mask[polygon.fire_id]
        offsets = polygon.offset[idxs]
        sizes = polygon.polygon[idxs + 1] - offsets
        return polygon.polygon[offsets : offsets + sizes]


class VNP14:

    def __init__(self):
        import earthaccess

        earthaccess.login()

    @log_time
    def download_fire(self, d1, d2):
        import earthaccess

        n = 10
        wait_times = [10] * 4 + [300]
        for i in range(n):
            try:
                vnp14_items = earthaccess.search_data(
                    short_name="VNP14IMG",
                    temporal=(d1, d2),
                    count=-1,
                )
                earthaccess.download(vnp14_items, config.vnp14_dir)
                return
            except Exception as e:
                print(f"Error downloading {i + 1}/{n}: {e}")
                time.sleep(wait_times[i % len(wait_times)])

    def download(self, start_date: date, end_date: date):
        cur = start_date
        while cur <= end_date:
            d1 = cur.strftime("%Y-%m-%d")
            to = min(cur + timedelta(days=Const.days_per_batch - 1), end_date)
            d2 = to.strftime("%Y-%m-%d")
            print(f"Processing {d1} to {d2}")

            self.download_fire(d1, d2)
            cur = cur + timedelta(days=Const.days_per_batch)

    @log_time
    def process_data(self, res: list[str]):
        h5_file = h5py.File(os.path.join(config.vnp14_fires_dir, "vnp14_fires.h5"), "w")
        # TODO: Rewrite to singe batch insert
        images: list[Fires.Images] = []
        pixels: list[Fires.Pixels] = []
        aggr_fire = np.zeros((180 * 10, 360 * 10), dtype=np.int32)
        min_dt = np.datetime64("2030-01-01")
        max_dt = np.datetime64("1970-01-01")
        for tup in tqdm(res, desc="Processing fire batch"):
            try:
                ds_path = tup
                ds: VNP14_Attributes = netCDF4.Dataset(ds_path, "r")
                data: FirePixelRaw = ds.variables

                start_time = ds.StartTime
                min_dt = min(min_dt, np.datetime64(start_time))
                max_dt = max(max_dt, np.datetime64(start_time))
                vnp14_image = Fires.Images(
                    id=len(images),
                    file_name=ds_path,
                    start_time=np.datetime64(start_time),
                    num_fire_pixels=ds.FirePix,
                )
                images.append(vnp14_image)
                start_time = ds.StartTime

                n = len(data["FP_longitude"])
                width, height = viirs_i_latlon_width(
                    data["FP_latitude"][...], data["FP_ViewZenAng"][...]
                )

                cls = data["FP_confidence"][...]
                lon = data["FP_longitude"][...]
                lat = data["FP_latitude"][...]

                # Aggr fire
                aggr_x = np.round((lon + 180.0) * 10).astype(np.int32)
                aggr_y = np.round((lat + 90.0) * 10).astype(np.int32)
                # Ensure indices are within bounds
                aggr_x = np.clip(aggr_x, 0, aggr_fire.shape[1] - 1)
                aggr_y = np.clip(aggr_y, 0, aggr_fire.shape[0] - 1)
                aggr_fire[aggr_y, aggr_x] += 1

                # Create pixels
                batch = Fires.Pixels(
                    cls=cls,
                    x=lon,
                    y=lat,
                    width=width,
                    height=height,
                    date=np.repeat(np.datetime64(start_time), n),
                    image_id=np.full(n, len(images) - 1, dtype=np.int32),
                    fire_id=np.zeros(n, dtype=np.int32),
                )
                pixels.append(batch)
            finally:
                try:
                    ds.close()
                except:
                    pass
        pixels: Fires.Pixels = Store.merge(pixels)
        Store.cast_fields(pixels)
        print(f"Num fire pixels: {len(pixels.x)}")
        images: Fires.Images = Store.merge(images, op="stack")
        meta = Fires.Meta(
            start_date=min_dt,
            end_date=max_dt,
        )
        self.finish_processing(h5_file, pixels, images, meta)

    def finish_processing(self, h5_file: h5py.File, pixels, images, meta):

        stats, projections = self.fire_search(pixels)

        def select_min_fire(min_fire: int):
            mask = stats.num_fire >= min_fire
            kv = {k: v[mask] for k, v in asdict(stats).items()}
            return Fires.Stats(**kv)

        fires = Fires(
            meta=meta,
            images=images,
            stats=stats,
            stats_10=select_min_fire(10),
            stats_100=select_min_fire(100),
            stats_1000=select_min_fire(1000),
            stats_10000=select_min_fire(10000),
            pixels=pixels,
            **projections,
        )
        print(f"Saving to {h5_file.filename}...")
        fires.save(h5_file)
        h5_file.close()

    def fire_search(self, pixels: Fires.Pixels):
        import torch
        from firecomp import cpp

        x = torch.tensor(pixels.x + 180.0) / DEG_CELL_SIZE
        y = torch.tensor(pixels.y + 90.0) / DEG_CELL_SIZE
        width = torch.tensor(pixels.width) / DEG_CELL_SIZE
        height = torch.tensor(pixels.height) / DEG_CELL_SIZE
        fcls = torch.tensor(pixels.cls).to(torch.int8)
        min_date = pixels.date.min()
        t = (pixels.date - min_date) // np.timedelta64(1, "D")
        t = torch.tensor(t, dtype=torch.int32)
        fcls[fcls < 7] = 1
        fcls[fcls >= 7] = 2
        print(f"{len(x)} nodes")
        start_time = time.time()
        components, pixel_to_fire, pixel_ignition, xytci = cpp.search(
            x, y, width, height, fcls, t
        )
        print(f"{len(components)} components, took {time.time() - start_time} seconds")

        def create_projection(xytci):
            return Fires.Projection(
                x=xytci[:, 0],
                y=xytci[:, 1],
                t=xytci[:, 2],
                component=xytci[:, 3],
                ignition=xytci[:, 4].astype(np.bool),
            )

        projection = create_projection(xytci.numpy())

        # ======== Create Fire from dict ========
        stats = Fires.Stats(
            id=np.arange(len(components), dtype=np.int32),
            start_date=[
                min_date + np.timedelta64(components[i]["t_min"], "D")
                for i in range(len(components))
            ],
            end_date=[
                min_date + np.timedelta64(components[i]["t_max"], "D")
                for i in range(len(components))
            ],
            num_fire=[components[i]["num_fire"] for i in range(len(components))],
            min_x=[
                components[i]["x_min"] * DEG_CELL_SIZE - 180.0
                for i in range(len(components))
            ],
            min_y=[
                components[i]["y_min"] * DEG_CELL_SIZE - 90.0
                for i in range(len(components))
            ],
            max_x=[
                components[i]["x_max"] * DEG_CELL_SIZE - 180.0
                for i in range(len(components))
            ],
            max_y=[
                components[i]["y_max"] * DEG_CELL_SIZE - 90.0
                for i in range(len(components))
            ],
            country=[1],
            avg_xy_neighbors=[
                components[i]["avg_xy_neighbors"] for i in range(len(components))
            ],
            ignition_ratio=[
                components[i]["ignition_ratio"] for i in range(len(components))
            ],
            t_ratio=[components[i]["t_ratio"] for i in range(len(components))],
        )
        stats = Store.cast(stats)
        projection_by_fire = dict()
        print(stats.num_fire.dtype)
        # >10 makes this super slow, soo many keys to write to disk
        # >2M 'files' to write to h5.
        xytci_by_fire, sizes = cpp.group_by_fire(
            xytci, torch.tensor(stats.num_fire, dtype=torch.int32), 100
        )
        print("xytci_by_fire", xytci_by_fire.shape)
        off = 0
        xytci_by_fire = xytci_by_fire.numpy()
        for i in range(len(sizes)):
            data = xytci_by_fire[off : off + sizes[i], :]
            c = data[0, 3].item()
            projection_by_fire[str(c)] = create_projection(data)
            off += sizes[i]
        projection_by_t = dict()
        xytci_by_t, sizes = cpp.group_by_time(xytci)
        print("xytci_by_t", xytci_by_t.shape)
        off = 0
        xytci_by_t = xytci_by_t.numpy()
        for i in range(len(sizes)):
            data = xytci_by_t[off : off + sizes[i], :]
            if len(data) == 0:
                continue
            t = data[0, 2].item()
            projection_by_t[str(t)] = create_projection(data)
            off += sizes[i]
        # Log RAM usage
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"RAM usage: {ram_usage:.1f} MB")
        # ======== Country ========
        countries_ds: gdal.Dataset = gdal.Open(
            os.path.join(config.data_dir, "countries", "countries.tif")
        )
        countries_raster = countries_ds.ReadAsArray()
        x_mid = (stats.min_x + stats.max_x) / 2
        y_mid = (stats.min_y + stats.max_y) / 2
        x_mid = np.round((x_mid + 180.0) * 10).astype(np.int32)
        x_mid = np.clip(x_mid, 0, countries_raster.shape[1] - 1)
        y_mid = np.round((y_mid + 90.0) * 10).astype(np.int32)
        y_mid = np.clip(y_mid, 0, countries_raster.shape[0] - 1)
        y_mid = countries_raster.shape[0] - y_mid - 1
        country = countries_raster[y_mid, x_mid]

        # ======== Pad coast lines ========
        countries_raster[countries_raster == 0] = countries_raster.min() - 1
        countries_raster_torch = (
            torch.from_numpy(countries_raster).unsqueeze(0).unsqueeze(0).float()
        )
        countries_raster_torch = torch.nn.functional.max_pool2d(
            countries_raster_torch, kernel_size=3, stride=1, padding=1
        )
        countries_raster = (
            countries_raster_torch.squeeze()
            .squeeze()
            .numpy()
            .astype(countries_raster.dtype)
        )
        was_zero = country == 0
        country[was_zero] = countries_raster[y_mid[was_zero], x_mid[was_zero]]
        stats.country = country.astype(np.int32)

        pixels.fire_id = pixel_to_fire.numpy()
        pixels.ignition = pixel_ignition.numpy()
        return stats, {
            "projection": projection,
            "projection_by_fire": projection_by_fire,
            "projection_by_t": projection_by_t,
        }


def viirs_i_latlon_width(lat_deg: np.ndarray, view_zen_deg: np.ndarray):
    """
    Per‑pixel footprint size of VIIRS I‑band pixels *on the ellipsoid*.

    Parameters
    ----------
    lat_deg        : array‑like     Geodetic latitude of pixel centre [deg].
    view_zen_deg   : array‑like     VIIRS FP_ViewZenAng (local zenith) [deg].

    Returns
    -------
    dx_deg, dy_deg
    """
    # Constants – tweak only if you have a reason
    _H_ORBIT = 8.29e5  # S‑NPP/JPSS orbital height [m]
    _R_EARTH = 6_371_008.8  # IERS 2003 authalic radius [m]

    # Nadir ground‑projection of a *single* detector IFOV (I‑bands)
    _D_SCAN = 129.3  # cross‑track, east‑west, at θ = 0°  [m]
    _D_TRACK = 371.0  # along‑track, north‑south, at θ = 0° [m]

    # Aggregation regime break‑points (°)
    _BRK1, _BRK2 = 31.72, 44.86
    lat = np.asarray(lat_deg, dtype=float)
    theta = np.deg2rad(view_zen_deg)

    # ------------------------
    # 1. Aggregation selector
    # ------------------------
    # |θ| ≤ 31.72° → 3× aggregation
    # 31.72° < |θ| ≤ 44.86° → 2×
    # otherwise 1× (native)
    a = np.select(
        [np.abs(view_zen_deg) <= _BRK1, np.abs(view_zen_deg) <= _BRK2],
        [3, 2],
        default=1,
    ).astype(float)

    # ------------------------------------
    # 2. North‑south (along‑track) size
    # ------------------------------------
    # δy ≈ D_track / cos θ
    dy_m = _D_TRACK / np.cos(theta)

    # ------------------------------------
    # 3. East‑west (cross‑track) size
    # ------------------------------------
    # δx ≈ (a · D_scan) / cos² θ
    dx_m = (a * _D_SCAN) / (np.cos(theta) ** 2)

    # ------------------------------------
    # 4. Convert to angular deltas
    # ------------------------------------
    metres_per_deg_lat = (2 * np.pi * _R_EARTH) / 360.0
    metres_per_deg_lon = metres_per_deg_lat * np.cos(np.deg2rad(lat))

    dx_deg = dx_m / metres_per_deg_lon
    dy_deg = dy_m / metres_per_deg_lat

    return dx_deg, dy_deg


class FirePixelRaw(TypedDict):
    # Single axis variables (phony_dim_0)
    FP_AdjCloud: np.ndarray[np.uint16]  # (n,)
    FP_AdjWater: np.ndarray[np.uint16]  # (n,)
    FP_MAD_DT: np.ndarray[np.float32]  # (n,)
    FP_MAD_T4: np.ndarray[np.float32]  # (n,)
    FP_MAD_T5: np.ndarray[np.float32]  # (n,)
    FP_MeanDT: np.ndarray[np.float32]  # (n,)
    FP_MeanRad13: np.ndarray[np.float32]  # (n,)
    FP_MeanT4: np.ndarray[np.float32]  # (n,)
    FP_MeanT5: np.ndarray[np.float32]  # (n,)
    FP_Rad13: np.ndarray[np.float32]  # (n,)
    FP_SolAzAng: np.ndarray[np.float32]  # (n,)
    FP_SolZenAng: np.ndarray[np.float32]  # (n,)
    FP_T4: np.ndarray[np.float32]  # (n,)
    FP_T5: np.ndarray[np.float32]  # (n,)
    FP_ViewAzAng: np.ndarray[np.float32]  # (n,)
    FP_ViewZenAng: np.ndarray[np.float32]  # (n,)
    FP_WinSize: np.ndarray[np.uint16]  # (n,)
    FP_confidence: np.ndarray[np.uint8]  # (n,)
    FP_day: np.ndarray[np.uint8]  # (n,)
    FP_latitude: np.ndarray[np.float32]  # (n,)
    FP_line: np.ndarray[np.uint16]  # (n,)
    FP_longitude: np.ndarray[np.float32]  # (n,)
    FP_power: np.ndarray[np.float32]  # (n,)
    FP_sample: np.ndarray[np.uint16]  # (n,)

    algorithm_qa: np.ndarray[np.uint32]  # (6464, 6400)
    fire_mask: np.ndarray[np.uint8]  # ACTUALLY "fire mask" !!! (6464, 6400)


class VNP14_Attributes:
    # Important:
    VNP02IMG: str  # Example: "VNP02CCIMG.A2018166.0900.002.2022280050422.nc"
    VNP03IMG: str  # Example: "VNP03IMG.A2018166.0900.002.2021082193312.nc"
    VNP02MOD: str  # Example: "VNP02CCMOD.A2018166.0900.002.2022280050422.nc"
    VNP02GDC: str  # Example: "VNP02GDC.A2018166.0900.002.2021083044027.nc"
    ProcessVersionNumber: str  # Example: "3.1.9"
    ExecutableCreationDate: str  # Example: "Oct 31 2023"
    ExecutableCreationTime: str  # Example: "17:46:00"
    SystemID: str  # Example: "Linux minion20179 5.4.0-1082-fips #91-Ubuntu SMP Wed Jul 19 21:56:44 UTC 2023 x86_64"
    Unagg_GRingLatitude: str  # Example: "31.236309,36.270294,57.028767,49.998020"
    Unagg_GRingLongitude: (
        str  # Example: "-93.251976,-125.995964,-125.774040,-79.778999"
    )
    NorthBoundingCoordinate: np.float32  # Example: 57.16582107543945
    SouthBoundingCoordinate: np.float32  # Example: 31.236309051513672
    EastBoundingCoordinate: np.float32  # Example: -79.77899932861328
    WestBoundingCoordinate: np.float32  # Example: -125.99596405029297
    DayNightFlag: str  # Example: "Night"
    FirePix: np.int32  # Example: 443
    DayPix: np.int32  # Example: 0
    LandPix: np.int32  # Example: 0
    PGE_Name: str  # Example: "PGE510"
    NightPix: np.int32  # Example: 41369600
    ShortName: str  # Example: "VNP14IMG"
    WaterPix: np.int32  # Example: 0
    MissingPix: np.int32  # Example: 195788
    GlintPix: np.int32  # Example: 0
    CloudPix: np.int32  # Example: 10122915
    GRingPointSequenceNo: np.ndarray[np.int32]  # Example: [1 2 3 4]
    project: str  # Example: "VIIRS Land SIPS Active Fire"
    GRingPointLongitude: np.ndarray[
        np.float32
    ]  # Example: [-93.252 -125.996 -125.774 -79.779]
    EndTime: str  # Example: "2018-06-15 09:06:00.000"
    RangeEndingDate: str  # Example: "2018-06-15"
    InputPointer: str  # Example: "/MODAPSops4/archive/f20179/running/VNP_L1bglu/59482346/VNP02CCIMG.A2018166.0900.002.2022280050422.nc, ..."
    identifier_product_doi: str  # Example: "10.5067/VIIRS/VNP14IMG.002"
    Conventions: str  # Example: "CF-1.6"
    license: str  # Example: "http://science.nasa.gov/earth-science/earth-science-data/data-information-policy/"
    processing_level: str  # Example: "Level 2"
    publisher_name: str  # Example: "LAADS"
    LocalGranuleID: str  # Example: "VNP14IMG.A2018166.0900.002.2024080145033.nc"
    stdname_vocabulary: (
        str  # Example: "NetCDF Climate and Forecast (CF) Metadata Convention"
    )
    SensorShortname: str  # Example: "VIIRS"
    StartTime: str  # Example: "2018-06-15 09:00:00.000"
    publisher_email: str  # Example: "modis-ops@lists.nasa.gov"
    RangeBeginningDate: str  # Example: "2018-06-15"
    VersionID: str  # Example: "002"
    PGENumber: str  # Example: "510"
    Satellite: str  # Example: "NPP"
    creator_email: str  # Example: "modis-ops@lists.nasa.gov"
    PGE_EndTime: str  # Example: "2018-06-15 09:06:00.000"
    PGE_StartTime: str  # Example: "2018-06-15 09:00:00.000"
    ProductionTime: str  # Example: "2024-03-20 14:50:33.000"
    cdm_data_type: str  # Example: "swath"
    title: str  # Example: "VIIRS 375m Active Fire Data"
    AlgorithmType: str  # Example: "OPS"
    keywords_vocabulary: (
        str  # Example: "NASA Global Change Master Directory (GCMD) Science Keywords"
    )
    PlatformShortName: str  # Example: "SUOMI-NPP"
    GRingPointLatitude: np.ndarray[
        np.float32
    ]  # Example: [31.2363 36.2703 57.0288 49.998]
    RangeEndingTime: str  # Example: "09:06:00.00000"
    AlgorithmVersion: str  # Example: "NPP_PR14IMG 3.1.6"
    creator_url: str  # Example: "https://ladsweb.modaps.eosdis.nasa.gov"
    naming_authority: str  # Example: "gov.nasa.gsfc.VIIRSland"
    creator_name: str  # Example: "VIIRS Land SIPS Processing Group"
    publisher_url: str  # Example: "https://ladsweb.modaps.eosdis.nasa.gov"
    RangeBeginningTime: str  # Example: "09:00:00.00000"
    ProcessingEnvironment: str  # Example: "Linux minion20179 5.4.0-1082-fips #91-Ubuntu SMP Wed Jul 19 21:56:44 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux"
    LongName: str  # Example: "VIIRS/NPP Active Fires 6-Min L2 Swath 375m"
    ProcessingCenter: str  # Example: "MODAPS-NASA"
    SatelliteInstrument: str  # Example: "NPP_OPS"
    identifier_product_doi_authority: str  # Example: "https://doi.org"
    institution: str  # Example: "NASA Goddard Space Flight Center"
    PGEVersion: str  # Example: "2.0.13"
