from datetime import datetime
import math
import time
from typing import Callable, Literal, Optional

import hdf5plugin
import numpy as np


import h5py
from PIL import Image
from firecomp.helpers import pipeline, Pipeline
from osgeo import gdal, osr

gdal.UseExceptions()

RenderTypes = Literal["component", "not_-1", "bool"]


class DSRC(Pipeline):
    name: str = ""
    channel_names: list[str] = []
    no_data: Optional[float] = None

    def __init__(
        self,
        name: Optional[str] = None,
        channel_names: Optional[list[str]] = None,
        no_data: Optional[float] = None,
        render_type: Optional[RenderTypes] = None,
    ):
        super().__init__()
        self.name = name if name is not None else self.name
        self.channel_names = (
            channel_names if channel_names is not None else self.channel_names
        )
        if not self.channel_names:
            self.channel_names = ["0"]

        self.no_data = no_data if no_data is not None else self.no_data
        self.render_type: Optional[RenderTypes] = render_type

    def _get_src_patch(self, bbox: list[float], ds: gdal.Dataset):
        tl_x, dx, _, tl_y, _, dy = ds.GetGeoTransform()
        min_lon, min_lat, max_lon, max_lat = bbox

        xoff = math.floor((min_lon - tl_x) / dx)
        xsize = math.ceil((max_lon - tl_x) / dx) - xoff + 1
        yoff = math.floor((max_lat - tl_y) / dy)
        ysize = math.ceil((min_lat - tl_y) / dy) - yoff + 1

        # Clip to raster bounds
        raster_w = ds.RasterXSize
        raster_h = ds.RasterYSize
        if xoff < 0:
            xsize += xoff
            xoff = 0
        if yoff < 0:
            ysize += yoff
            yoff = 0
        if xoff + xsize > raster_w:
            xsize = raster_w - xoff
        if yoff + ysize > raster_h:
            ysize = raster_h - yoff
        xsize = max(0, xsize)
        ysize = max(0, ysize)

        return xoff, yoff, xsize, ysize

    def _get_actual_bbox(self, xoff: int, yoff: int, xsize: int, ysize: int, 
                         ds: gdal.Dataset) -> list[float]:
        """Compute the actual geographic bbox for the source patch pixels.
        
        Returns [min_lon, min_lat, max_lon, max_lat] for the pixel region.
        """
        tl_x, dx, _, tl_y, _, dy = ds.GetGeoTransform()
        # dx > 0 (left to right), dy < 0 (top to bottom)
        min_lon = tl_x + xoff * dx
        max_lon = tl_x + (xoff + xsize) * dx
        max_lat = tl_y + yoff * dy  # Top of region
        min_lat = tl_y + (yoff + ysize) * dy  # Bottom of region
        return [min_lon, min_lat, max_lon, max_lat]

    def get_sample(self, bbox: list[float], t: datetime, size: int):
        pass

    def get_sample_xy(self, x: int, y: int, t: datetime, size: int):
        pass

    def get_png(self, arr: np.ndarray, **kwargs) -> Image.Image:
        if self.render_type == "component":
            return Image.fromarray(arr)
        elif self.render_type == "not_-1":
            arr[arr == -1] = 0
            arr[arr != 0] = 255
            # Y X -> Y X 3
            img = np.array([arr, arr, arr, arr]).astype(np.uint8).transpose(1, 2, 0)
            return Image.fromarray(img)
        elif self.render_type == "bool":
            arr[arr > 0] = 255
            img = np.array([arr, arr, arr, arr]).astype(np.uint8).transpose(1, 2, 0)

            return Image.fromarray(img)
        else:
            raise NotImplementedError

    @property
    def channel_paths(self) -> list[str]:
        return [self.name + "." + channel for channel in self.channel_names]

    @staticmethod
    def get_storage_opts(data: np.ndarray):
        return dict(chunks=data.shape, **hdf5plugin.Zstd(clevel=3), shuffle=True)


class SampleArgTypes:
    def __init__(self, sample):
        self.sample = sample

    def resolve(dsrc: DSRC):
        if dsrc.arg_type == "bbox":
            ...
        elif dsrc.arg_type == "xy":
            ...


_METERS_PER_DEG_EQ = 111_319.49079327357


def deg_per_px_from_m_equator(m: float) -> float:
    return m / _METERS_PER_DEG_EQ


TIF_CREATE_OPTIONS = [
    "COMPRESS=ZSTD",
    "ZSTD_LEVEL=9",
    "TILED=YES",
    "BLOCKXSIZE=128",
    "BLOCKYSIZE=128",
    "INTERLEAVE=PIXEL",
    "BIGTIFF=YES",
]


def create_tif(
    path: str, bbox: list[float], px_deg: float, no_data: float, num_bands: int
):
    width = round((bbox[2] - bbox[0]) / px_deg)
    height = round((bbox[3] - bbox[1]) / px_deg)
    px_deg_x = (bbox[2] - bbox[0]) / width
    px_deg_y = (
        bbox[3] - bbox[1]
    ) / height  # positive magnitude; y pixel size will be negative in GT
    drv: gdal.Driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = drv.Create(
        path, width, height, num_bands, gdal.GDT_Float32, options=TIF_CREATE_OPTIONS
    )
    ds.SetGeoTransform((bbox[0], px_deg_x, 0.0, bbox[3], 0.0, -px_deg_y))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    # (GDAL 3+) ensure GIS-friendly axis order (lon,lat)
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetSpatialRef(srs)

    for b in range(1, num_bands + 1):
        band: gdal.Band = ds.GetRasterBand(b)
        # band.Fill(FLOAT32_NODATA)
        band.SetNoDataValue(no_data)

    return ds
