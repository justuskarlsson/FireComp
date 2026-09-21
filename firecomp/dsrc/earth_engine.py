from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional
import os

import requests

from firecomp.config import config
from pqdm.threads import pqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

has_init = False


def init_ee():
    import ee

    global has_init
    if not has_init:
        ee.Authenticate()
        ee.Initialize(project="ee-karlssonjustus")
        has_init = True


def ee_to_list(collection, count=None, offset=None):
    import ee

    if count is None:
        count = collection.size().getInfo()
    image_list = collection.toList(count, offset)
    images = [ee.Image(image_list.get(i)) for i in range(count)]
    return images


def ee_image_get_date_str(img, include_hour: bool = False):
    import ee

    ee_fmt = "YYYY-MM-dd"
    if include_hour:
        ee_fmt = "YYYY-MM-dd-HH"

    date_str = ee.Date(img.get("system:time_start")).format(ee_fmt).getInfo()
    return date_str


def dbg_proj(img):
    p = img.projection()
    return {
        "crs": p.crs().getInfo(),
        "nominalScale_m": p.nominalScale().getInfo(),
        "transform": p.transform().getInfo(),
    }


def _safe_bounds(img, proj, scale: int):
    # Tight-ish box in your target projection, avoids crazy global geometries
    # The 1 meter error helps simplify topology without changing extents materially.
    return img.geometry().transform(proj, scale / 10).bounds(scale / 10, proj)


@dataclass
class DownloadOptions:
    format: Literal["GeoTIFF"] = "GeoTIFF"
    crs: str | None = "EPSG:4326"
    # Use crs_transform for precise grid alignment across tiles.
    # Format: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
    # When set, 'scale' parameter is ignored and all tiles align to the same pixel grid.
    crs_transform: list[float] | None = None
    quantize_uint8: bool = False  # Convert to uint8 server-side before download
    # Linear quantization params (matching ae_embedding constants)
    quantize_min: float = -0.6
    quantize_max: float = 0.6


def download_image(
    image,
    dst_path: str,
    rect,
    scale: int,
    options: DownloadOptions = DownloadOptions(),
):
    """
    Download an Earth Engine image to a local GeoTIFF file.

    Args:
        image: ee.Image to download
        dst_path: Local file path for output
        rect: ee.Geometry.Rectangle defining the region to download
        scale: Pixel scale in meters (ignored if options.crs_transform is set)
        options: DownloadOptions with CRS, format, and transform settings

    Note on crs_transform:
        When options.crs_transform is set, it overrides the scale parameter.
        This ensures all tiles align to the same global pixel grid, avoiding
        NoData gaps at tile boundaries that occur when using meter-based scale
        with geographic CRS (EPSG:4326) at different latitudes.
    """
    import ee

    image: ee.Image
    rect: ee.Geometry.Rectangle
    if os.path.exists(dst_path):
        return dst_path

    # Quantize to uint8 server-side if requested (saves 8x bandwidth)
    if options.quantize_uint8:
        vmin, vmax = options.quantize_min, options.quantize_max
        # Linear scale: (val - min) / (max - min) * 254, with 255 = nodata
        image = (
            image.subtract(vmin)
            .divide(vmax - vmin)
            .multiply(254)
            .round()
            .clamp(0, 254)
            .unmask(255)  # Masked/nodata pixels become 255
            .toUint8()
        )

    crs = options.crs
    proj = crs
    if crs is None:
        proj = image.projection()
        crs = proj.crs()
    if rect is not None:
        region = rect
    else:
        region = _safe_bounds(image, proj, scale)

    image = image.clip(region)
    params = {
        "name": dst_path,
        "crs": crs,
        "format": options.format,
        "bands": image.bandNames().getInfo(),
        "region": region,
    }

    # Use crs_transform for precise grid alignment, or fall back to scale
    if options.crs_transform is not None:
        params["crs_transform"] = options.crs_transform
    else:
        params["scale"] = scale

    url = image.getDownloadURL(params)
    r = requests.get(url, stream=True, verify=False)
    if r.status_code == 200:
        with open(dst_path, "wb") as f:
            f.write(r.content)
        return dst_path
    else:
        r.raise_for_status()
        return None


def collection_batches(coll, batch_size=1000):
    import ee

    coll: ee.ImageCollection
    total = coll.size().getInfo()
    for start in range(0, total, batch_size):
        yield ee_to_list(coll, batch_size, start)


from osgeo import gdal

gdal.UseExceptions()
from PIL import Image
import numpy as np


def tif_to_png(
    path: str,
    vmin,
    vmax,
    gamma: float = 1.0,
    band_list: list[int] = [1, 2, 3],
    arr=None,
):
    if arr is None:
        ds: gdal.Dataset = gdal.Open(path)
        arr = ds.ReadAsArray(band_list=band_list)
    # 3 x W x H
    # print("along dims:", arr.min(axis=(1,2)), arr.max(axis=(1,2)))
    if hasattr(vmin, "__iter__"):
        for i in range(len(vmin)):
            arr[i] = (arr[i] - vmin[i]) / (vmax[i] - vmin[i])
    else:
        arr = (arr - vmin) / (vmax - vmin)
    arr = arr.transpose(1, 2, 0)
    arr = np.clip(arr, 0, 1)
    arr = np.power(arr, 1 / gamma)
    arr = (arr * 255 + 0.5).astype(np.uint8)
    png_path = path.replace(".tif", ".png")
    Image.fromarray(arr).save(png_path)
    return png_path


if __name__ == "__main__":
    ...

    # bbox = [151.7867431640625, -31.580577850341797 , 152.66932678222656, -30.327434539794922]
    # center_x = (bbox[0] + bbox[2]) / 2
    # center_y = (bbox[1] + bbox[3]) / 2
    # delta = 0.025
    # bbox = [center_x - delta, center_y - delta, center_x + delta, center_y + delta]
    # sen2(bbox, "2019-10-17", "2019-12-11")
