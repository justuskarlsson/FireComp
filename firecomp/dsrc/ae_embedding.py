from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from glob import glob
import json
import math
import os
import re
import shutil

from PIL import Image
import h5py
from tqdm import tqdm
from firecomp.dsrc.earth_engine import (
    DownloadOptions,
    collection_batches,
    download_image,
    ee_image_get_date_str,
    ee_to_list,
    init_ee,
    tif_to_png,
)
from firecomp.dsrc.vnp14 import DEG_CELL_SIZE
from firecomp.config import FLOAT32_NODATA, config
from pqdm.threads import pqdm
from pqdm.processes import pqdm as pqdm_process
from osgeo import gdal, gdalconst, osr
from firecomp.helpers import dump_json, load_json
from firecomp.dsrc.dsrc import (
    DSRC,
    TIF_CREATE_OPTIONS,
    deg_per_px_from_m_equator,
    pipeline,
    create_tif,
)
import numpy as np

gdal.UseExceptions()

# uint8 linear quantization constants
# Observed data range: [-0.567, 0.509], using symmetric [-0.6, 0.6] with headroom
AE_UINT8_MIN = -0.6
AE_UINT8_MAX = 0.6
AE_UINT8_NODATA = 255  # Reserve 255 for nodata, use 0-254 for values


def ae_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float embedding values to uint8 indices (linear quantization)."""
    result = np.full(arr.shape, AE_UINT8_NODATA, dtype=np.uint8)
    valid = np.isfinite(arr)
    if np.any(valid):
        scaled = (arr[valid] - AE_UINT8_MIN) / (AE_UINT8_MAX - AE_UINT8_MIN)
        result[valid] = np.clip(np.round(scaled * 254), 0, 254).astype(np.uint8)
    return result


def ae_uint8_to_float(arr: np.ndarray, dst_no_data: float = FLOAT32_NODATA) -> np.ndarray:
    """Convert uint8 indices back to float embedding values."""
    result = np.full(arr.shape, dst_no_data, dtype=np.float32)
    valid = arr != AE_UINT8_NODATA
    if np.any(valid):
        result[valid] = AE_UINT8_MIN + (arr[valid].astype(np.float32) / 254) * (
            AE_UINT8_MAX - AE_UINT8_MIN
        )
    return result


def create_ae_tif_uint8(path: str, bbox: list[float], num_bands: int = 64):
    """Create a uint8 GeoTIFF for AE embeddings at 300mpp."""
    px_deg = deg_per_px_from_m_equator(300.0)
    w = int((bbox[2] - bbox[0]) / px_deg)
    h = int((bbox[3] - bbox[1]) / px_deg)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, num_bands, gdal.GDT_Byte, TIF_CREATE_OPTIONS)
    ds.SetGeoTransform([bbox[0], px_deg, 0, bbox[3], 0, -px_deg])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    for i in range(1, num_bands + 1):
        ds.GetRasterBand(i).SetNoDataValue(AE_UINT8_NODATA)
    return ds


class AEEmbeddings(DSRC):
    name = "ae_embeddings"
    channel_names = [str(i) for i in range(64)]
    no_data = FLOAT32_NODATA  # For get_sample output (decoded to float)

    def __init__(
        self,
        *,
        year: int = 2017,
        band_list: list[int] = None,
        pca_bands: int | None = None,
    ):
        super().__init__()
        self.year = year
        self.pca_bands = pca_bands
        self.band_list = band_list

        # Directory for downloaded uint8 tiles (suffixed by year)
        self.ee_dir = os.path.join(config.data_dir, f"ae_embeddings_small_{year}")
        # Directory for zone-warped files and final outputs (suffixed by year)
        self.dst_dir = os.path.join(config.data_dir, f"ae_embeddings_by_zone_{year}")

        if pca_bands is not None:
            if band_list is not None:
                raise ValueError("band_list and pca_bands cannot be used together")
            self.tif_path = os.path.join(
                self.dst_dir, f"ae_embeddings_pca_{pca_bands}.tif"
            )
            self.channel_names = [str(i) for i in range(pca_bands)]
            self.name += f"_pca_{pca_bands}"

        else:
            self.tif_path = os.path.join(self.dst_dir, "ae_embeddings.tif")
            if band_list is not None:
                self.channel_names = [str(i) for i in band_list]
                self.name += f"_{''.join(str(i) for i in band_list)}"

        # Shared for PCA and non-PCA
        self.vrt_path = os.path.join(self.dst_dir, "ae_embeddings.vrt")
        self.num_geod_groups = 120
        self.geod_group_size = round(360.0 / self.num_geod_groups)
        self.pca_seed = 0  # RNG for subsampling

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        size: int = None,
        x_size: int = None,
        y_size: int = None,
    ):
        if size is not None:
            x_size = size
            y_size = size
        ds: gdal.Dataset = gdal.Open(self.tif_path)
        xoff, yoff, xsize, ysize = self._get_src_patch(bbox, ds)

        if self.pca_bands is not None:
            # PCA output is float32 - read directly
            patch = np.zeros(
                (len(self.channel_names), y_size, x_size), dtype=np.float32
            )
            ds.ReadAsArray(
                xoff,
                yoff,
                xsize,
                ysize,
                buf_obj=patch,
                resample_alg=gdalconst.GRIORA_NearestNeighbour,
            )
        else:
            # 64-band uint8 - decode to float32
            indices = np.zeros(
                (len(self.channel_names), y_size, x_size), dtype=np.uint8
            )
            ds.ReadAsArray(
                xoff,
                yoff,
                xsize,
                ysize,
                buf_obj=indices,
                resample_alg=gdalconst.GRIORA_NearestNeighbour,
                band_list=self.band_list,
            )
            patch = indices
        return patch

    @pipeline
    def download_test(self, n: int = 5):
        """Download a few images to validate the pipeline before full download."""
        sen2_embeddings(
            None,
            f"{self.year}-01-01",
            f"{self.year + 1}-01-01",
            scale=300,
            test_limit=n,
            dst_dir=self.ee_dir,
        )

    @pipeline
    def download_all(self):
        """Download all embeddings at 300 mpp with uint8 quantization."""
        sen2_embeddings(
            None,
            f"{self.year}-01-01",
            f"{self.year + 1}-01-01",
            scale=300,
            dst_dir=self.ee_dir,
        )

    @pipeline
    def find_groups(self):
        geod_groups = defaultdict(list)
        # 6 deg per
        paths = [os.path.join(self.ee_dir, file) for file in os.listdir(self.ee_dir)]
        paths.sort()
        print(f"Getting geod groups for {len(paths)} paths")
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        # for path in tqdm(paths, desc="Getting geod groups"):
        def get_cells(path):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            ds: gdal.Dataset = gdal.Open(path)
            vrt = gdal.Warp(
                "",
                ds,
                dstSRS="EPSG:4326",
                format="VRT",
                errorThreshold=0.0,
                warpOptions=["SKIP_NOSOURCE=YES"],
            )

            gt = vrt.GetGeoTransform()
            w, h = vrt.RasterXSize, vrt.RasterYSize
            xs = [gt[0], gt[0] + gt[1] * w + gt[2] * h]
            ys = [gt[3], gt[3] + gt[4] * w + gt[5] * h]
            tl_lon, br_lon = min(xs), max(xs)
            vrt = None

            cell_x = int((tl_lon + 180) / self.geod_group_size)
            cell_x2 = int((br_lon + 180) / self.geod_group_size)
            cells = []
            # Anti meridian?

            while cell_x <= cell_x2:
                cells.append(cell_x % self.num_geod_groups)
                cell_x += 1
            return cells, path

        pairs = pqdm(paths, get_cells, n_jobs=20, desc="Getting geod groups")
        for cells, path in pairs:
            for cell in cells:
                geod_groups[cell].append(path)
        geod_groups = dict(geod_groups)
        os.makedirs(self.dst_dir, exist_ok=True)

        dump_json(geod_groups, os.path.join(self.dst_dir, "geod_groups.json"))

    @pipeline
    def warp_groups(self):
        geod_groups = load_json(os.path.join(self.dst_dir, "geod_groups.json"))

        def single(cell_x, paths):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            paths = [gdal.Open(path) for path in paths]
            cell_x = int(cell_x)
            min_lon = round(-180.0 + cell_x * self.geod_group_size)
            max_lon = round(-180.0 + (cell_x + 1) * self.geod_group_size)
            bbox = [min_lon, -90.0, max_lon, 90.0]
            dst_path = os.path.join(self.dst_dir, f"cell_{cell_x}.tif")
            dst_ds = create_ae_tif_uint8(dst_path, bbox)
            gdal.Warp(
                dst_ds,
                paths,
                options=gdal.WarpOptions(
                    dstSRS="EPSG:4326",
                    multithread=True,
                    srcNodata=AE_UINT8_NODATA,
                    dstNodata=AE_UINT8_NODATA,
                    resampleAlg=gdalconst.GRA_NearestNeighbour,
                ),
            )
            dst_ds.FlushCache()
            dst_ds = None
            return dst_path

        paths = pqdm(
            geod_groups.items(),
            single,
            n_jobs=3,
            desc="Warping groups",
            argument_type="args",
            exception_behaviour="immediate",
        )
        gdal.BuildVRT(self.vrt_path, paths)
        print(self.vrt_path)

    @pipeline
    def translate_to_vrt_to_tif(self):
        """
        Create final global TIF from zone VRT.
        - If pca_bands is None: creates 64-band uint8 TIF
        - If pca_bands is set: creates PCA-reduced float32 TIF
        """
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        print("Creating final outputs...")

        if not os.path.exists(self.vrt_path):
            raise RuntimeError("VRT not built. Run warp_groups() first.")

        # (A) Full 64-band uint8 TIF
        if self.pca_bands is None:
            print(f"Creating 64-band uint8 TIF: {self.tif_path}")
            dst_ds = create_ae_tif_uint8(
                self.tif_path, [-180.0, -90.0, 180.0, 90.0], 64
            )
            gdal.Warp(
                dst_ds,
                self.vrt_path,
                options=gdal.WarpOptions(
                    multithread=True,
                    resampleAlg=gdalconst.GRA_NearestNeighbour,
                    srcNodata=AE_UINT8_NODATA,
                    dstNodata=AE_UINT8_NODATA,
                ),
            )
            dst_ds.FlushCache()
            dst_ds = None
            print(f"Done: {self.tif_path}")

        # (B) PCA-reduced TIF (float32 output - PCA values have different range)
        else:
            scratch_dir = "/scratch/local"
            if not os.path.exists(scratch_dir):
                raise RuntimeError("Please run on a cpu node with /scratch/local")

            ds: gdal.Dataset = gdal.Open(self.vrt_path)
            paths = []
            for path in tqdm(ds.GetFileList(), desc="Copying to local disk"):
                if path.endswith(".vrt"):
                    continue
                dst_path = os.path.join(scratch_dir, os.path.basename(path))
                shutil.copy(path, dst_path)
                paths.append(dst_path)

            # --- FIT PCA (read uint8, decode to float) ---
            n_total = 0
            mean = None
            M2 = None

            for path in tqdm(paths, desc="Fitting PCA"):
                ds: gdal.Dataset = gdal.Open(path)
                C, H, W = 64, ds.RasterYSize, ds.RasterXSize

                # Read uint8 and decode to float32
                X_uint8 = ds.ReadAsArray()  # (64, H, W) uint8
                X = ae_uint8_to_float(X_uint8)  # (64, H, W) float32
                X = X.reshape(C, -1)  # (64, N)

                valid = ~np.any(X == FLOAT32_NODATA, axis=0)
                if not np.any(valid):
                    continue
                X = X[:, valid]

                b_mean = X.mean(axis=1, dtype=np.float64)
                np.subtract(X, b_mean[:, None], out=X, casting="unsafe")
                b_M2 = (X @ X.T).astype(np.float64, copy=False)

                m = X.shape[1]
                if mean is None:
                    mean, M2, n_total = b_mean.copy(), b_M2.copy(), m
                else:
                    n_old = n_total
                    n_total += m
                    delta = b_mean - mean
                    mean += (m / n_total) * delta
                    M2 += b_M2 + np.outer(delta, delta) * (n_old * m / n_total)

            if n_total < 2:
                raise RuntimeError("No valid samples for PCA.")

            cov = M2 / (n_total - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            Wk = eigvecs[:, : self.pca_bands]  # (64, k)

            print(
                f"PCA explained variance: {eigvals[:self.pca_bands].sum() / eigvals.sum():.2%}"
            )

            # --- TRANSFORM ---
            pca_paths = []
            for path in tqdm(paths, desc="Transforming to PCA"):
                ds: gdal.Dataset = gdal.Open(path)
                dst_path = path.replace(".tif", "_pca.tif")
                drv = gdal.GetDriverByName("GTiff")
                dst_ds = drv.Create(
                    dst_path,
                    ds.RasterXSize,
                    ds.RasterYSize,
                    self.pca_bands,
                    gdal.GDT_Float32,
                    options=TIF_CREATE_OPTIONS,
                )
                dst_ds.SetGeoTransform(ds.GetGeoTransform())
                dst_ds.SetSpatialRef(ds.GetSpatialRef())
                for b in range(1, self.pca_bands + 1):
                    dst_ds.GetRasterBand(b).SetNoDataValue(FLOAT32_NODATA)

                # Read uint8, decode, apply PCA
                X_uint8 = ds.ReadAsArray()
                X = ae_uint8_to_float(X_uint8).astype(np.float64)
                C, H, W = X.shape
                X = X.reshape(C, -1)

                valid = ~np.any(X == FLOAT32_NODATA, axis=0)
                Y = np.full(
                    (self.pca_bands, X.shape[1]), FLOAT32_NODATA, dtype=np.float32
                )

                X = X[:, valid] - mean[:, None]
                X = Wk.T @ X
                Y[:, valid] = X.astype(np.float32)

                dst_ds.WriteArray(Y.reshape(self.pca_bands, H, W))
                dst_ds.FlushCache()
                dst_ds = None
                pca_paths.append(dst_path)

            # Merge PCA tiles
            vrt_path = os.path.join(scratch_dir, "pca.vrt")
            gdal.BuildVRT(vrt_path, pca_paths).FlushCache()

            tif_path = os.path.join(scratch_dir, "pca.tif")
            out = gdal.Warp(
                tif_path,
                vrt_path,
                options=gdal.WarpOptions(
                    format="GTiff",
                    multithread=True,
                    resampleAlg=gdalconst.GRA_NearestNeighbour,
                    outputType=gdal.GDT_Float32,
                    creationOptions=TIF_CREATE_OPTIONS,
                    srcNodata=FLOAT32_NODATA,
                    dstNodata=FLOAT32_NODATA,
                ),
            )
            if out is None:
                raise RuntimeError("Failed to warp PCA VRT to TIF")
            shutil.copy(tif_path, self.tif_path)
            print(f"Done: {self.tif_path}")

    def get_png(
        self,
        arr: np.ndarray,
        vmin: float = -0.3,
        vmax: float = 0.3,
        r: int = 0,
        g: int = 1,
        b: int = 2,
        **kwargs,
    ) -> Image.Image:
        arr = arr[[r, g, b]]  # 64 x H x W
        arr = arr.transpose(1, 2, 0)
        arr = np.clip(arr, vmin, vmax)
        arr = (arr - vmin) / (vmax - vmin)
        arr = (arr * 255 + 0.5).astype(np.uint8)
        return Image.fromarray(arr)


def sen2(bbox: list[float], start_date: str, end_date: str):
    import ee

    init_ee()
    DST_DIR = os.path.join(config.data_dir, "ee_sen2")
    os.makedirs(DST_DIR, exist_ok=True)

    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    rect = ee.Geometry.Rectangle(bbox)
    collection = collection.filterBounds(rect)
    collection = collection.filterDate(start_date, end_date)
    collection = collection.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 10))
    collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE", True)
    # collection = collection.select("B2", "B3", "B4", "B8A", "B11", "B12")
    collection = collection.limit(10)

    images = ee_to_list(collection)
    print(len(images), "images ready to download")
    args = []
    for i, image in enumerate(images):
        args.append((image, DST_DIR + f"/sen2_{i}.tif", rect, 10))
    paths = pqdm(args, download_image, n_jobs=10, argument_type="args")
    print(paths)
    return paths


def sen2_embeddings(
    bbox: list[float] | None,
    start_date: str,
    end_date: str,
    *,
    scale: int = 300,
    test_limit: int | None = None,
    dst_dir: str | None = None,
):
    """
    Download Alpha Earth embeddings from Earth Engine as uint8.

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat] or None for global
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        scale: Resolution in meters per pixel (default 300)
        test_limit: If set, only download this many images (for testing)
        dst_dir: Output directory (default: ae_embeddings_small in data_dir)
    """
    import ee

    init_ee()
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    if dst_dir is None:
        dst_dir = os.path.join(config.data_dir, "ae_embeddings_small")
    os.makedirs(dst_dir, exist_ok=True)

    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    if bbox:
        rect = ee.Geometry.Rectangle(bbox)
        collection = collection.filterBounds(rect)
    else:
        rect = None
    collection = collection.filterDate(start_date, end_date)

    if test_limit:
        collection = collection.limit(test_limit)

    tot_size = collection.size().getInfo()
    print(f"Total images to download: {tot_size}")

    if tot_size == 0:
        print("No images found for the given filters")
        return []

    batch_size = min(1000, tot_size)
    all_paths = []

    # Quantize to uint8 server-side (8x smaller download)
    options = DownloadOptions(
        crs=None,
        quantize_uint8=True,
        quantize_min=AE_UINT8_MIN,
        quantize_max=AE_UINT8_MAX,
    )

    for i, images in enumerate(collection_batches(collection, batch_size)):
        print(
            f"Downloading batch {i+1}, images {i*batch_size+1}-{i*batch_size+len(images)} of {tot_size}"
        )

        offset = i * batch_size

        def get_args(idx, image):
            return (
                image,
                os.path.join(dst_dir, f"ae_{offset+idx}.tif"),
                rect,
                scale,
                options,
            )

        print(f"Preparing {len(images)} images...")
        args = list(enumerate(images))
        args = pqdm(
            args, get_args, n_jobs=20, argument_type="args", desc="Getting args"
        )
        paths = pqdm(
            args,
            download_image,
            n_jobs=10,  # Reduced to avoid rate limiting
            argument_type="args",
            desc=f"Downloading batch {i+1}",
            exception_behaviour="defer",
        )
        all_paths.extend(paths)

        if test_limit and len(all_paths) >= test_limit:
            break

    print(f"Downloaded {len(all_paths)} images to {dst_dir}")
    return all_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()
    year = args.year

    # 64-band uint8 output
    ae = AEEmbeddings(year=year)
    ae.download_all()  # 1. Download tiles
    ae.find_groups()  # 2. Group by longitude
    ae.warp_groups()  # 3. Warp to zones
    ae.translate_to_vrt_to_tif()  # 4. Create global tif

    # # PCA output (float32)
    ae = AEEmbeddings(year=year, pca_bands=5)
    ae.translate_to_vrt_to_tif()  # Uses existing zones, outputs PCA
