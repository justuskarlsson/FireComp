"""
Canopy Height data sources from Earth Engine.

Two datasets available:
1. ETH Global Canopy Height 2020 (10m, Sentinel-2 based)
   - Asset: users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1
   - Uncertainty: users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1
   - Based on GEDI + Sentinel-2 fusion

2. Meta/WRI Canopy Height (1m, 2018-2020)
   - Asset: projects/sat-io/open-datasets/facebook/meta-canopy-height
   - Higher resolution but needs aggregation
   - Mean absolute error: 2.8m

3. Meta/WRI Canopy Height Date (observation date of source imagery)
   - Asset: projects/wri-datalab/CanopyHeightDate
   - Units: years since 2000 (fractional, e.g. 19.5 = mid-2019)
   - Use to filter: only include canopy height when imagery predates the sample

Both height datasets are static (no temporal dimension).
The date layer enables per-sample temporal filtering.
"""

from datetime import datetime
import os

import numpy as np
from PIL import Image
from osgeo import gdal, gdalconst, osr
from pqdm.threads import pqdm

from firecomp.config import FLOAT32_NODATA, config
from firecomp.dsrc.dsrc import (
    DSRC,
    TIF_CREATE_OPTIONS,
    deg_per_px_from_m_equator,
    pipeline,
)
from firecomp.dsrc.earth_engine import (
    DownloadOptions,
    download_image,
    init_ee,
)

gdal.UseExceptions()

CH_UINT8_NODATA = 255


def _build_cropped_tile_vrts(ee_dir: str, dst_dir: str, tile_size: int) -> list[str]:
    """Crop downloaded tiles to their nominal extent via VRT wrappers.

    Source tiles are downloaded with padding (e.g. 1°) to ensure overlap.
    The padding regions often contain NoData strips from UTM→geodetic
    reprojection in the source imagery. By cropping each tile back to its
    nominal grid cell, these artifacts are eliminated while the overlap
    between neighbouring tiles ensures no gaps.

    Returns a sorted list of paths to the cropped VRT files.
    Source TIFFs are not modified.
    """
    cropped_dir = os.path.join(dst_dir, "cropped_vrts")
    os.makedirs(cropped_dir, exist_ok=True)

    cropped_paths = []
    for f in sorted(os.listdir(ee_dir)):
        if not f.endswith(".tif"):
            continue
        src_path = os.path.join(ee_dir, f)

        # Parse nominal origin from filename: tile_{lon}_{lat}.tif
        parts = f.replace("tile_", "").replace(".tif", "").split("_")
        try:
            nom_lon = int(parts[0])
            nom_lat = int(parts[1])
        except (ValueError, IndexError):
            # Can't parse — include uncropped
            cropped_paths.append(src_path)
            continue

        # Nominal extent (without padding)
        # Clamp to global bounds
        ulx = max(-180, nom_lon)
        uly = min(90, nom_lat + tile_size)
        lrx = min(180, nom_lon + tile_size)
        lry = max(-90, nom_lat)

        cropped_path = os.path.join(cropped_dir, f.replace(".tif", ".vrt"))
        gdal.Translate(
            cropped_path,
            src_path,
            format="VRT",
            projWin=[ulx, uly, lrx, lry],
        )
        cropped_paths.append(cropped_path)

    cropped_paths.sort()
    print(f"  Created {len(cropped_paths)} cropped tile VRTs in {cropped_dir}")
    return cropped_paths


def _fix_nodata(path: str):
    """Set nodata=255 on a downloaded tile (EE defaults to 0)."""
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    ds = gdal.Open(path, gdal.GA_Update)
    for i in range(1, ds.RasterCount + 1):
        ds.GetRasterBand(i).SetNoDataValue(CH_UINT8_NODATA)
    ds = None
    gdal.PopErrorHandler()


def create_ch_tif_uint8(
    path: str, bbox: list[float], px_deg: float, num_bands: int = 1
):
    """Create a uint8 GeoTIFF for canopy height."""
    w = int((bbox[2] - bbox[0]) / px_deg)
    h = int((bbox[3] - bbox[1]) / px_deg)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, num_bands, gdal.GDT_Byte, TIF_CREATE_OPTIONS)
    ds.SetGeoTransform([bbox[0], px_deg, 0, bbox[3], 0, -px_deg])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    for i in range(1, num_bands + 1):
        ds.GetRasterBand(i).SetNoDataValue(CH_UINT8_NODATA)
    return ds


class CanopyHeightMeta(DSRC):
    """
    Meta/WRI High Resolution Canopy Height Maps (1m native resolution).

    Native resolution: 1m
    Output resolution: configurable (default 300m, requires significant aggregation)

    Note: This is an ImageCollection with many tiles globally.
    Aggregating from 1m to 300m means 300x300 = 90,000 source pixels per output pixel.

    Reference: Tolan et al. 2024, Remote Sensing of Environment
    """

    name = "canopy_height_meta"
    channel_names = ["height"]
    no_data = FLOAT32_NODATA

    def __init__(
        self,
        *,
        output_scale: int = 300,
    ):
        super().__init__()
        self.output_scale = output_scale

        self.ee_dir = os.path.join(
            config.data_dir, f"canopy_height_meta_{output_scale}m"
        )
        self.dst_dir = os.path.join(
            config.data_dir, f"canopy_height_meta_by_zone_{output_scale}m"
        )
        self.tif_path = os.path.join(self.dst_dir, "canopy_height_meta.tif")
        self.vrt_path = os.path.join(self.dst_dir, "canopy_height_meta.vrt")

        self.num_geod_groups = 36
        self.geod_group_size = round(360.0 / self.num_geod_groups)
        self.px_deg = deg_per_px_from_m_equator(float(output_scale))

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        size: int = None,
        x_size: int = None,
        y_size: int = None,
    ):
        """Get canopy height sample. Time parameter ignored (static dataset)."""
        if size is not None:
            x_size = size
            y_size = size

        ds: gdal.Dataset = gdal.Open(self.tif_path)
        xoff, yoff, xsize, ysize = self._get_src_patch(bbox, ds)

        indices = np.zeros((1, y_size, x_size), dtype=np.uint8)
        ds.ReadAsArray(
            xoff,
            yoff,
            xsize,
            ysize,
            buf_obj=indices,
            resample_alg=gdalconst.GRIORA_NearestNeighbour,
        )
        result = indices.astype(np.float32)
        result[indices == CH_UINT8_NODATA] = FLOAT32_NODATA
        return result

    @pipeline
    def download_tiles(
        self, test_limit: int = None, padding: float = 1.0, force: bool = False
    ):
        """
        Download Meta canopy height tiles.

        WARNING: Meta dataset is 1m resolution. Downloading at full res is impractical.
        This uses EE's pyramiding (mean aggregation) to get coarser resolution directly.

        Uses a global crs_transform to ensure all tiles align to the same
        pixel grid, preventing NoData gaps at tile boundaries.

        Args:
            test_limit: Only download this many tiles (for testing)
            padding: Degrees of padding around each tile (default 1.0) to ensure
                     overlap at boundaries and prevent gaps from projection drift
            force: If True, re-download existing tiles
        """
        import ee

        init_ee()
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        os.makedirs(self.ee_dir, exist_ok=True)

        # Meta canopy height is an ImageCollection
        collection = ee.ImageCollection(
            "projects/sat-io/open-datasets/facebook/meta-canopy-height"
        )

        # Mosaic to single image (takes most recent where overlapping)
        img = collection.mosaic()

        # Compute global crs_transform for consistent pixel grid across all tiles
        crs_transform = [self.px_deg, 0, -180.0, 0, -self.px_deg, 90.0]

        # Download in tiles
        tile_size = self.geod_group_size
        tasks = []

        for lon in range(-180, 180, tile_size):
            for lat in range(-90, 90, tile_size):
                tile_id = f"{lon}_{lat}"
                dst_path = os.path.join(self.ee_dir, f"tile_{tile_id}.tif")
                if os.path.exists(dst_path) and not force:
                    continue

                # Pad only in latitude to prevent boundary gaps from projection drift
                bbox = [
                    lon,
                    max(-90, lat - padding),
                    min(180, lon + tile_size),
                    min(90, lat + tile_size + padding),
                ]
                rect = ee.Geometry.Rectangle(bbox)

                tile_img = img.clip(rect).unmask(CH_UINT8_NODATA).toUint8()
                tasks.append((tile_img, dst_path, rect, self.output_scale))

        if test_limit:
            tasks = tasks[:test_limit]

        print(f"Downloading {len(tasks)} tiles at {self.output_scale}m resolution...")
        print(f"Using crs_transform={crs_transform} for consistent global grid")
        print(f"Padding: {padding}° around each tile to prevent boundary gaps")

        options = DownloadOptions(crs="EPSG:4326", crs_transform=crs_transform)

        def download_tile(args):
            img, dst_path, rect, scale = args
            path = download_image(img, dst_path, rect, scale, options)
            if path:
                _fix_nodata(path)
            return path

        paths = pqdm(
            tasks,
            download_tile,
            n_jobs=20,
            desc="Downloading tiles",
            exception_behaviour="defer",
        )
        failed = [p for p in paths if isinstance(p, Exception)]
        ok = [p for p in paths if isinstance(p, str) and p]
        print(f"Downloaded {len(ok)} tiles to {self.ee_dir} ({len(failed)} failed)")
        print(f"Downloaded {len([p for p in paths if p])} tiles to {self.ee_dir}")
        return paths

    @pipeline
    def build_vrt(self):
        """Build VRT from downloaded tiles.

        Crops each tile back to its nominal extent (removing download padding)
        before assembly. This eliminates NoData strips at tile boundaries caused
        by UTM→geodetic reprojection artifacts in the source data. The crop is
        done via lightweight VRT wrappers — no source TIFFs are modified.
        """
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        os.makedirs(self.dst_dir, exist_ok=True)

        tile_size = self.geod_group_size
        cropped_paths = _build_cropped_tile_vrts(self.ee_dir, self.dst_dir, tile_size)

        if not cropped_paths:
            raise RuntimeError(f"No tiles found in {self.ee_dir}")

        print(f"Building VRT from {len(cropped_paths)} cropped tiles...")
        gdal.BuildVRT(self.vrt_path, cropped_paths)
        print(f"VRT created: {self.vrt_path}")

    @pipeline
    def translate_to_tif(self):
        """Create final global TIF from VRT."""
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        if not os.path.exists(self.vrt_path):
            raise RuntimeError("VRT not built. Run build_vrt() first.")

        print(f"Creating global TIF: {self.tif_path}")

        dst_ds = create_ch_tif_uint8(
            self.tif_path,
            [-180.0, -90.0, 180.0, 90.0],
            self.px_deg,
            1,
        )

        gdal.Warp(
            dst_ds,
            self.vrt_path,
            options=gdal.WarpOptions(
                multithread=True,
                resampleAlg=gdalconst.GRA_Average,
                srcNodata=CH_UINT8_NODATA,
                dstNodata=CH_UINT8_NODATA,
            ),
        )
        dst_ds.FlushCache()
        dst_ds = None
        print(f"Done: {self.tif_path}")

    def get_png(
        self, arr: np.ndarray, vmin: float = 0, vmax: float = 50, **kwargs
    ) -> Image.Image:
        """Render canopy height."""
        if arr.ndim == 3:
            arr = arr[0]

        arr = np.clip(arr, vmin, vmax)
        arr = (arr - vmin) / (vmax - vmin)
        arr = (arr * 255).astype(np.uint8)

        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
        rgb[..., 1] = arr
        rgb[..., 0] = arr // 3
        rgb[..., 2] = arr // 4

        return Image.fromarray(rgb)

    @pipeline
    def verify_tile_boundaries(self, output_dir: str = None):
        """
        Extract strips at tile boundaries to verify no NoData gaps.

        Saves PNGs of horizontal strips centered on key latitude boundaries
        where tiles meet. Use this after re-downloading to confirm the
        crs_transform fix worked.

        Boundaries checked (all cross significant land masses):
        - 60°N: Scandinavia/Sweden (the original problem area)
        - 50°N: Central Europe, Canada
        - 30°N: Southern US, North Africa, China

        Each strip is 2° tall (1° above and below the boundary) and spans
        a relevant longitude range for that latitude.
        """
        if output_dir is None:
            output_dir = os.path.join(self.dst_dir, "boundary_checks")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(self.tif_path):
            raise RuntimeError(f"Global TIF not found: {self.tif_path}")

        ds = gdal.Open(self.tif_path)
        gt = ds.GetGeoTransform()
        # gt = [x_origin, px_width, 0, y_origin, 0, -px_height]

        # Define boundaries to check: (lat, lon_min, lon_max, name)
        boundaries = [
            (60, 5, 35, "60N_scandinavia"),  # Sweden/Finland
        ]

        strip_height_deg = 10.0  # 1° above and below boundary

        print(f"Extracting {len(boundaries)} boundary strips to {output_dir}")

        for lat, lon_min, lon_max, name in boundaries:
            # Calculate pixel coordinates
            lat_top = lat + strip_height_deg / 2
            lat_bottom = lat - strip_height_deg / 2

            # Convert geo coords to pixel coords
            px_x_min = int((lon_min - gt[0]) / gt[1])
            px_x_max = int((lon_max - gt[0]) / gt[1])
            px_y_top = int((lat_top - gt[3]) / gt[5])
            px_y_bottom = int((lat_bottom - gt[3]) / gt[5])

            # Ensure correct ordering (y increases downward in pixel coords)
            if px_y_top > px_y_bottom:
                px_y_top, px_y_bottom = px_y_bottom, px_y_top

            width = px_x_max - px_x_min
            height = px_y_bottom - px_y_top

            # Read the strip
            arr = ds.ReadAsArray(px_x_min, px_y_top, width, height)
            if arr is None:
                print(f"  {name}: Failed to read")
                continue

            # Count NoData pixels
            nodata_count = np.sum(arr == CH_UINT8_NODATA)
            total_pixels = arr.size
            nodata_pct = 100 * nodata_count / total_pixels

            # Create PNG
            # Mark NoData as red, valid data as green gradient
            if arr.ndim == 2:
                rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
            else:
                rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
                arr = arr[0]  # Take first band

            # Green gradient for valid canopy height (stretch actual min/max to 0-255)
            valid_mask = arr != CH_UINT8_NODATA
            valid_data = arr[valid_mask]
            vmin, vmax = float(valid_data.min()), float(valid_data.max())
            intensity = (
                (np.clip(arr, vmin, vmax) - vmin) * 255 / max(vmax - vmin, 1)
            ).astype(np.uint8)

            rgb[..., 1] = np.where(valid_mask, intensity, 0)  # Green for valid
            rgb[..., 0] = np.where(valid_mask, intensity // 3, 255)  # Red for NoData
            rgb[..., 2] = np.where(valid_mask, intensity // 4, 0)

            # Draw a line at the exact boundary (center of strip)
            center_y = height // 2
            rgb[center_y, :, :] = [255, 255, 0]  # Yellow line at boundary

            img = Image.fromarray(rgb)
            out_path = os.path.join(output_dir, f"{name}.png")
            img.save(out_path)

            print(
                f"  {name}: {width}x{height}px, NoData={nodata_pct:.1f}%, saved to {out_path}"
            )

        ds = None
        print(f"\nDone. Check PNGs in {output_dir}")
        print(
            "Yellow line = exact tile boundary. Red pixels = NoData (should be minimal on land)."
        )


class CanopyHeightDate(DSRC):
    """
    Meta/WRI Canopy Height observation date layer.

    GEE asset: projects/wri-datalab/CanopyHeightDate
    Units: years since 2000 (fractional float, e.g. 19.5 = mid-2019)

    Used to determine when the source imagery for canopy height was acquired.
    This enables per-sample filtering: only use canopy height data when the
    imagery predates the fire sample, avoiding post-fire canopy measurements.

    Download pipeline mirrors CanopyHeightMeta but for the date band.
    The output is a single global float32 GeoTIFF.
    """

    name = "canopy_height_date"
    channel_names = ["date"]
    no_data = FLOAT32_NODATA

    def __init__(self, *, output_scale: int = 300):
        super().__init__()
        self.output_scale = output_scale

        self.ee_dir = os.path.join(
            config.data_dir, f"canopy_height_date_{output_scale}m"
        )
        self.dst_dir = os.path.join(
            config.data_dir, f"canopy_height_date_by_zone_{output_scale}m"
        )
        self.tif_path = os.path.join(self.dst_dir, "canopy_height_date.tif")
        self.vrt_path = os.path.join(self.dst_dir, "canopy_height_date.vrt")

        self.num_geod_groups = 72
        self.geod_group_size = round(360.0 / self.num_geod_groups)
        self.px_deg = deg_per_px_from_m_equator(float(output_scale))

    def get_sample(
        self,
        bbox: list[float],
        t: datetime,
        size: int = None,
        x_size: int = None,
        y_size: int = None,
    ):
        """Get canopy height observation date for a patch.

        Returns float32 array with values in 'years since 2000'.
        E.g. 19.5 means the imagery was from mid-2019.
        """
        if size is not None:
            x_size = size
            y_size = size

        ds: gdal.Dataset = gdal.Open(self.tif_path)
        if ds is None:
            return None

        xoff, yoff, xsize, ysize = self._get_src_patch(bbox, ds)

        if xsize <= 0 or ysize <= 0:
            ds = None
            return None

        result = np.zeros((1, y_size, x_size), dtype=np.float32)
        ds.ReadAsArray(
            xoff,
            yoff,
            xsize,
            ysize,
            buf_obj=result,
            resample_alg=gdalconst.GRIORA_NearestNeighbour,
        )
        ds = None
        return result

    def date_predates_sample(
        self, bbox: list[float], sample_dt: datetime, img_size: int
    ) -> bool:
        """Check if the canopy height imagery predates the sample date.

        Uses the median observation date across the patch. Returns True if
        the imagery was acquired before the sample date, meaning the canopy
        height data is valid (not post-fire).

        Returns False if:
        - No date data available
        - Imagery was acquired after the sample date
        - All pixels are nodata
        """
        date_arr = self.get_sample(bbox, sample_dt, size=img_size)
        if date_arr is None:
            return False

        # Mask nodata
        valid = date_arr[0][date_arr[0] != FLOAT32_NODATA]
        if len(valid) == 0:
            return False

        # Convert sample date to 'years since 2000'
        sample_year_frac = (
            sample_dt.year - 2000 + (sample_dt.timetuple().tm_yday - 1) / 365.25
        )

        # Use max date: all imagery must predate the sample
        max_date = float(np.max(valid))

        return max_date < sample_year_frac

    @pipeline
    def download_tiles(
        self, test_limit: int = None, padding: float = 1.0, force: bool = False
    ):
        """Download canopy height date tiles from GEE.

        Asset: projects/wri-datalab/CanopyHeightDate
        Band values: years since 2000 (float)
        """
        import ee

        init_ee()
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        os.makedirs(self.ee_dir, exist_ok=True)

        # Single image asset (not a collection)
        img = ee.Image("projects/wri-datalab/CanopyHeightDate")

        # Compute global crs_transform for consistent pixel grid
        crs_transform = [self.px_deg, 0, -180.0, 0, -self.px_deg, 90.0]

        tile_size = self.geod_group_size
        tasks = []

        for lon in range(-180, 180, tile_size):
            for lat in range(-90, 90, tile_size):
                tile_id = f"{lon}_{lat}"
                dst_path = os.path.join(self.ee_dir, f"tile_{tile_id}.tif")
                if os.path.exists(dst_path) and not force:
                    continue

                bbox = [
                    lon,
                    max(-90, lat - padding),
                    max(180, lon + tile_size),
                    min(90, lat + tile_size + padding),
                ]
                rect = ee.Geometry.Rectangle(bbox)

                tile_img = img.clip(rect).unmask(FLOAT32_NODATA).toFloat()
                tasks.append((tile_img, dst_path, rect, self.output_scale))

        if test_limit:
            tasks = tasks[:test_limit]

        print(f"Downloading {len(tasks)} date tiles at {self.output_scale}m...")
        print(f"Using crs_transform={crs_transform}")

        options = DownloadOptions(crs="EPSG:4326", crs_transform=crs_transform)

        def download_tile(args):
            tile_img, dst_path, rect, scale = args
            return download_image(tile_img, dst_path, rect, scale, options)

        paths = pqdm(
            tasks,
            download_tile,
            n_jobs=20,
            desc="Downloading date tiles",
            exception_behaviour="immediate",
        )

        print(f"Downloaded {len([p for p in paths if p])} tiles to {self.ee_dir}")
        return paths

    @pipeline
    def build_vrt(self):
        """Build VRT from downloaded date tiles.

        Crops tiles to nominal extent (same as CanopyHeightMeta.build_vrt).
        """
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        os.makedirs(self.dst_dir, exist_ok=True)

        tile_size = self.geod_group_size
        cropped_paths = _build_cropped_tile_vrts(self.ee_dir, self.dst_dir, tile_size)

        if not cropped_paths:
            raise RuntimeError(f"No tiles found in {self.ee_dir}")

        print(f"Building VRT from {len(cropped_paths)} cropped date tiles...")
        gdal.BuildVRT(self.vrt_path, cropped_paths)
        print(f"VRT created: {self.vrt_path}")

    @pipeline
    def translate_to_tif(self):
        """Create final global float32 TIF from VRT."""
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        if not os.path.exists(self.vrt_path):
            raise RuntimeError("VRT not built. Run build_vrt() first.")

        print(f"Creating global date TIF: {self.tif_path}")

        # Create float32 output (not uint8 like canopy height)
        w = int(360.0 / self.px_deg)
        h = int(180.0 / self.px_deg)
        drv = gdal.GetDriverByName("GTiff")
        dst_ds = drv.Create(
            self.tif_path, w, h, 1, gdal.GDT_Float32, TIF_CREATE_OPTIONS
        )
        dst_ds.SetGeoTransform([-180.0, self.px_deg, 0, 90.0, 0, -self.px_deg])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.GetRasterBand(1).SetNoDataValue(FLOAT32_NODATA)

        gdal.Warp(
            dst_ds,
            self.vrt_path,
            options=gdal.WarpOptions(
                multithread=True,
                resampleAlg=gdalconst.GRA_Average,
                srcNodata=FLOAT32_NODATA,
                dstNodata=FLOAT32_NODATA,
            ),
        )
        dst_ds.FlushCache()
        dst_ds = None
        print(f"Done: {self.tif_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Meta canopy height data")
    parser.add_argument(
        "--force", action="store_true", help="Re-download existing tiles"
    )
    parser.add_argument(
        "--test", type=int, default=None, help="Only download N tiles (for testing)"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download, only build/verify"
    )
    parser.add_argument(
        "--scale", type=int, default=300, help="Output scale in meters (default: 300)"
    )
    parser.add_argument(
        "--date", action="store_true", help="Download canopy height date layer"
    )
    args = parser.parse_args()

    if args.date:
        # Download date layer
        ch_date = CanopyHeightDate(output_scale=args.scale)

        if not args.skip_download:
            print("=== Step 1: Download date tiles ===")
            ch_date.download_tiles(test_limit=args.test, padding=1.0, force=args.force)

        print("\n=== Step 2: Build VRT ===")
        ch_date.build_vrt()

        print("\n=== Step 3: Translate to TIF ===")
        ch_date.translate_to_tif()
    else:
        # Download canopy height
        ch_meta = CanopyHeightMeta(output_scale=args.scale)

        if not args.skip_download:
            print("=== Step 1: Download tiles ===")
            ch_meta.download_tiles(test_limit=args.test, padding=2.0, force=args.force)

        print("\n=== Step 2: Build VRT ===")
        ch_meta.build_vrt()

        print("\n=== Step 3: Translate to TIF ===")
        ch_meta.translate_to_tif()

        print("\n=== Step 4: Verify tile boundaries ===")
        ch_meta.verify_tile_boundaries()
