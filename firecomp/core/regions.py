"""
core/regions.py — Region registry, raster I/O, grouping, and rasterization.

Regions is a thin object for lookups + a static grouping method that slices
Metrics by region. It doesn't run models or accumulate state. Grouping is
post-hoc: compute Metrics once, group after.

RegionRaster handles the 0.1-degree global raster (wildfire_regions.tif)
and coordinate-to-region lookups. Used by cluster generation, preprocess,
and fire-region assignment. `rasterize_regions()` builds that raster from
the WildfireRegions.kml polygon file.
"""

import json
import os
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()


class Regions:
    """
    Immutable registry of wildfire study regions.

    Usage:
        regions = Regions()                              # loads default
        regions = Regions("data/regions/wildfire_regions.json")
        regions.names()                                  # ["California", "Mediterranean", ...]
        regions["Mediterranean"]                         # {"id": 3, "bbox": [...]}
        regions.id_to_name(3)                            # "Mediterranean"

    Grouping:
        from core.metrics import Metrics
        m = Metrics(pred, target, mask)
        by_region = Regions.group(m, samples)             # {name: Metrics}
    """

    _default_path = None   # set at import time or via config

    def __init__(self, path: str = None):
        if path is None:
            path = self._resolve_default_path()
        with open(path) as f:
            raw = json.load(f)
        # raw is {str(int_id): name} e.g. {"1": "California", "2": "Oregon"}
        self._id_to_name = {int(k): v for k, v in raw.items()}
        self._name_to_id = {v: k for k, v in self._id_to_name.items()}

    def names(self) -> list[str]:
        """All region names, sorted by id."""
        return [self._id_to_name[k] for k in sorted(self._id_to_name)]

    def ids(self) -> list[int]:
        return sorted(self._id_to_name.keys())

    def id_to_name(self, region_id: int) -> str:
        return self._id_to_name.get(region_id, f"unknown_{region_id}")

    def name_to_id(self, name: str) -> int:
        return self._name_to_id[name]

    def __getitem__(self, name: str) -> int:
        """regions["Mediterranean"] -> region id."""
        return self._name_to_id[name]

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_id

    def __len__(self) -> int:
        return len(self._id_to_name)

    def __iter__(self):
        return iter(self.names())

    # --- grouping ---

    @staticmethod
    def group(metrics, samples, fire_regions: dict[int, int] = None) -> dict[str, "Metrics"]:
        """
        Group per-sample metrics by region.

        Args:
            metrics:      a Metrics object (has .per_sample)
            samples:      list[Sample] matching the metrics (same length, same order)
            fire_regions: dict mapping fire_id -> region_id.
                          If None, samples must have a .region_id attribute.

        Returns:
            {region_name: Metrics} for each region with ≥1 sample.

        Usage:
            m = Metrics(pred, target, mask)
            by_region = Regions.group(m, ds.test_samples, fire_regions=fire_regions)
            print(by_region["Mediterranean"].f1)
        """
        from firecomp.core.metrics import Metrics as M

        per_sample = metrics.per_sample
        assert len(per_sample) == len(samples), (
            f"Metrics has {len(per_sample)} samples but got {len(samples)} sample metadata"
        )

        # bucket per-sample metrics by region
        regions_inst = Regions()
        buckets = {}   # region_name -> list[SampleMetrics]

        for sm, sample in zip(per_sample, samples):
            if fire_regions is not None:
                rid = fire_regions.get(getattr(sample, "fire_id", None), 0)
            else:
                rid = getattr(sample, "region_id", 0)
            name = regions_inst.id_to_name(rid)
            buckets.setdefault(name, []).append(sm)

        return {name: M.from_subset(sms) for name, sms in buckets.items()}

    @staticmethod
    def build_fire_regions(samples) -> dict[int, int]:
        """Build fire_id → region_id from sample (lon, lat) via RegionRaster.

        Deduplicates by fire_id (uses first sample's coordinates per fire).
        Raises on failure — no silent fallback.
        """
        rr = RegionRaster.load()
        fire_coords: dict[int, tuple[float, float]] = {}
        for s in samples:
            fid = getattr(s, "fire_id", None)
            if fid is not None and fid not in fire_coords:
                fire_coords[fid] = (s.lon, s.lat)
        if not fire_coords:
            return {}
        fire_ids = list(fire_coords)
        lons = np.array([fire_coords[fid][0] for fid in fire_ids])
        lats = np.array([fire_coords[fid][1] for fid in fire_ids])
        region_ids = rr.lookup_coords(lons, lats)
        return {fid: int(rid) for fid, rid in zip(fire_ids, region_ids)}

    @staticmethod
    def filter(samples, include=None, exclude=None,
               fire_regions: dict[int, int] | None = None):
        """Filter *samples* to those whose fire is in *include* (or not in *exclude*).

        Args:
            samples:      list of Sample objects with a .fire_id attribute.
            include:      keep only samples whose region is in this list of names.
            exclude:      drop samples whose region is in this list of names.
            fire_regions: pre-built fire_id → region_id mapping.  Built via
                          build_fire_regions() if not supplied.

        Returns the original list unchanged when neither include nor exclude is
        set.  Raises on failure — never silently returns unfiltered data.
        """
        if include is None and exclude is None:
            return samples
        if fire_regions is None:
            fire_regions = Regions.build_fire_regions(samples)

        regions = Regions()

        def _region_id(s):
            return fire_regions.get(getattr(s, "fire_id", None), 0)

        if include is not None:
            include_ids = {regions.name_to_id(n) for n in include}
            return [s for s in samples if _region_id(s) in include_ids]

        exclude_ids = {regions.name_to_id(n) for n in exclude}
        return [s for s in samples if _region_id(s) not in exclude_ids]

    @staticmethod
    def _resolve_default_path():
        # try common locations
        from firecomp.config import config
        candidates = [
            os.path.join(config.data_dir, "regions", "wildfire_regions.json"),
            "data/regions/wildfire_regions.json",
            os.path.join(os.environ.get("FIRECOMP_DATA", ""), "regions/wildfire_regions.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError("No regions JSON found. Pass path explicitly.")


# ---------------------------------------------------------------------------
# RegionRaster — 0.1-degree global raster for coordinate lookups
# ---------------------------------------------------------------------------

# Raster params: global WGS84, 0.1 deg
RASTER_XMIN, RASTER_YMIN = -180.0, -90.0
RASTER_XMAX, RASTER_YMAX = 180.0, 90.0
RASTER_PIX = 0.1
RASTER_COLS = int((RASTER_XMAX - RASTER_XMIN) / RASTER_PIX)  # 3600
RASTER_ROWS = int((RASTER_YMAX - RASTER_YMIN) / RASTER_PIX)  # 1800


class RegionRaster:
    """Global region raster (wildfire_regions.tif) and coordinate lookups.

    The raster is 3600x1800 (0.1 deg), values 0 = no region, 1-9 = region id.

    Usage:
        rr = RegionRaster.load()
        raster, names = rr.raster, rr.names
        fire_regions = rr.lookup_fires(stats)

    For cluster generation, pass rr.raster and rr.gt to sampling routines.
    """

    def __init__(self, raster: np.ndarray, names: dict[int, str], gt: tuple):
        self.raster = raster         # (1800, 3600) uint8
        self.names = names           # {1: "California", ...}
        self.gt = gt                 # GDAL geotransform

    @staticmethod
    def load(tif_path: str = None, json_path: str = None) -> "RegionRaster":
        """Load region raster + name mapping from disk."""
        from firecomp.config import config

        region_dir = os.path.join(config.data_dir, "regions")
        if tif_path is None:
            tif_path = os.path.join(region_dir, "wildfire_regions.tif")
        if json_path is None:
            json_path = os.path.join(region_dir, "wildfire_regions.json")

        ds = gdal.Open(tif_path)
        raster = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        ds = None

        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        names = {int(k): v for k, v in mapping.items()}
        return RegionRaster(raster, names, gt)

    def valid_ids(self) -> list[int]:
        """Sorted list of region ids (excluding 0)."""
        return sorted(self.names.keys())

    def lookup_fires(self, stats) -> np.ndarray:
        """Map fire centroids to region ids.

        Args:
            stats: Fires.Stats with min_x, max_x, min_y, max_y.

        Returns:
            (N,) int array of region ids (0 = no region).
        """
        cx = (stats.min_x + stats.max_x) / 2.0
        cy = (stats.min_y + stats.max_y) / 2.0
        return self.lookup_coords(cx, cy)

    def lookup_coords(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Map (lon, lat) arrays to region ids.

        Args:
            lons, lats: coordinate arrays (same length).

        Returns:
            (N,) int array of region ids (0 = no region).
        """
        px = ((lons - RASTER_XMIN) / RASTER_PIX).astype(np.int32)
        py = ((RASTER_YMAX - lats) / RASTER_PIX).astype(np.int32)
        px = np.clip(px, 0, self.raster.shape[1] - 1)
        py = np.clip(py, 0, self.raster.shape[0] - 1)
        return self.raster[py, px]


def rasterize_regions(kml_path: str = None, out_dir: str = None) -> None:
    """Rasterize WildfireRegions.kml into the global 0.1-degree byte raster.

    Writes `wildfire_regions.tif` (0 = no region, 1-9 = region index) and
    `wildfire_regions.json` ({index: name}) into `out_dir`.
    """
    from firecomp.config import config

    kml_path = kml_path or os.path.join(config.data_dir, "WildfireRegions.kml")
    out_dir = out_dir or os.path.join(config.data_dir, "regions")
    tif_path = os.path.join(out_dir, "wildfire_regions.tif")
    json_path = os.path.join(out_dir, "wildfire_regions.json")
    os.makedirs(out_dir, exist_ok=True)

    ds = ogr.Open(kml_path)
    if ds is None:
        raise RuntimeError(f"Could not open KML: {kml_path}")
    lyr = ds.GetLayer(0)
    print(f"Layer: {lyr.GetName()}, features: {lyr.GetFeatureCount()}")

    mapping = {}
    features_data = []
    idx = 1
    for feat in lyr:
        name = feat.GetField("Name")
        if feat.GetGeometryRef() is not None:
            mapping[idx] = name
            features_data.append((idx, feat.GetFID()))
            print(f"  {idx}: {name}")
            idx += 1
    lyr.ResetReading()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Saved mapping to {json_path}")

    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(tif_path, RASTER_COLS, RASTER_ROWS, 1, gdal.GDT_Byte,
                     options=["TILED=YES", "COMPRESS=LZW"])
    dst.SetGeoTransform((RASTER_XMIN, RASTER_PIX, 0, RASTER_YMAX, 0, -RASTER_PIX))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst.SetProjection(srs.ExportToWkt())
    band = dst.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)

    for region_idx, fid in features_data:
        lyr.SetAttributeFilter(f"FID = {fid}")
        gdal.RasterizeLayer(dst, [1], lyr, burn_values=[region_idx],
                            options=["ALL_TOUCHED=FALSE"])
        lyr.SetAttributeFilter(None)

    dst.FlushCache()
    dst = None
    ds = None
    print(f"Saved raster to {tif_path} ({RASTER_COLS}x{RASTER_ROWS} px, {RASTER_PIX} deg)")


if __name__ == "__main__":
    rasterize_regions()
