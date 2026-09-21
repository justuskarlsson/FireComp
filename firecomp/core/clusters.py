"""
core/clusters.py — Land-cover cluster priors, raster I/O, and generation.

Clustering pipeline (run once per year):
    python -m firecomp.core.clusters --year 2020 --n-clusters 32
    python -m firecomp.core.clusters --smoke          # quick test (~2 min)

The pipeline samples AE embeddings globally, trains linear probes for
land-cover calibration, clusters in calibrated space, writes a raster,
and computes per-cluster fire / LC statistics.

Output directory: data/cluster_copernicus/
    cluster_stats_{K}_{year}.npz    — per-cluster fire/LC statistics
    cluster_raster_{K}_{year}.tif   — pixel-level cluster-ID raster (uint8)
    tile_metadata_{K}_{year}.npz    — per-tile cluster-fraction summaries
    calibrated_model_{K}_{year}.npz — probe weights + KMeans centres

Consumer interface:
    ClusterPriors.load(year, n_clusters)
    get_cluster_priors(ae_year)
    read_cluster_window(ae_year, bbox, size)
    read_cluster_stats(ae_year, bbox, size)
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firecomp.config import config


# ============================================================================
# Constants
# ============================================================================

CLUSTER_STATS_CHANNELS = ["tame_to_wild_ratio", "fire_density", "mean_canopy_height"]
N_CLUSTER_STATS = len(CLUSTER_STATS_CHANNELS)

# Copernicus CGLS-LC100 layer definitions (used by generation pipeline)
_LC_LABELS = {
    20: "Shrub", 30: "Grass", 40: "Cropland", 50: "Urban",
    60: "Bare", 80: "Water", 90: "Wetland", 100: "Moss_Lichen",
    111: "EG_Needle", 112: "EG_Broad", 113: "DC_Needle", 114: "DC_Broad",
    115: "Mixed_Forest", 116: "Other_Forest",
    121: "EG_Needle", 122: "EG_Broad", 123: "DC_Needle", 124: "DC_Broad",
    125: "Mixed_Forest", 126: "Other_Forest",
    200: "Ocean",
}
_LC_SKIP = {0, 70}  # Unknown, Snow/Ice

_COPERNICUS_LAYERS = [
    ("lc_class", "discrete_classification.tif", "near"),
    ("tree_pct", "tree_coverfraction.tif", "average"),
    ("shrub_pct", "shrub_coverfraction.tif", "average"),
    ("grass_pct", "grass_coverfraction.tif", "average"),
    ("crop_pct", "crops_coverfraction.tif", "average"),
    ("builtup_pct", "builtup_coverfraction.tif", "average"),
    ("bare_pct", "bare_coverfraction.tif", "average"),
]
_FRACTION_NAMES = [n for n, _, _ in _COPERNICUS_LAYERS if n != "lc_class"]
_CONTINUOUS_NAMES = ["height"] + _FRACTION_NAMES

# Smoke test eval region (California)
_SMOKE_EVAL_REGION = {
    "lon": (-125.0, -114.0), "lat": (32.0, 42.0), "label": "California",
}


# ============================================================================
# ClusterPriors — load precomputed cluster artefacts
# ============================================================================

@dataclass
class ClusterPriors:
    """LC-calibrated cluster priors (from the generation pipeline below).

    Burnability is determined at training time via tame_to_wild_ratio threshold.
    """

    cluster_stats: dict[str, np.ndarray]
    cluster_raster_path: str
    tile_metadata_path: str
    n_clusters: int
    year: int

    # -- Construction --------------------------------------------------------

    @staticmethod
    def load(year: int, n_clusters: int = 32) -> "ClusterPriors":
        """Load precomputed cluster artefacts for *year* with *n_clusters*."""
        base = Path(config.data_dir) / "cluster_copernicus"
        stats_path = base / f"cluster_stats_{n_clusters}_{year}.npz"
        raster_path = base / f"cluster_raster_{n_clusters}_{year}.tif"
        tile_path = base / f"tile_metadata_{n_clusters}_{year}.npz"

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Cluster stats not found: {stats_path}\n"
                f"Run: python -m firecomp.core.clusters --year {year}")

        data = np.load(str(stats_path), allow_pickle=True)
        stats = {k: data[k] for k in data.files}
        return ClusterPriors(
            cluster_stats=stats,
            cluster_raster_path=str(raster_path),
            tile_metadata_path=str(tile_path),
            n_clusters=n_clusters,
            year=year,
        )

    # -- Cluster burnability -------------------------------------------------

    def non_burnable_ids(self, threshold: float = 2.0) -> set[int]:
        """Cluster IDs where tame_to_wild_ratio > threshold."""
        ratio = self.cluster_stats["tame_to_wild_ratio"]
        return {int(c) for c in range(self.n_clusters) if ratio[c] > threshold}

    def burnable_ids(self, threshold: float = 2.0) -> set[int]:
        """Cluster IDs where tame_to_wild_ratio <= threshold."""
        ratio = self.cluster_stats["tame_to_wild_ratio"]
        return {int(c) for c in range(self.n_clusters) if ratio[c] <= threshold}

    def non_burnable_tile_centres(self, threshold: float = 2.0
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lons, lats) of tile centres dominated by non-burnable clusters.

        Tiles where >50 % of pixels belong to non-burnable clusters are selected.

        Returns:
            (lons, lats): 1-D float arrays of tile centres.
        """
        meta = np.load(self.tile_metadata_path)
        fracs = meta["cluster_fractions"]   # (n_tiles, n_clusters)
        lons = meta["center_lon"]
        lats = meta["center_lat"]

        nb_ids = sorted(self.non_burnable_ids(threshold))
        if len(nb_ids) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        nb_frac = fracs[:, nb_ids].sum(axis=1)
        nb_mask = nb_frac > 0.5
        return lons[nb_mask], lats[nb_mask]

    # -- Per-pixel lookups ---------------------------------------------------

    def cluster_stats_for_patch(self, cid_patch: np.ndarray) -> np.ndarray:
        """Convert cluster-ID patch (H, W) uint8 -> stats (3, H, W) float32.

        Channels: tame_to_wild_ratio, fire_density, mean_canopy_height.
        """
        h, w = cid_patch.shape
        out = np.zeros((N_CLUSTER_STATS, h, w), dtype=np.float32)
        for ci, key in enumerate(CLUSTER_STATS_CHANNELS):
            lut = self.cluster_stats[key].astype(np.float32)
            safe_ids = np.clip(cid_patch, 0, len(lut) - 1)
            out[ci] = lut[safe_ids]
        return out


# ============================================================================
# Cluster raster I/O — read pixel-level cluster IDs from GeoTIFF
# ============================================================================

_priors_cache: dict[int, ClusterPriors] = {}
_raster_cache: dict[int, tuple] = {}


def get_cluster_priors(ae_year: int, n_clusters: int = 32) -> ClusterPriors:
    """Cached loader for ClusterPriors."""
    if ae_year not in _priors_cache:
        _priors_cache[ae_year] = ClusterPriors.load(year=ae_year, n_clusters=n_clusters)
    return _priors_cache[ae_year]


def _get_cluster_raster(ae_year: int, n_clusters: int = 32):
    """Open (and cache) the cluster-ID raster for *ae_year*."""
    if ae_year not in _raster_cache:
        from osgeo import gdal
        cp = get_cluster_priors(ae_year, n_clusters)
        ds = gdal.Open(cp.cluster_raster_path)
        gt = ds.GetGeoTransform()
        _raster_cache[ae_year] = (ds, ds.GetRasterBand(1), gt,
                                  ds.RasterXSize, ds.RasterYSize)
    return _raster_cache[ae_year]


def read_cluster_window(ae_year: int, bbox: list[float],
                        size: int = 256) -> np.ndarray:
    """Read a size x size uint8 cluster-ID patch from year-matched raster.

    Args:
        ae_year: AE embedding year (usually fire_year - 1).
        bbox: [lon_min, lat_min, lon_max, lat_max].
        size: output patch edge length in pixels.

    Returns:
        (size, size) uint8 array of cluster IDs.
    """
    ds_ref, band, gt, xsize, ysize = _get_cluster_raster(ae_year)
    lon_min, lat_min, lon_max, lat_max = bbox
    px0 = int((lon_min - gt[0]) / gt[1])
    py0 = int((lat_max - gt[3]) / gt[5])
    px0 = max(0, min(px0, xsize - size))
    py0 = max(0, min(py0, ysize - size))
    window = band.ReadAsArray(px0, py0, size, size)
    if window is None:
        return np.zeros((size, size), dtype=np.uint8)
    return window.astype(np.uint8)


def read_cluster_stats(ae_year: int, bbox: list[float],
                       size: int = 256) -> np.ndarray:
    """Read cluster-stats patch (3, size, size) float32 for a bbox.

    Convenience wrapper: reads the cluster-ID raster, then maps each pixel
    to its per-cluster statistics via ClusterPriors.
    """
    cid_patch = read_cluster_window(ae_year, bbox, size)
    cp = get_cluster_priors(ae_year)
    return cp.cluster_stats_for_patch(cid_patch)


# ############################################################################
#
#  GENERATION PIPELINE
#
#  Everything below is the offline cluster-building pipeline.
#  Heavy deps (sklearn, gdal, h5py, matplotlib) are imported locally.
#
#  Entry point: main()  /  python -m firecomp.core.clusters
#
# ############################################################################


# ============================================================================
# GenerateConfig — all knobs for the generation pipeline
# ============================================================================

@dataclass
class GenerateConfig:
    """Configuration for the cluster generation pipeline."""
    year: int = 2017
    n_clusters: int = 32
    n_samples: int = 150_000
    seed: int = 42
    smoke: bool = False
    skip_fire_overlay: bool = False
    skip_tile_metadata: bool = False
    skip_plots: bool = False


# ============================================================================
# main — CLI entry point
# ============================================================================

def main():
    """Generate per-year cluster artefacts from AE embeddings."""
    from firecomp.core.cli import parse_config

    cfg, _ = parse_config(GenerateConfig, prog="firecomp.core.clusters",
                          description="Calibrated AE clustering "
                          "— generate per-year cluster artefacts.")
    _generate(cfg)


def _generate(cfg: GenerateConfig):
    """Run the full generation pipeline."""
    t0_global = time.time()

    n_samples = cfg.n_samples
    if cfg.smoke:
        n_samples = min(n_samples, 5000)
        print(f"[SMOKE] n_samples={n_samples}, n_clusters={cfg.n_clusters}")

    from sklearn.cluster import MiniBatchKMeans

    out_dir = os.path.join(config.data_dir, "cluster_copernicus")
    os.makedirs(out_dir, exist_ok=True)
    prefix = "smoke_" if cfg.smoke else ""

    ae_tif = os.path.join(
        config.data_dir, f"ae_embeddings_by_zone_{cfg.year}/ae_embeddings.tif")
    canopy_tif = os.path.join(
        config.data_dir, "canopy_height_meta_by_zone_300m/canopy_height_meta.tif")
    cop_dir = os.path.join(config.data_dir, "copernicus_lc100")

    # Ensure Copernicus layers are co-registered to AE grid
    cop_paths = _ensure_copernicus_on_ae_grid(ae_tif, cop_dir)

    # Load regions raster
    from firecomp.core.regions import RegionRaster
    rr = RegionRaster.load()
    valid_ids = rr.valid_ids()
    print(f"Regions: {len(valid_ids)} defined "
          f"({', '.join(rr.names[i] for i in valid_ids)})")

    # ---- Step 1: Sample training data globally ----
    _step(1, f"Sample {n_samples:,} training points "
          f"(uniform across {len(valid_ids)} regions)")
    train = Sampling.sample_global(
        ae_tif, canopy_tif, cop_paths, rr,
        n_samples=n_samples, seed=cfg.seed)

    # ---- Step 2: Train probes ----
    _step(2, "Train discriminative probes")
    projection = Probes.fit_all(train)

    # ---- Step 3: Cluster ----
    _step(3, "Build calibrated embeddings + cluster")
    cal_train = projection.transform(train["ae"])
    print(f"  Calibrated: {cal_train.shape[1]}d "
          f"(cont={projection.n_continuous} + lc={projection.n_lc} "
          f"+ residual={projection.n_residual})")

    kmeans = MiniBatchKMeans(n_clusters=cfg.n_clusters, random_state=42,
                             batch_size=min(10000, len(cal_train)))
    labels_train = kmeans.fit_predict(cal_train)
    print(f"  {cfg.n_clusters} clusters, inertia={kmeans.inertia_:.0f}")

    # ---- Step 4: Evaluate on training samples ----
    _step(4, "Evaluate (training samples)")
    Evaluate.report(labels_train, train, cfg.n_clusters)

    # ---- Step 5: Assign clusters to raster ----
    if cfg.smoke:
        _step(5, f"Assign clusters to eval region: {_SMOKE_EVAL_REGION['label']}")
        raster, raster_gt = RasterAssign.assign_region(
            ae_tif, projection, kmeans, _SMOKE_EVAL_REGION)
    else:
        _step(5, "Assign clusters to global raster (all valid regions)")
        raster, raster_gt = RasterAssign.assign_global(
            ae_tif, projection, kmeans, rr)

    raster_path = os.path.join(
        out_dir, f"{prefix}cluster_raster_{cfg.n_clusters}_{cfg.year}.tif")
    RasterAssign.save_raster(raster, raster_gt, raster_path)
    print(f"  Saved: {raster_path}")

    # ---- Step 6: Fire overlay -> per-cluster stats ----
    fire_stats_dict, lc_stats_dict = None, None
    if not cfg.skip_fire_overlay:
        _step(6, "Fire overlay -> per-cluster stats")
        lc_stats_dict = FireOverlay.compute_lc_stats(
            labels_train, train, cfg.n_clusters)
        fire_stats_dict = FireOverlay.compute_fire_stats(
            raster_path, cfg.n_clusters, cfg.year)
        all_stats = {**lc_stats_dict, **fire_stats_dict,
                     "n_clusters": np.int32(cfg.n_clusters),
                     "year": np.int32(cfg.year)}
        stats_path = os.path.join(
            out_dir, f"{prefix}cluster_stats_{cfg.n_clusters}_{cfg.year}.npz")
        np.savez_compressed(stats_path, **all_stats)
        print(f"  Saved: {stats_path}")
        _print_cluster_table(fire_stats_dict, lc_stats_dict, cfg.n_clusters)
    else:
        _step(6, "Fire overlay SKIPPED")

    # ---- Step 7: Tile metadata ----
    if not cfg.skip_tile_metadata:
        _step(7, "Tile metadata -> per-tile cluster fractions")
        tile_meta = TileMetadata.compute(raster_path, cfg.n_clusters)
        tile_path = os.path.join(
            out_dir, f"{prefix}tile_metadata_{cfg.n_clusters}_{cfg.year}.npz")
        np.savez_compressed(tile_path, **tile_meta)
        print(f"  Saved: {tile_path} ({len(tile_meta['tile_index']):,} tiles)")
    else:
        _step(7, "Tile metadata SKIPPED")

    # ---- Step 8: Save calibrated model ----
    _step(8, "Save model")
    model_path = os.path.join(
        out_dir, f"{prefix}calibrated_model_{cfg.n_clusters}_{cfg.year}.npz")
    projection.save(model_path, kmeans)
    print(f"  Saved: {model_path}")

    # ---- Step 9: Plots ----
    if not cfg.skip_plots:
        _step(9, "Save plots")
        Evaluate.plot_composition(labels_train, train["lc_class"],
                                  cfg.n_clusters, out_dir, cfg.year,
                                  prefix + "train")
        Evaluate.plot_cluster_maps(raster, raster_gt, cfg.n_clusters,
                                   out_dir, cfg.year, prefix,
                                   fire_stats=fire_stats_dict,
                                   lc_stats=lc_stats_dict)
    else:
        _step(9, "Plots SKIPPED")

    elapsed = time.time() - t0_global
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def _step(n, msg):
    print(f"\n{'=' * 70}\nSTEP {n}: {msg}\n{'=' * 70}")


def _print_cluster_table(fire_stats, lc_stats, n_clusters):
    """Print a compact summary table of per-cluster stats."""
    has_lc = (lc_stats is not None
              and "dominant_lc_class" in lc_stats
              and "lc_purity_pct" in lc_stats)
    print(f"\n  {'C':>3} {'Dom LC':<16} {'Pur%':>5} "
          f"{'Wild':>6} {'Tame':>6} {'T:W':>6} {'Dens':>7} "
          f"{'>=100':>5} {'>=1K':>4}")
    print(f"  {'-' * 80}")
    for c in range(n_clusters):
        lc_name = str(lc_stats['dominant_lc_class'][c]) if has_lc else "?"
        lc_pur = float(lc_stats['lc_purity_pct'][c]) if has_lc else 0.0
        print(f"  {c:3d} {lc_name:<16} "
              f"{lc_pur:5.1f} "
              f"{fire_stats['n_wildfires'][c]:6d} "
              f"{fire_stats['n_tame'][c]:6d} "
              f"{fire_stats['tame_to_wild_ratio'][c]:6.2f} "
              f"{fire_stats['fire_density'][c]:7.1f} "
              f"{fire_stats['n_fires_above_100px'][c]:5d} "
              f"{fire_stats['n_fires_above_1000px'][c]:4d}")


# ============================================================================
# Copernicus reprojection (one-time, cached)
# ============================================================================

def _ensure_copernicus_on_ae_grid(ae_tif, cop_dir):
    """Reproject Copernicus layers onto the AE grid (same resolution/extent).

    Creates copernicus_lc100_ae_grid/<name>.tif for each layer. Skips existing.
    Returns dict {feature_name: path_to_reprojected_tif}.
    """
    from osgeo import gdal

    out_dir = os.path.join(os.path.dirname(cop_dir), "copernicus_lc100_ae_grid")
    os.makedirs(out_dir, exist_ok=True)

    ae_ds = gdal.Open(ae_tif)
    ae_gt = ae_ds.GetGeoTransform()
    ae_W, ae_H = ae_ds.RasterXSize, ae_ds.RasterYSize
    ae_srs = ae_ds.GetProjection()
    ae_ds = None

    paths = {}
    for feat_name, src_fname, resample in _COPERNICUS_LAYERS:
        src_path = os.path.join(cop_dir, src_fname)
        dst_path = os.path.join(out_dir, f"{feat_name}.tif")
        paths[feat_name] = dst_path

        if os.path.exists(dst_path):
            continue

        print(f"  Reprojecting {src_fname} -> {feat_name}.tif ({resample})...")
        t0 = time.time()
        warp_opts = gdal.WarpOptions(
            format="GTiff",
            outputBounds=(ae_gt[0], ae_gt[3] + ae_H * ae_gt[5],
                          ae_gt[0] + ae_W * ae_gt[1], ae_gt[3]),
            width=ae_W, height=ae_H,
            resampleAlg=resample,
            dstSRS=ae_srs,
            creationOptions=["COMPRESS=ZSTD", "ZSTD_LEVEL=3",
                             "TILED=YES", "BLOCKXSIZE=128", "BLOCKYSIZE=128"],
        )
        gdal.Warp(dst_path, src_path, options=warp_opts)
        print(f"    Done in {time.time() - t0:.0f}s "
              f"({os.path.getsize(dst_path) / 1e9:.2f} GB)")

    return paths


# ============================================================================
# Sampling — globally uniform training-point extraction
# ============================================================================

class Sampling:
    """Globally uniform sampling from co-registered rasters.

    Samples are allocated proportionally to each region's pixel count.
    Uses tile-based reads (128 x 128) matching the on-disk tile layout
    for efficient I/O — each unique tile is read at most once.
    """

    TILE = 128

    @staticmethod
    def sample_global(ae_tif, canopy_tif, cop_paths, region_raster,
                      n_samples=150_000, seed=42):
        """Sample training points across all valid regions.

        Args:
            ae_tif: path to AE embeddings GeoTIFF.
            canopy_tif: path to canopy height GeoTIFF.
            cop_paths: {feature_name: tif_path} from _ensure_copernicus_on_ae_grid.
            region_raster: RegionRaster instance.
            n_samples: total points to sample.
            seed: random seed.

        Returns:
            dict with keys: ae, height, lc_class, lon, lat, region_id,
            + each fraction name.
        """
        from osgeo import gdal
        from firecomp.core.regions import RASTER_XMIN, RASTER_YMAX, RASTER_PIX

        rng = np.random.RandomState(seed)
        regions = region_raster.raster
        region_names = region_raster.names
        valid_ids = region_raster.valid_ids()
        TILE = Sampling.TILE

        ae_ds = gdal.Open(ae_tif)
        ae_gt = ae_ds.GetGeoTransform()
        ae_W, ae_H = ae_ds.RasterXSize, ae_ds.RasterYSize
        ch_ds = gdal.Open(canopy_tif)
        lc_ds = gdal.Open(cop_paths["lc_class"])
        frac_ds = {name: gdal.Open(cop_paths[name]) for name in _FRACTION_NAMES}

        # Pre-index region pixels, compute proportional targets
        print("  Building per-region pixel indices...")
        region_pixels = {}
        for rid in valid_ids:
            ys, xs = np.where(regions == rid)
            region_pixels[rid] = (ys, xs)
            print(f"    {region_names[rid]:20s}: {len(ys):>8,} pixels (0.1 deg)")

        total_rpx = sum(len(ys) for ys, _ in region_pixels.values())
        targets = {rid: max(1, int(n_samples * len(region_pixels[rid][0]) / total_rpx))
                   for rid in valid_ids}
        print(f"  Total region pixels: {total_rpx:,}")

        # Generate candidate AE pixels from all regions
        cand_ax, cand_ay = [], []
        cand_lon, cand_lat, cand_rid = [], [], []

        for rid in valid_ids:
            ys, xs = region_pixels[rid]
            n_pool = len(ys)
            if n_pool == 0:
                continue
            n_draw = min(targets[rid] * 3, n_pool)
            idxs = rng.choice(n_pool, size=n_draw, replace=n_draw > n_pool)

            lons = RASTER_XMIN + (xs[idxs] + 0.5) * RASTER_PIX
            lats = RASTER_YMAX - (ys[idxs] + 0.5) * RASTER_PIX
            ae_pxs = ((lons - ae_gt[0]) / ae_gt[1]).astype(np.int32)
            ae_pys = ((lats - ae_gt[3]) / ae_gt[5]).astype(np.int32)

            ok = (ae_pxs >= 0) & (ae_pxs < ae_W) & (ae_pys >= 0) & (ae_pys < ae_H)
            vi = np.where(ok)[0]
            cand_ax.extend(ae_pxs[vi].tolist())
            cand_ay.extend(ae_pys[vi].tolist())
            cand_lon.extend(lons[vi].tolist())
            cand_lat.extend(lats[vi].tolist())
            cand_rid.extend([rid] * len(vi))

        cand_ax = np.array(cand_ax, dtype=np.int32)
        cand_ay = np.array(cand_ay, dtype=np.int32)
        cand_lon = np.array(cand_lon, dtype=np.float32)
        cand_lat = np.array(cand_lat, dtype=np.float32)
        cand_rid = np.array(cand_rid, dtype=np.uint8)

        # Group by tile (128 x 128), sorted by row for sequential I/O
        tile_groups: dict[tuple, list[int]] = {}
        for i in range(len(cand_ax)):
            key = (int(cand_ay[i] // TILE), int(cand_ax[i] // TILE))
            tile_groups.setdefault(key, []).append(i)
        print(f"  {len(cand_ax):,} candidates in {len(tile_groups):,} tiles")

        # Read tiles and extract samples
        out: dict[str, list] = {
            "ae": [], "height": [], "lc_class": [],
            "lon": [], "lat": [], "region_id": [],
        }
        for name in _FRACTION_NAMES:
            out[name] = []

        collected = {rid: 0 for rid in valid_ids}
        t0 = time.time()

        for ty, tx in sorted(tile_groups.keys()):
            x0, y0 = tx * TILE, ty * TILE
            w = min(TILE, ae_W - x0)
            h = min(TILE, ae_H - y0)
            if w <= 0 or h <= 0:
                continue

            ae_tile = ae_ds.ReadAsArray(x0, y0, w, h)
            if ae_tile is None:
                continue
            ch_tile = ch_ds.GetRasterBand(1).ReadAsArray(x0, y0, w, h)
            lc_tile = lc_ds.GetRasterBand(1).ReadAsArray(x0, y0, w, h)
            fr_tiles = {name: frac_ds[name].GetRasterBand(1).ReadAsArray(x0, y0, w, h)
                        for name in _FRACTION_NAMES}

            for i in tile_groups[(ty, tx)]:
                rid = int(cand_rid[i])
                if collected[rid] >= targets[rid]:
                    continue

                lx = int(cand_ax[i]) - x0
                ly = int(cand_ay[i]) - y0
                if lx < 0 or lx >= w or ly < 0 or ly >= h:
                    continue

                ae_vec = ae_tile[:, ly, lx]
                if np.all(ae_vec == 0):
                    continue

                lc_val = int(lc_tile[ly, lx])
                if lc_val in _LC_SKIP:
                    continue
                lc_label = _LC_LABELS.get(lc_val)
                if lc_label is None:
                    continue

                out["ae"].append(ae_vec)
                out["height"].append(int(ch_tile[ly, lx]) if ch_tile is not None else 0)
                out["lc_class"].append(lc_label)
                out["lon"].append(float(cand_lon[i]))
                out["lat"].append(float(cand_lat[i]))
                out["region_id"].append(rid)
                for name in _FRACTION_NAMES:
                    v = int(fr_tiles[name][ly, lx])
                    out[name].append(v if v <= 100 else 0)
                collected[rid] += 1

        for rid in valid_ids:
            print(f"    {region_names[rid]:20s}: {collected[rid]:>6,} / {targets[rid]:,}")

        total = len(out["ae"])
        print(f"\n  Total: {total:,} samples from {len(valid_ids)} regions "
              f"in {time.time() - t0:.0f}s")

        # Close GDAL datasets
        ae_ds = ch_ds = lc_ds = None
        frac_ds = None

        result = {
            "ae": np.array(out["ae"], dtype=np.uint8),
            "height": np.array(out["height"], dtype=np.uint8),
            "lc_class": np.array(out["lc_class"]),
            "lon": np.array(out["lon"], dtype=np.float32),
            "lat": np.array(out["lat"], dtype=np.float32),
            "region_id": np.array(out["region_id"], dtype=np.uint8),
        }
        for name in _FRACTION_NAMES:
            result[name] = np.array(out[name], dtype=np.uint8)
        return result


# ============================================================================
# CalibratedProjection — learned AE -> semantic space
# ============================================================================

class CalibratedProjection:
    """Learned projection: raw AE uint8 -> calibrated semantic space."""

    def __init__(self, scaler, cont_W, cont_b, cont_names,
                 lc_W, lc_b, lc_labels,
                 residual_pca, residual_mean, cal_scaler):
        self.scaler = scaler
        self.cont_W, self.cont_b, self.cont_names = cont_W, cont_b, cont_names
        self.lc_W, self.lc_b, self.lc_labels = lc_W, lc_b, lc_labels
        self.residual_pca, self.residual_mean = residual_pca, residual_mean
        self.cal_scaler = cal_scaler
        self.n_continuous = cont_W.shape[0]
        self.n_lc = lc_W.shape[0]
        self.n_residual = residual_pca.n_components_

    def transform(self, ae_uint8):
        """Project raw AE uint8 array to calibrated space."""
        ae_s = self.scaler.transform(ae_uint8.astype(np.float32) / 255.0)
        cont = ae_s @ self.cont_W.T + self.cont_b
        lc = ae_s @ self.lc_W.T + self.lc_b
        used = np.vstack([self.cont_W, self.lc_W])
        Q, _ = np.linalg.qr(used.T)
        res = ae_s - ae_s @ Q @ Q.T - self.residual_mean
        r = self.residual_pca.transform(res)
        return self.cal_scaler.transform(np.hstack([cont, lc, r]))

    def save(self, path, kmeans):
        """Save projection model + KMeans centres to npz."""
        np.savez(path,
                 scaler_mean=self.scaler.mean_,
                 scaler_scale=self.scaler.scale_,
                 cont_W=self.cont_W, cont_b=self.cont_b,
                 cont_names=np.array(self.cont_names),
                 lc_W=self.lc_W, lc_b=self.lc_b,
                 lc_labels=np.array(self.lc_labels),
                 residual_pca_components=self.residual_pca.components_,
                 residual_pca_variance=self.residual_pca.explained_variance_,
                 residual_mean=self.residual_mean,
                 cal_scaler_mean=self.cal_scaler.mean_,
                 cal_scaler_scale=self.cal_scaler.scale_,
                 kmeans_centers=kmeans.cluster_centers_)


# ============================================================================
# Probes — train linear probes for calibrated embedding
# ============================================================================

class Probes:
    """Train linear probes (Ridge for continuous, LogReg for LC class)."""

    @staticmethod
    def fit_all(train_data, n_residual=8):
        """Fit probes and return a CalibratedProjection."""
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        ae = train_data["ae"].astype(np.float32) / 255.0
        scaler = StandardScaler()
        ae_s = scaler.fit_transform(ae)

        # Continuous probes (Ridge)
        cont_W_list, cont_b_list, cont_names = [], [], []
        for name in _CONTINUOUS_NAMES:
            vals = train_data[name].astype(np.float32)
            mask = ((vals > 0) & (vals < 255)) if name == "height" else np.ones(len(vals), dtype=bool)
            n_valid = mask.sum()
            if n_valid < 200:
                print(f"  {name:>12}: SKIP ({n_valid} valid)")
                continue
            ridge = Ridge(alpha=1.0)
            ridge.fit(ae_s[mask], vals[mask])
            cv_scores = cross_val_score(ridge, ae_s[mask], vals[mask], cv=5, scoring="r2")
            print(f"  {name:>12}: R2={cv_scores.mean():.3f}+/-{cv_scores.std():.3f} "
                  f"(n={n_valid:,})")
            cont_W_list.append(ridge.coef_.astype(np.float32))
            cont_b_list.append(float(ridge.intercept_))
            cont_names.append(name)

        cont_W = np.array(cont_W_list)
        cont_b = np.array(cont_b_list, dtype=np.float32)
        print(f"  => {len(cont_names)} continuous probes")

        # LC class probe (LogReg)
        lc = train_data["lc_class"]
        types, counts = np.unique(lc, return_counts=True)
        keep = {t for t, c in zip(types, counts) if c >= 50}
        keep_mask = np.isin(lc, list(keep))
        lc_kept, ae_lc = lc[keep_mask], ae_s[keep_mask]
        print(f"\n  LC probe: {len(ae_lc):,} samples, {len(keep)} classes")
        for t, c in sorted(zip(*np.unique(lc_kept, return_counts=True)),
                           key=lambda x: -x[1]):
            print(f"    {t:<20} {c:>7,}")

        lr = LogisticRegression(max_iter=500, C=1.0, random_state=42, n_jobs=-1)
        lr.fit(ae_lc, lc_kept)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(lr, ae_lc, lc_kept, cv=cv, scoring="accuracy")
        print(f"  LC accuracy = {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

        lc_W = lr.coef_.astype(np.float32)
        lc_b = lr.intercept_.astype(np.float32)
        lc_labels = list(lr.classes_)

        # Residual PCA
        used = np.vstack([cont_W, lc_W])
        Q, _ = np.linalg.qr(used.T)
        residual = ae_s - ae_s @ Q @ Q.T
        res_mean = residual.mean(axis=0)
        pca = PCA(n_components=n_residual, random_state=42)
        pca.fit(residual - res_mean)
        print(f"\n  Residual PCA: {n_residual}d, "
              f"explained={pca.explained_variance_ratio_.sum() * 100:.1f}%")

        # Calibrated scaler
        cont_scores = ae_s @ cont_W.T + cont_b
        lc_scores = ae_s @ lc_W.T + lc_b
        r_scores = pca.transform(residual - res_mean)
        cal_scaler = StandardScaler()
        cal_scaler.fit(np.hstack([cont_scores, lc_scores, r_scores]))

        return CalibratedProjection(
            scaler, cont_W, cont_b, cont_names,
            lc_W, lc_b, lc_labels, pca, res_mean, cal_scaler)


# ============================================================================
# RasterAssign — assign cluster labels to raster pixels
# ============================================================================

class RasterAssign:
    """Assign cluster labels to raster pixels via strip-based reads."""

    STRIP_H = 512

    @staticmethod
    def assign_region(ae_tif, projection, kmeans, region):
        """Assign clusters within a geographic bounding box. Returns (raster, gt)."""
        from osgeo import gdal

        ae_ds = gdal.Open(ae_tif)
        ae_gt = ae_ds.GetGeoTransform()
        lon0, lon1 = region["lon"]
        lat0, lat1 = region["lat"]

        x0 = max(0, int((lon0 - ae_gt[0]) / ae_gt[1]))
        x1 = min(ae_ds.RasterXSize, int((lon1 - ae_gt[0]) / ae_gt[1]))
        y0 = max(0, int((lat1 - ae_gt[3]) / ae_gt[5]))
        y1 = min(ae_ds.RasterYSize, int((lat0 - ae_gt[3]) / ae_gt[5]))
        w, h = x1 - x0, y1 - y0
        print(f"  Region: {region['label']} [{x0}:{x1}, {y0}:{y1}] ({w}x{h} px)")

        raster = np.full((h, w), 255, dtype=np.uint8)
        gt = (ae_gt[0] + x0 * ae_gt[1], ae_gt[1], 0,
              ae_gt[3] + y0 * ae_gt[5], 0, ae_gt[5])

        t0 = time.time()
        SH = RasterAssign.STRIP_H
        n_strips = (h + SH - 1) // SH
        assigned = 0
        for si in range(n_strips):
            sy0 = si * SH
            sy1 = min(sy0 + SH, h)
            sh = sy1 - sy0

            strip = ae_ds.ReadAsArray(x0, y0 + sy0, w, sh)
            if strip is None:
                continue

            flat = strip.reshape(strip.shape[0], -1).T
            valid = ~np.all(flat == 0, axis=1)
            if valid.sum() == 0:
                continue

            cal = projection.transform(flat[valid])
            labels = kmeans.predict(cal).astype(np.uint8)

            row_labels = np.full(flat.shape[0], 255, dtype=np.uint8)
            row_labels[valid] = labels
            raster[sy0:sy1, :] = row_labels.reshape(sh, w)
            assigned += valid.sum()

            if si % 20 == 0:
                print(f"    Strip {si}/{n_strips}: {assigned:,} assigned "
                      f"({time.time() - t0:.0f}s)")

        ae_ds = None
        print(f"  Assigned {assigned:,} pixels in {time.time() - t0:.0f}s")
        return raster, gt

    @staticmethod
    def assign_global(ae_tif, projection, kmeans, region_raster):
        """Assign clusters globally, skipping pixels outside defined regions."""
        from osgeo import gdal
        from firecomp.core.regions import RASTER_XMIN, RASTER_YMAX, RASTER_PIX

        ae_ds = gdal.Open(ae_tif)
        ae_gt = ae_ds.GetGeoTransform()
        W, H = ae_ds.RasterXSize, ae_ds.RasterYSize
        regions = region_raster.raster
        print(f"  Global raster: {W}x{H}")

        raster = np.full((H, W), 255, dtype=np.uint8)
        gt = ae_gt

        t0 = time.time()
        SH = RasterAssign.STRIP_H
        n_strips = (H + SH - 1) // SH
        assigned = 0
        for si in range(n_strips):
            sy0 = si * SH
            sy1 = min(sy0 + SH, H)
            sh = sy1 - sy0

            # Quick check: does this strip overlap any valid region?
            lat_top = ae_gt[3] + sy0 * ae_gt[5]
            lat_bot = ae_gt[3] + sy1 * ae_gt[5]
            ry0 = max(0, int((RASTER_YMAX - lat_top) / RASTER_PIX))
            ry1 = min(regions.shape[0],
                      int((RASTER_YMAX - lat_bot) / RASTER_PIX) + 1)
            if ry0 >= ry1 or not np.any(regions[ry0:ry1, :] > 0):
                continue

            strip = ae_ds.ReadAsArray(0, sy0, W, sh)
            if strip is None:
                continue

            flat = strip.reshape(strip.shape[0], -1).T
            valid = ~np.all(flat == 0, axis=1)

            # Vectorised region mask
            reg_mask = RasterAssign._region_mask_for_strip(
                ae_gt, regions, sy0, sh, W).ravel()
            valid &= reg_mask

            n_valid = valid.sum()
            if n_valid == 0:
                continue

            cal = projection.transform(flat[valid])
            labels = kmeans.predict(cal).astype(np.uint8)

            row_labels = np.full(flat.shape[0], 255, dtype=np.uint8)
            row_labels[valid] = labels
            raster[sy0:sy1, :] = row_labels.reshape(sh, W)
            assigned += n_valid

            if si % 10 == 0:
                print(f"    Strip {si}/{n_strips}: {assigned:,} assigned "
                      f"({time.time() - t0:.0f}s)")

        ae_ds = None
        print(f"  Assigned {assigned:,} pixels in {time.time() - t0:.0f}s")
        return raster, gt

    @staticmethod
    def _region_mask_for_strip(ae_gt, regions, sy0, sh, W):
        """Vectorised region lookup: returns bool mask (sh, W)."""
        from firecomp.core.regions import RASTER_XMIN, RASTER_YMAX, RASTER_PIX

        rows = np.arange(sh)
        lats = ae_gt[3] + (sy0 + rows) * ae_gt[5]
        lons = ae_gt[0] + np.arange(W) * ae_gt[1]

        reg_ry = np.clip(((RASTER_YMAX - lats) / RASTER_PIX).astype(np.int32),
                         0, regions.shape[0] - 1)
        reg_rx = np.clip(((lons - RASTER_XMIN) / RASTER_PIX).astype(np.int32),
                         0, regions.shape[1] - 1)

        region_vals = regions[reg_ry[:, None], reg_rx[None, :]]
        return region_vals > 0

    @staticmethod
    def save_raster(raster, gt, path):
        """Save cluster raster as compressed GeoTIFF with overviews."""
        from osgeo import gdal, osr

        H, W = raster.shape
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, W, H, 1, gdal.GDT_Byte,
                        options=["COMPRESS=ZSTD", "TILED=YES"])
        ds.SetGeoTransform(gt)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(255)
        band.WriteArray(raster)
        ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
        ds.FlushCache()
        ds = None


# ============================================================================
# Evaluate — cluster quality metrics and plots
# ============================================================================

class Evaluate:
    """Evaluation and visualisation of cluster assignments."""

    @staticmethod
    def report(labels, data, n_clusters):
        """Print NMI/ARI against LC class and per-cluster composition tables."""
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

        lc = data["lc_class"]
        nmi = normalized_mutual_info_score(lc, labels)
        ari = adjusted_rand_score(lc, labels)
        print(f"\n  LC vs cluster: NMI={nmi:.3f}, ARI={ari:.3f}")

        # Per-cluster dominant LC
        print(f"\n  {'C':>4} {'N':>7} {'Dominant':<20} {'Pur%':>6} {'Top 3'}")
        print(f"  {'-' * 75}")
        for c in range(n_clusters):
            mask = labels == c
            n = mask.sum()
            if n == 0:
                continue
            t = lc[mask]
            ut, uc = np.unique(t, return_counts=True)
            order = np.argsort(-uc)
            dom = ut[order[0]]
            pur = uc[order[0]] / n * 100
            top3 = ", ".join(f"{ut[i]}={uc[i] / n * 100:.0f}%" for i in order[:3])
            print(f"  {c:4d} {n:7,} {dom:<20} {pur:5.1f}% {top3}")

        # Per-cluster mean cover fractions
        hdr = "  ".join(f"{n[:6]:>6}" for n in _FRACTION_NAMES)
        print(f"\n  {'C':>4} {'N':>7} {'Ht':>4}  {hdr}")
        print(f"  {'-' * (18 + 8 * len(_FRACTION_NAMES))}")
        height = data["height"]
        for c in range(n_clusters):
            mask = labels == c
            n = mask.sum()
            if n < 10:
                continue
            hm = height[mask]
            h_valid = (hm > 0) & (hm < 255)
            h_str = (f"{hm[h_valid].astype(float).mean():4.1f}"
                     if h_valid.sum() > 5 else "  - ")
            fracs = "  ".join(
                f"{data[name][mask].astype(float).mean():6.1f}"
                for name in _FRACTION_NAMES)
            print(f"  {c:4d} {n:7,} {h_str}  {fracs}")

    @staticmethod
    def plot_composition(labels, lc_class, n_clusters, out_dir, year, tag):
        """Stacked bar chart of LC composition per cluster."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _LC_COLORS = {
            "EG_Needle": "#006400", "EG_Broad": "#228B22",
            "DC_Needle": "#8FBC8F", "DC_Broad": "#32CD32",
            "Mixed_Forest": "#2E8B57", "Other_Forest": "#3CB371",
            "Shrub": "#D2691E", "Grass": "#BDB76B", "Cropland": "#FFD700",
            "Urban": "#808080", "Bare": "#F4A460", "Wetland": "#4682B4",
            "Moss_Lichen": "#9ACD32", "Water": "#1E90FF", "Ocean": "#000080",
        }

        all_types = sorted(set(lc_class))
        type_idx = {t: i for i, t in enumerate(all_types)}
        tab = np.zeros((n_clusters, len(all_types)), dtype=np.int32)
        for l, t in zip(labels, lc_class):
            tab[l, type_idx[t]] += 1

        row_sums = tab.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1
        pct = tab / row_sums * 100

        forest_types = {"EG_Needle", "EG_Broad", "DC_Needle", "DC_Broad",
                        "Mixed_Forest", "Other_Forest"}
        forest_cols = [type_idx[t] for t in all_types if t in forest_types]
        forest_frac = (pct[:, forest_cols].sum(axis=1) if forest_cols
                       else np.zeros(n_clusters))
        sort_order = np.argsort(-forest_frac)

        fig, ax = plt.subplots(figsize=(max(16, n_clusters * 0.5), 8))
        x = np.arange(n_clusters)
        bottom = np.zeros(n_clusters)
        for j, t in enumerate(all_types):
            color = _LC_COLORS.get(t, "#888888")
            vals = pct[sort_order, j]
            ax.bar(x, vals, bottom=bottom, label=t, color=color, width=0.8)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{sort_order[i]}" for i in range(n_clusters)],
                           fontsize=6, rotation=90)
        ax.set_ylabel("% of cluster")
        ax.set_title(f"Calibrated Cluster Composition -- {tag} ({year})")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        path = os.path.join(out_dir, f"composition_{tag}_{year}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    @staticmethod
    def plot_cluster_maps(raster, gt, n_clusters, out_dir, year, prefix,
                          fire_stats=None, lc_stats=None):
        """Per-cluster map panels with fixed extent and fire annotations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import cartopy.crs as ccrs
        from scipy.ndimage import zoom

        GRID_COLS = 4
        grid_rows = math.ceil(n_clusters / GRID_COLS)
        cmap = plt.colormaps["nipy_spectral"]

        # Downsample for display
        target_w = min(raster.shape[1], 3000)
        scale = target_w / raster.shape[1]
        ds_h = max(1, int(raster.shape[0] * scale))
        ds_raster = zoom(raster, (ds_h / raster.shape[0], target_w / raster.shape[1]),
                         order=0)

        lon0 = gt[0]
        lon1 = gt[0] + raster.shape[1] * gt[1]
        lat0 = gt[3] + raster.shape[0] * gt[5]
        lat1 = gt[3]
        extent = [lon0, lon1, lat0, lat1]

        cluster_counts = np.array([np.sum(ds_raster == c) for c in range(n_clusters)])
        sort_order = np.argsort(-cluster_counts)

        fig, axes = plt.subplots(grid_rows, GRID_COLS,
                                 figsize=(GRID_COLS * 6, grid_rows * 4),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        axes = axes.flatten()

        for idx in range(n_clusters):
            c = sort_order[idx]
            ax = axes[idx]
            ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
            ax.stock_img()
            ax.coastlines(linewidth=0.3, color="#333333")

            mask = np.where(ds_raster == c, 1.0, np.nan)
            color = cmap(c / max(n_clusters - 1, 1))
            cmap_single = mcolors.ListedColormap([color])
            ax.imshow(mask, extent=extent, transform=ccrs.PlateCarree(),
                      origin="upper", cmap=cmap_single,
                      vmin=0.5, vmax=1.5, interpolation="nearest",
                      zorder=1, alpha=0.8)

            n_px = cluster_counts[c]
            dom_lc = lc_stats["dominant_lc_class"][c] if lc_stats else ""
            ax.set_title(f"C{c}  {dom_lc}  ({n_px:,} px)",
                         fontsize=7, fontweight="bold", pad=2)

            if fire_stats is not None:
                n100 = fire_stats["n_fires_above_100px"][c]
                n1k = fire_stats["n_fires_above_1000px"][c]
                tw = fire_stats["tame_to_wild_ratio"][c]
                sprd = fire_stats.get("mean_spread_large",
                                      np.zeros(n_clusters))[c]
                sprd_str = f"  sprd={sprd:.1f}" if sprd > 0 else ""
                ann = f">=100: {n100}  >=1K: {n1k}  T:W={tw:.2f}{sprd_str}"
                ax.text(0.02, 0.02, ann, transform=ax.transAxes,
                        fontsize=6, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.6, pad=1.5,
                                  edgecolor="none"),
                        va="bottom", ha="left", zorder=5)

        for idx in range(n_clusters, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"Cluster Maps -- {prefix.rstrip('_')} ({year})",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{prefix}cluster_maps_{year}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ============================================================================
# FireOverlay — per-cluster fire + LC statistics
# ============================================================================

class FireOverlay:
    """Compute per-cluster fire + LC statistics for cluster_stats npz."""

    @staticmethod
    def compute_lc_stats(labels, train_data, n_clusters):
        """Per-cluster LC statistics from training samples.

        Returns dict of arrays, each shape (n_clusters,).
        """
        dominant_lc = np.array([""] * n_clusters, dtype="U20")
        lc_purity_pct = np.zeros(n_clusters, dtype=np.float32)
        mean_height = np.zeros(n_clusters, dtype=np.float32)
        frac_means = {name: np.zeros(n_clusters, dtype=np.float32)
                      for name in _FRACTION_NAMES}

        lc = train_data["lc_class"]
        height = train_data["height"]

        for c in range(n_clusters):
            mask = labels == c
            n = mask.sum()
            if n == 0:
                continue

            types, counts = np.unique(lc[mask], return_counts=True)
            best = np.argmax(counts)
            dominant_lc[c] = types[best]
            lc_purity_pct[c] = counts[best] / n * 100

            hm = height[mask]
            h_valid = (hm > 0) & (hm < 255)
            if h_valid.sum() > 0:
                mean_height[c] = hm[h_valid].astype(np.float32).mean()

            for name in _FRACTION_NAMES:
                frac_means[name][c] = train_data[name][mask].astype(np.float32).mean()

        result = {
            "dominant_lc_class": dominant_lc,
            "lc_purity_pct": lc_purity_pct,
            "mean_canopy_height": mean_height,
        }
        for name in _FRACTION_NAMES:
            result[f"mean_{name}"] = frac_means[name]
        return result

    @staticmethod
    def compute_fire_stats(cluster_raster_path, n_clusters, year):
        """Centroid-based fire overlay: map fires -> clusters, aggregate stats.

        Uses FireClassification from core/fire_stats.py for tame/wild
        classification.  Fires are weighted by sqrt(num_fire) so that
        large wildfires count more than tiny crop burns.

        All fires in the VNP14 database are used (not just *year*) so that
        burnability ratios have maximum statistical evidence.  The *year*
        argument is stored in the output for bookkeeping only.

        Returns dict of arrays.  Per-cluster arrays have shape (n_clusters,).
        Per-region-per-cluster arrays (suffix ``_by_region``) have shape
        (max_region_id + 1, n_clusters) so they can be indexed directly by
        region ID.
        """
        import h5py
        from firecomp.dsrc.vnp14 import Fires
        from firecomp.core.fire_stats import FireClassification
        from firecomp.core.regions import RegionRaster

        with h5py.File(config.vnp14_path, "r") as f:
            stats = Fires.load(f, "stats")

        fire_years = stats.start_date.astype("datetime64[Y]").astype(int) + 1970
        unique_years = np.unique(fire_years)
        print(f"  All fires: {len(stats.id):,} "
              f"(years {int(unique_years.min())}–{int(unique_years.max())})")

        # Classify fires using shared FireClassification
        is_tame = FireClassification.identify_tame(stats)
        # For overlay stats we use a relaxed wild definition (no area threshold)
        # — we're counting fires for statistics, not filtering for training
        is_wild = FireClassification.identify_wild(
            stats, is_tame, areas_km2=np.ones(len(stats.id)), min_area_km2=0.0)
        print(f"    wild={is_wild.sum():,}, tame={is_tame.sum():,}, "
              f"unknown={(~is_wild & ~is_tame).sum():,}")

        # Map centroids to clusters
        fire_clusters = FireOverlay._lookup_fire_clusters(
            cluster_raster_path, stats)

        # Map centroids to regions
        rr = RegionRaster.load()
        fire_regions = rr.lookup_fires(stats)
        max_rid = int(fire_regions.max())
        n_regions = max_rid + 1
        print(f"  Regions: max_id={max_rid}, "
              f"{int((fire_regions > 0).sum()):,} fires in valid regions")

        # Count cluster pixels
        cluster_px = FireOverlay._count_cluster_pixels(
            cluster_raster_path, n_clusters)

        # Per-fire weights: sqrt(num_fire) — large fires count more
        weights = np.sqrt(stats.num_fire.astype(np.float64))

        # Aggregate per cluster (global)
        n_wildfires = np.zeros(n_clusters, dtype=np.int32)
        n_tame = np.zeros(n_clusters, dtype=np.int32)
        tame_to_wild_ratio = np.zeros(n_clusters, dtype=np.float32)
        fire_density = np.zeros(n_clusters, dtype=np.float32)
        n_above_100px = np.zeros(n_clusters, dtype=np.int32)
        n_above_1000px = np.zeros(n_clusters, dtype=np.int32)

        # Per region × cluster
        wild_weight_rc = np.zeros((n_regions, n_clusters), dtype=np.float64)
        tame_weight_rc = np.zeros((n_regions, n_clusters), dtype=np.float64)

        for c in range(n_clusters):
            in_c = fire_clusters == c
            wild_in = in_c & is_wild
            tame_in = in_c & is_tame

            nw = int(wild_in.sum())
            nt = int(tame_in.sum())
            n_wildfires[c] = nw
            n_tame[c] = nt
            fire_density[c] = nw / max(int(cluster_px[c]), 1) * 1e6

            if nw > 0:
                wild_nf = stats.num_fire[wild_in]
                n_above_100px[c] = int((wild_nf >= 100).sum())
                n_above_1000px[c] = int((wild_nf >= 1000).sum())

            # Global sqrt-weighted ratio
            w_wild = float(weights[wild_in].sum()) if nw > 0 else 0.0
            w_tame = float(weights[tame_in].sum()) if nt > 0 else 0.0
            tame_to_wild_ratio[c] = w_tame / max(w_wild, 1.0)

            # Per-region sqrt-weighted accumulation
            for rid in range(1, n_regions):
                r_wild = wild_in & (fire_regions == rid)
                r_tame = tame_in & (fire_regions == rid)
                wild_weight_rc[rid, c] = float(weights[r_wild].sum())
                tame_weight_rc[rid, c] = float(weights[r_tame].sum())

        # Per-region ratio: tame_weight / max(wild_weight, 1)
        # NaN when total evidence (wild + tame) is below threshold,
        # so the figure code falls back to the global ratio.
        MIN_EVIDENCE_WEIGHT = 20.0   # ~sqrt of 400 detections
        tame_to_wild_by_region = np.full((n_regions, n_clusters),
                                         np.nan, dtype=np.float32)
        n_sufficient = 0
        for rid in range(1, n_regions):
            for c in range(n_clusters):
                ww = wild_weight_rc[rid, c]
                wt = tame_weight_rc[rid, c]
                if ww + wt >= MIN_EVIDENCE_WEIGHT:
                    tame_to_wild_by_region[rid, c] = wt / max(ww, 1.0)
                    n_sufficient += 1
        n_total = (n_regions - 1) * n_clusters
        print(f"  Per-region ratios: {n_sufficient}/{n_total} pairs above "
              f"min evidence weight ({MIN_EVIDENCE_WEIGHT}), "
              f"{n_total - n_sufficient} will fall back to global")

        # Log per-region summary
        for rid in range(1, n_regions):
            name = rr.names.get(rid, f"R{rid}")
            total_w = wild_weight_rc[rid].sum()
            total_t = tame_weight_rc[rid].sum()
            n_ok = int(np.isfinite(tame_to_wild_by_region[rid]).sum())
            print(f"    {name:20s}: wild_w={total_w:8.0f}  "
                  f"tame_w={total_t:8.0f}  "
                  f"ratio={total_t / max(total_w, 1.0):.2f}  "
                  f"({n_ok}/{n_clusters} clusters with data)")

        result = {
            "n_wildfires": n_wildfires,
            "n_tame": n_tame,
            "tame_to_wild_ratio": tame_to_wild_ratio,
            "tame_to_wild_ratio_by_region": tame_to_wild_by_region,
            "fire_density": fire_density,
            "n_fires_above_100px": n_above_100px,
            "n_fires_above_1000px": n_above_1000px,
            "cluster_pixel_counts": cluster_px,
        }

        # Spread stats (optional, from cached file)
        spread_path = os.path.join(config.data_dir, "fire_spread_stats.npz")
        if os.path.exists(spread_path):
            result.update(FireOverlay._compute_spread_stats(
                spread_path, stats, fire_clusters, is_wild, n_clusters))
        else:
            print(f"  WARNING: {spread_path} not found -- spread stats will be zero")
            for key in ("mean_spread_all", "mean_spread_large",
                        "spread_median", "spread_p80", "spread_max"):
                result[key] = np.zeros(n_clusters, dtype=np.float32)

        return result

    @staticmethod
    def _lookup_fire_clusters(cluster_raster_path, stats):
        """Map fire centroids -> cluster IDs via chunk-read of raster."""
        from osgeo import gdal

        ds = gdal.Open(cluster_raster_path)
        gt = ds.GetGeoTransform()
        H, W = ds.RasterYSize, ds.RasterXSize
        band = ds.GetRasterBand(1)

        cx = (stats.min_x + stats.max_x) / 2.0
        cy = (stats.min_y + stats.max_y) / 2.0
        px = ((cx - gt[0]) / gt[1]).astype(np.int32)
        py = ((cy - gt[3]) / gt[5]).astype(np.int32)

        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        fire_clusters = np.full(len(cx), 255, dtype=np.uint8)
        valid_idx = np.where(valid)[0]
        valid_py, valid_px = py[valid_idx], px[valid_idx]

        CHUNK = 4096
        for y0 in range(0, H, CHUNK):
            y1 = min(y0 + CHUNK, H)
            in_chunk = (valid_py >= y0) & (valid_py < y1)
            if not in_chunk.any():
                continue
            chunk = band.ReadAsArray(0, y0, W, y1 - y0)
            rows_local = valid_py[in_chunk] - y0
            cols = valid_px[in_chunk]
            fire_clusters[valid_idx[in_chunk]] = chunk[rows_local, cols]

        mapped = int(np.sum(fire_clusters != 255))
        print(f"  Mapped {mapped:,} / {len(cx):,} fires to clusters")
        ds = None
        return fire_clusters

    @staticmethod
    def _count_cluster_pixels(cluster_raster_path, n_clusters):
        """Count pixels per cluster in the raster."""
        from osgeo import gdal

        ds = gdal.Open(cluster_raster_path)
        H, W = ds.RasterYSize, ds.RasterXSize
        band = ds.GetRasterBand(1)
        counts = np.zeros(n_clusters, dtype=np.int64)
        CHUNK = 4096
        for y0 in range(0, H, CHUNK):
            y1 = min(y0 + CHUNK, H)
            chunk = band.ReadAsArray(0, y0, W, y1 - y0).ravel()
            valid = chunk[chunk < n_clusters]
            if len(valid) > 0:
                counts += np.bincount(valid, minlength=n_clusters)
        ds = None
        return counts

    @staticmethod
    def _compute_spread_stats(spread_path, stats, fire_clusters, is_wild,
                              n_clusters):
        """Aggregate spread rates per cluster from cached fire_spread_stats.npz."""
        ss = np.load(spread_path)
        sp_median = ss["spread_median"]
        sp_mean = ss["spread_mean"]
        sp_max = ss["spread_max"]
        sp_p80 = ss["spread_p80"]
        max_id = len(sp_median) - 1

        mean_spread_all = np.zeros(n_clusters, dtype=np.float32)
        mean_spread_large = np.zeros(n_clusters, dtype=np.float32)
        out_spread_median = np.zeros(n_clusters, dtype=np.float32)
        out_spread_p80 = np.zeros(n_clusters, dtype=np.float32)
        out_spread_max = np.zeros(n_clusters, dtype=np.float32)

        for c in range(n_clusters):
            mask_all = ((fire_clusters == c) & is_wild
                        & (stats.num_fire >= 100) & (stats.id <= max_id))
            fids = stats.id[mask_all]
            if len(fids) > 0:
                vals = sp_median[fids]
                has_spread = vals > 0
                if has_spread.sum() > 0:
                    fids_s = fids[has_spread]
                    mean_spread_all[c] = sp_mean[fids_s].mean()
                    out_spread_median[c] = sp_median[fids_s].mean()
                    out_spread_p80[c] = sp_p80[fids_s].mean()
                    out_spread_max[c] = sp_max[fids_s].max()

            mask_large = ((fire_clusters == c) & is_wild
                          & (stats.num_fire >= 1000) & (stats.id <= max_id))
            fids_l = stats.id[mask_large]
            if len(fids_l) > 0:
                vals_l = sp_median[fids_l]
                has_l = vals_l > 0
                if has_l.sum() > 0:
                    mean_spread_large[c] = sp_mean[fids_l[has_l]].mean()

        return {
            "mean_spread_all": mean_spread_all,
            "mean_spread_large": mean_spread_large,
            "spread_median": out_spread_median,
            "spread_p80": out_spread_p80,
            "spread_max": out_spread_max,
        }


# ============================================================================
# TileMetadata — per-tile cluster fractions
# ============================================================================

class TileMetadata:
    """Compute per-tile cluster fractions from cluster raster."""

    TILE_SIZE = 256

    @staticmethod
    def compute(cluster_raster_path, n_clusters):
        """Iterate 256x256 tiles, compute cluster fractions per tile.

        Returns dict for tile_metadata_K_Y.npz.
        """
        from osgeo import gdal

        ds = gdal.Open(cluster_raster_path)
        gt = ds.GetGeoTransform()
        W, H = ds.RasterXSize, ds.RasterYSize
        band = ds.GetRasterBand(1)
        T = TileMetadata.TILE_SIZE

        num_x = math.ceil(W / T)
        num_y = math.ceil(H / T)
        print(f"  Tile grid: {num_x}x{num_y} = {num_x * num_y:,} tiles ({T}px)")

        tile_indices = []
        center_lons = []
        center_lats = []
        cluster_fracs = []

        t0 = time.time()
        n_valid = 0

        for ty in range(num_y):
            y0 = ty * T
            y1 = min(y0 + T, H)
            sh = y1 - y0

            strip = band.ReadAsArray(0, y0, W, sh)

            for tx in range(num_x):
                x0 = tx * T
                x1 = min(x0 + T, W)

                patch = strip[:sh, x0:x1]
                valid = patch[patch != 255]
                if len(valid) == 0:
                    continue

                counts = np.bincount(valid, minlength=n_clusters).astype(np.float32)
                fracs = counts / len(valid)

                tile_idx = ty * num_x + tx
                cx_px = (x0 + x1) / 2.0
                cy_px = (y0 + y1) / 2.0
                c_lon = gt[0] + cx_px * gt[1]
                c_lat = gt[3] + cy_px * gt[5]

                tile_indices.append(tile_idx)
                center_lons.append(c_lon)
                center_lats.append(c_lat)
                cluster_fracs.append(fracs[:n_clusters])
                n_valid += 1

            if ty % 50 == 0:
                print(f"    Row {ty}/{num_y}: {n_valid:,} valid tiles "
                      f"({time.time() - t0:.0f}s)")

        ds = None
        print(f"  {n_valid:,} valid tiles in {time.time() - t0:.0f}s")

        return {
            "tile_index": np.array(tile_indices, dtype=np.int32),
            "center_lon": np.array(center_lons, dtype=np.float32),
            "center_lat": np.array(center_lats, dtype=np.float32),
            "cluster_fractions": np.array(cluster_fracs, dtype=np.float32),
            "num_x": np.int32(num_x),
        }


# ============================================================================
# recompute_fire_stats — rerun step 6 from existing raster
# ============================================================================

def recompute_fire_stats(year: int = 2020, n_clusters: int = 32):
    """Re-run only FireOverlay.compute_fire_stats on an existing cluster raster.

    Loads the existing stats NPZ, replaces all fire-related arrays with
    freshly computed ones (preserving LC stats), and saves back.

    Usage::

        python -m firecomp.core.clusters --recompute-fire-stats --year 2020
    """
    base = os.path.join(config.data_dir, "cluster_copernicus")
    raster_path = os.path.join(base, f"cluster_raster_{n_clusters}_{year}.tif")
    stats_path = os.path.join(base, f"cluster_stats_{n_clusters}_{year}.npz")

    if not os.path.exists(raster_path):
        raise FileNotFoundError(
            f"Cluster raster not found: {raster_path}\n"
            f"Run the full pipeline first: python -m firecomp.core.clusters --year {year}")

    # Load existing stats (preserve LC keys)
    old_stats = {}
    if os.path.exists(stats_path):
        data = np.load(stats_path, allow_pickle=True)
        old_stats = {k: data[k] for k in data.files}
        print(f"Loaded existing stats: {stats_path}")
        print(f"  Keys: {', '.join(sorted(old_stats.keys()))}")

    # Recompute fire stats
    print(f"\nRecomputing fire stats for year={year}, n_clusters={n_clusters}")
    fire_stats = FireOverlay.compute_fire_stats(raster_path, n_clusters, year)
    _print_cluster_table(fire_stats, old_stats, n_clusters)

    # Merge: fire stats overwrite, LC stats preserved
    merged = {k: v for k, v in old_stats.items() if k not in fire_stats}
    merged.update(fire_stats)
    merged["n_clusters"] = np.int32(n_clusters)
    merged["year"] = np.int32(year)

    np.savez_compressed(stats_path, **merged)
    print(f"\nSaved: {stats_path}")
    print(f"  Keys: {', '.join(sorted(merged.keys()))}")


# ============================================================================

if __name__ == "__main__":
    import sys
    if "--recompute-fire-stats" in sys.argv:
        # Quick parse: extract --year and --n-clusters
        _argv = sys.argv[1:]
        _year = 2020
        _nc = 32
        for i, a in enumerate(_argv):
            if a == "--year" and i + 1 < len(_argv):
                _year = int(_argv[i + 1])
            elif a == "--n-clusters" and i + 1 < len(_argv):
                _nc = int(_argv[i + 1])
        recompute_fire_stats(year=_year, n_clusters=_nc)
    else:
        main()
