"""
core/fire_filter.py — Per-region, LC-based fire type classifier.

Replaces the old global-threshold wild/tame classification
(fire_stats.py identify_wild/identify_tame) which was biased toward
North American boreal wildfires and incorrectly excluded ~90% of African
and S. Asian fires.

Three fire types:
  - VEGETATION: real fire on burnable land (natural or human-initiated).
                Includes savanna burns, pastoral fires, deforestation fires.
  - STATIC:     persistent industrial heat — oil flares, gas burn-off,
                offshore platforms.  High t_ratio, low xy.
  - CROP:       agricultural residue burning on cropland.  Region-dependent:
                keep in Africa/S.America (real vegetation fire), exclude in
                C.NA / MENA (prescribed burns / industrial agriculture).

Design informed by GFED fire type decomposition (van der Werf et al. 2010)
and per-(region, LC) fire property diagnostics.

Usage::

    from firecomp.core.fire_filter import classify_fires, FireType, REGION_FILTERS

    # Vectorised classification
    types = classify_fires(region_ids, lc_classes, t_ratios)
    veg_mask = types == FireType.VEGETATION

    # One-call convenience: classify directly from HDF5 fire stats
    fire_ids, types = classify_fires_from_h5()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


# ---------------------------------------------------------------------------
# Fire types
# ---------------------------------------------------------------------------

class FireType(IntEnum):
    """Fire classification result."""
    VEGETATION = 0   # real vegetation fire — include
    STATIC = 1       # persistent industrial — exclude
    CROP = 2         # agricultural residue — region-dependent


# ---------------------------------------------------------------------------
# Per-region filter configuration
# ---------------------------------------------------------------------------

# LC classes always classified as STATIC regardless of region.
# Urban (oil/gas/industry, t_ratio 15-30), Ocean (offshore platforms, t_ratio
# 12-33), Bare (desert oil/gas, t_ratio 17-33), Water (misclassified coastal
# industry or within-fire lakes — tiny count, not worth keeping).
STATIC_LC: frozenset[str] = frozenset({"Urban", "Ocean", "Bare", "Water"})

# Safety-net: any fire with t_ratio above this is STATIC regardless of LC
# (catches LC misclassification — e.g. an oil flare in a "Grass" pixel).
STATIC_T_RATIO_THRESH = 10.0


@dataclass(frozen=True)
class RegionFilter:
    """Per-region fire filter specification.

    Parameters
    ----------
    exclude_lc : set[str]
        LC classes to exclude (classified as STATIC).  Always includes
        the global STATIC_LC set.
    crop_is_vegetation : bool
        If True, Cropland fires in this region are reclassified as
        VEGETATION (e.g. Africa where crop burning is real vegetation fire).
        If False, Cropland fires are classified as CROP.
    """
    exclude_lc: frozenset[str] = frozenset()
    crop_is_vegetation: bool = False


# Region ID → filter rules.
# Region IDs match wildfire_regions.tif: 1=W.Europe, 2=MENA, 3=Africa,
# 4=N.Asia, 5=S.Asia, 6=Oceania, 7=N.NA, 8=C.NA, 9=S.America, 10=E.Europe.
#
# Rationale from per-region diagnostic analysis:
REGION_FILTERS: dict[int, RegionFilter] = {
    # W.Europe: tiny region, include everything burnable.  Crop (26 fires)
    # has low ign% and looks semi-wild.  Only Urban/Ocean/Water are tame.
    1: RegionFilter(crop_is_vegetation=True),

    # MENA: Bare/Urban/Ocean/Water/Cropland are industrial (t_ratio 17-33).
    # Grass/DC_Broad/Other_Forest/Shrub are actual vegetation fires (t=0.32-0.36).
    2: RegionFilter(exclude_lc=frozenset({"Cropland"})),

    # Africa: high ign% (3-7%) is normal — pastoral/slash-and-burn.  Do NOT
    # filter on ignition.  Only Bare/Urban/Ocean/Water have industrial t_ratio.
    # Cropland is clearly agricultural — classify as CROP.
    3: RegionFilter(),

    # N.Asia: clean wild signatures in forest/grass/shrub.  Bare has
    # industrial t_ratio=18.  Cropland (xy=11.9, t=0.29) is agricultural.
    4: RegionFilter(),

    # S.Asia: Forest fires are real but have high ign%.  Grass (t_ratio=12.85)
    # and Shrub (t_ratio=12.34) are persistent agricultural burning — exclude.
    # Cropland (ign=6.63%) is clearly agricultural.
    5: RegionFilter(exclude_lc=frozenset({"Grass", "Shrub"})),

    # Oceania: clean wild across DC_Broad, Grass, Shrub.  Only Bare/Urban/Ocean
    # are industrial.  Cropland (16 fires) is too few to matter either way.
    6: RegionFilter(),

    # N.NA: almost everything is boreal wildfire.  Even Water fires (lakes
    # within burn areas) are wild — but too few to matter, keep the exclusion.
    7: RegionFilter(),

    # C.NA: Grass (t_ratio=10.8) and Cropland (t=13.1) are Great Plains
    # prescribed burns.  Forest/Shrub/Mixed_Forest are wild.
    8: RegionFilter(exclude_lc=frozenset({"Grass"})),

    # S.America: deforestation fires (Cerrado/Amazon) often start on cropland
    # edges (xy=14.7, t=0.33), but still classify Cropland as CROP to be
    # conservative — the forest LC classes capture the real fires.
    9: RegionFilter(),

    # E.Europe: Wetland (peatland burns, xy=15.4, t=0.27) is the highlight.
    # Cropland (199 fires, ign=1.57%) is agricultural.
    10: RegionFilter(),
}

# Default for unknown regions: conservative (exclude cropland, static LC only)
_DEFAULT_FILTER = RegionFilter(crop_is_vegetation=False)


# ---------------------------------------------------------------------------
# Region name ↔ ID mapping (convenience)
# ---------------------------------------------------------------------------

REGION_NAMES: dict[int, str] = {
    0: "(none)", 1: "W.Europe", 2: "MENA", 3: "Africa",
    4: "N.Asia", 5: "S.Asia", 6: "Oceania", 7: "N.NA",
    8: "C.NA", 9: "S.America", 10: "E.Europe",
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_fires(
    region_ids: np.ndarray,
    lc_classes: np.ndarray,
    t_ratios: np.ndarray | None = None,
) -> np.ndarray:
    """Classify fires into VEGETATION / STATIC / CROP per region.

    Parameters
    ----------
    region_ids : (N,) int array — region ID per fire.
    lc_classes : (N,) str array — LC class name per fire (e.g. "DC_Broad").
    t_ratios   : (N,) float array, optional — temporal ratio per fire.
                 If provided, fires with t_ratio > STATIC_T_RATIO_THRESH
                 are forced to STATIC regardless of LC class.

    Returns
    -------
    (N,) int array of FireType values.
    """
    n = len(region_ids)
    result = np.full(n, FireType.VEGETATION, dtype=np.int8)

    for i in range(n):
        lc = str(lc_classes[i])
        rid = int(region_ids[i])
        filt = REGION_FILTERS.get(rid, _DEFAULT_FILTER)

        # Global static LC classes
        if lc in STATIC_LC or lc in filt.exclude_lc:
            result[i] = FireType.STATIC
            continue

        # Cropland: region-dependent
        if lc == "Cropland":
            result[i] = (FireType.VEGETATION
                         if filt.crop_is_vegetation
                         else FireType.CROP)
            continue

        # Everything else is vegetation
        result[i] = FireType.VEGETATION

    # Safety net: t_ratio override
    if t_ratios is not None:
        static_override = t_ratios > STATIC_T_RATIO_THRESH
        result[static_override] = FireType.STATIC

    return result


def classify_fires_sql() -> str:
    """Return a SQL CASE expression for fire classification in DuckDB.

    Uses the same logic as ``classify_fires`` but as a SQL expression
    that can be used in DuckDB queries.  Assumes columns: region_id,
    lc_class, t_ratio.

    Returns
    -------
    str : SQL CASE expression evaluating to 0 (VEGETATION), 1 (STATIC),
          or 2 (CROP).
    """
    # Build per-region WHEN clauses for non-default behaviour
    parts = []

    # 1. Safety net: t_ratio override (highest priority)
    parts.append(
        f"WHEN t_ratio > {STATIC_T_RATIO_THRESH} THEN {FireType.STATIC}"
    )

    # 2. Global static LC
    static_lc_str = ", ".join(f"'{lc}'" for lc in sorted(STATIC_LC))
    parts.append(
        f"WHEN lc_class IN ({static_lc_str}) THEN {FireType.STATIC}"
    )

    # 3. Per-region extra exclusions
    for rid, filt in sorted(REGION_FILTERS.items()):
        if filt.exclude_lc:
            extra = ", ".join(f"'{lc}'" for lc in sorted(filt.exclude_lc))
            parts.append(
                f"WHEN region_id = {rid} AND lc_class IN ({extra}) "
                f"THEN {FireType.STATIC}"
            )

    # 4. Cropland: per-region
    crop_veg_rids = [rid for rid, f in REGION_FILTERS.items()
                     if f.crop_is_vegetation]
    if crop_veg_rids:
        rids_str = ", ".join(str(r) for r in sorted(crop_veg_rids))
        parts.append(
            f"WHEN lc_class = 'Cropland' AND region_id IN ({rids_str}) "
            f"THEN {FireType.VEGETATION}"
        )
    # Cropland in other regions → CROP
    parts.append(
        f"WHEN lc_class = 'Cropland' THEN {FireType.CROP}"
    )

    # 5. Default → VEGETATION
    parts.append(f"ELSE {FireType.VEGETATION}")

    return "CASE\n    " + "\n    ".join(parts) + "\nEND"


def fire_type_label(ft: int) -> str:
    """Human-readable label for a FireType value."""
    return {
        FireType.VEGETATION: "vegetation",
        FireType.STATIC: "static",
        FireType.CROP: "crop",
    }.get(ft, "unknown")


# ---------------------------------------------------------------------------
# LC lookup
# ---------------------------------------------------------------------------

# Copernicus CGLS-LC100 discrete class → short name
_LC_CODE_TO_NAME: dict[int, str] = {
    20: "Shrub", 30: "Grass", 40: "Cropland", 50: "Urban",
    60: "Bare", 80: "Water", 90: "Wetland", 100: "Moss_Lichen",
    111: "EG_Needle", 112: "EG_Broad", 113: "DC_Needle",
    114: "DC_Broad", 115: "Mixed_Forest", 116: "Other_Forest",
    121: "EG_Needle", 122: "EG_Broad", 123: "DC_Needle",
    124: "DC_Broad", 125: "Mixed_Forest", 126: "Other_Forest",
    200: "Ocean",
}


def _default_lc_path() -> str:
    """Return default path to Copernicus LC100 discrete_classification.tif."""
    import os
    from firecomp.config import config
    return os.path.join(
        config.data_dir, "copernicus_lc100", "discrete_classification.tif",
    )


def lookup_lc(
    lons: np.ndarray, lats: np.ndarray, lc_path: str | None = None,
) -> np.ndarray:
    """Vectorised (lon, lat) → LC class name via Copernicus LC100 raster.

    Parameters
    ----------
    lons, lats : (N,) float arrays — fire centroids.
    lc_path    : path to discrete_classification.tif.  Defaults to
                 ``data/copernicus_lc100/discrete_classification.tif``.

    Returns
    -------
    (N,) str array of LC class names (e.g. "DC_Broad", "Cropland").
    """
    from osgeo import gdal
    gdal.UseExceptions()

    if lc_path is None:
        lc_path = _default_lc_path()

    ds = gdal.Open(lc_path)
    gt = ds.GetGeoTransform()
    H, W = ds.RasterYSize, ds.RasterXSize
    band = ds.GetRasterBand(1)

    px = ((lons - gt[0]) / gt[1]).astype(np.int32)
    py = ((lats - gt[3]) / gt[5]).astype(np.int32)
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    result = np.full(len(lons), "Unknown", dtype="U16")
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
        codes = chunk[rows_local, cols].astype(np.int32)
        for i, code in zip(valid_idx[in_chunk], codes):
            result[i] = _LC_CODE_TO_NAME.get(int(code), "Unknown")

    ds = None
    return result


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------


def classify_fires_from_h5(
    h5_path: str | None = None,
    lc_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify all fires in the HDF5 database by type.

    Loads fire stats, looks up regions and LC classes, and classifies
    each fire.  All paths default to the standard ``data/`` locations.

    Returns
    -------
    fire_ids : (N,) int32 array
    fire_types : (N,) int8 array of FireType values
    """
    import h5py

    from firecomp.config import config
    from firecomp.core.regions import RegionRaster

    if h5_path is None:
        h5_path = config.vnp14_path

    h5 = h5py.File(h5_path, "r")
    g = h5["stats"]
    fire_ids = g["id"][:]
    cx = (g["min_x"][:].astype(np.float64)
          + g["max_x"][:].astype(np.float64)) / 2
    cy = (g["min_y"][:].astype(np.float64)
          + g["max_y"][:].astype(np.float64)) / 2
    t_ratios = g["t_ratio"][:].astype(np.float64)
    h5.close()

    # Region + LC lookups
    rr = RegionRaster.load()
    region_ids = rr.lookup_coords(cx, cy)
    lc_classes = lookup_lc(cx, cy, lc_path)

    fire_types = classify_fires(region_ids, lc_classes, t_ratios)
    return fire_ids, fire_types
