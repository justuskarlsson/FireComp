"""next_day/download.py — Download VNP03IMG geolocation files for next-day prediction.

Workflow:

  1. Create manifest (deterministic image selection, run once):
       python -m firecomp.next_day.download --manifest --top 1000

  2. Download from manifest (distributed, resume-safe):
       python -m firecomp.next_day.download --top 1000
       python -m firecomp.next_day.download --top 1000 -ji 0 -jn 4

  3. Analyze coverage (which fire-day pairs have both T and T+1):
       python -m firecomp.next_day.download --analyze --top 1000

  4. Download supplemental T+1 images for uncovered pairs:
       python -m firecomp.next_day.download --supplemental --top 1000

Download jobs load the manifest and never recompute the selection,
so all jobs use the exact same image list and ordering.

Resume-safe: files already on disk are skipped.  Top-2000 is a strict
superset of top-1000, so you can create a new manifest with higher
--top and only the delta gets downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from firecomp.config import config
from firecomp.dsrc.viirs import (
    group_prefixes_by_date,
    prefix_to_date_str,
    result_filename,
    viirs_granule_prefix,
)
from firecomp.dsrc.earthaccess import safe_download

REGION_NAMES: dict[int, str] = {
    0: "(none)", 1: "W.Europe", 2: "MENA", 3: "Africa",
    4: "N.Asia", 5: "S.Asia", 6: "Oceania", 7: "N.NA",
    8: "C.NA", 9: "S.America", 10: "E.Europe",
}


# ── Data loading ─────────────────────────────────────────────────────


def _lookup_regions(
    lons: np.ndarray, lats: np.ndarray, tif_path: str,
) -> np.ndarray:
    """Vectorised (lon, lat) -> region_id via the 0.1-deg raster."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(tif_path)
    raster = ds.ReadAsArray()  # (1800, 3600) uint8
    ds = None
    px = np.clip(((lons + 180.0) / 0.1).astype(np.int32), 0, 3599)
    py = np.clip(((90.0 - lats) / 0.1).astype(np.int32), 0, 1799)
    return raster[py, px].astype(np.int32)


def _load_images(h5: h5py.File) -> pd.DataFrame:
    """Load image id -> filename mapping from HDF5."""
    ids = h5["images/id"][:]
    fnames_raw = h5["images/file_name"][:]
    fnames = [
        f.decode() if isinstance(f, bytes) else str(f) for f in fnames_raw
    ]
    start_times = h5["images/start_time"][:].astype("datetime64[ms]")
    num_fire_pixels = h5["images/num_fire_pixels"][:]
    return pd.DataFrame({
        "image_id": ids,
        "file_name": fnames,
        "start_time": pd.to_datetime(start_times),
        "num_fire_pixels": num_fire_pixels,
    })


def _load_fires(h5: h5py.File, region_tif: str) -> pd.DataFrame:
    """Load all fire stats with region assignment, filtered to 2018+.

    Uses the unfiltered ``stats`` group — fire-type filtering is handled
    downstream by the LC-based classifier (fire_filter.py), so the old
    ≥100-pixel threshold is no longer needed.
    """
    print("  Loading stats...")
    g = h5["stats"]
    fire_ids = g["id"][:]
    start_dates = g["start_date"][:].astype("datetime64[ms]")
    cx = (g["min_x"][:].astype(np.float64)
          + g["max_x"][:].astype(np.float64)) / 2
    cy = (g["min_y"][:].astype(np.float64)
          + g["max_y"][:].astype(np.float64)) / 2

    print("  Looking up regions...")
    region_ids = _lookup_regions(cx, cy, region_tif)

    df = pd.DataFrame({
        "fire_id": fire_ids,
        "start_date": pd.to_datetime(start_dates),
        "region_id": region_ids,
    })
    df = df[df["start_date"] >= "2018-01-01"].copy()
    print(f"    {len(df):,} fires (2018+)")
    return df


def _build_fire_image(
    h5: h5py.File, valid_fire_ids: np.ndarray,
) -> pd.DataFrame:
    """Build unique (fire_id, image_id) pairs from pixel data (chunked)."""
    n_pix = h5["pixels/fire_id"].shape[0]
    print(f"  Building fire-image links ({n_pix:,} pixels, chunked)...")

    CHUNK = 50_000_000
    all_keys: list[np.ndarray] = []
    kept = 0
    for lo in range(0, n_pix, CHUNK):
        hi = min(lo + CHUNK, n_pix)
        fids = h5["pixels/fire_id"][lo:hi]
        mask = np.isin(fids, valid_fire_ids)
        if mask.any():
            iids = h5["pixels/image_id"][lo:hi]
            keys = (fids[mask].astype(np.int64) << 32
                    | iids[mask].astype(np.int64))
            all_keys.append(np.unique(keys))
            kept += int(mask.sum())
        print(f"    {hi:>12,}/{n_pix:,}  kept {kept:,}", end="\r")
    print()

    if not all_keys:
        return pd.DataFrame(columns=["fire_id", "image_id"])

    merged = np.unique(np.concatenate(all_keys))
    df = pd.DataFrame({
        "fire_id": (merged >> 32).astype(np.int32),
        "image_id": (merged & 0xFFFFFFFF).astype(np.int32),
    })
    print(f"    {len(df):,} unique fire-image pairs")
    return df


# ── Image selection ──────────────────────────────────────────────────


def _build_image_stats(
    fire_image: pd.DataFrame, fires_df: pd.DataFrame,
    images_df: pd.DataFrame,
) -> pd.DataFrame:
    """Get fire pixel count per image and assign primary region."""
    fi = fire_image.merge(fires_df[["fire_id", "region_id"]], on="fire_id")

    # Primary region = mode of fire regions per image
    regions = fi.groupby("image_id")["region_id"].agg(
        lambda x: x.mode().iloc[0]
    ).reset_index()
    regions.columns = ["image_id", "region_id"]

    # num_fire_pixels comes directly from the HDF5 images table
    stats = regions.merge(
        images_df[["image_id", "num_fire_pixels"]], on="image_id",
    )
    return stats


def select_top_images(
    h5_path: str, region_tif: str, top_n: int,
) -> pd.DataFrame:
    """Load HDF5, rank images by fire pixel count per region, return top N.

    Returns DataFrame with columns: image_id, num_fire_pixels, region_id,
    rank, file_name, start_time, prefix, date.
    """
    t0 = time.time()
    h5 = h5py.File(h5_path, "r")

    images_df = _load_images(h5)
    fires_df = _load_fires(h5, region_tif)
    fire_image = _build_fire_image(h5, fires_df["fire_id"].values)

    h5.close()

    print("  Computing image stats (fire pixels per image, region)...")
    image_stats = _build_image_stats(fire_image, fires_df, images_df)

    # Keep only valid regions (> 0)
    image_stats = image_stats[image_stats["region_id"] > 0].copy()

    # Rank within region (most fire pixels first)
    image_stats["rank"] = (
        image_stats
        .groupby("region_id")["num_fire_pixels"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Select top N per region
    selected = image_stats[image_stats["rank"] <= top_n].copy()

    # Join with image metadata (drop num_fire_pixels from images_df to
    # avoid duplicate column — it's already in image_stats)
    selected = selected.merge(
        images_df.drop(columns=["num_fire_pixels"]), on="image_id",
    )

    # Add granule prefix and date string
    selected["prefix"] = selected["file_name"].apply(viirs_granule_prefix)
    selected["date"] = selected["start_time"].dt.strftime("%Y-%m-%d")

    # Sort by rank first, then region — so rank 1 from all regions comes
    # before rank 2.  With stride-based job distribution (iloc[ji::jn]),
    # a half-finished top-2000 run still yields a complete top-1000.
    selected = (
        selected
        .sort_values(["rank", "region_id"])
        .reset_index(drop=True)
    )

    dt = time.time() - t0
    print(f"  Selected {len(selected):,} images in {dt:.0f}s")
    return selected


# ── Manifest ─────────────────────────────────────────────────────────


def save_manifest(
    selected: pd.DataFrame, output_dir: str, top_n: int,
) -> str:
    """Save the image selection as a JSON manifest file.

    Returns the manifest file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"manifest_top{top_n}.json")

    # Per-region summary
    region_summary = {}
    for rid, grp in selected.groupby("region_id"):
        rname = REGION_NAMES.get(int(rid), f"R{rid}")
        region_summary[rname] = {
            "region_id": int(rid),
            "n_images": len(grp),
            "total_fire_pixels": int(grp["num_fire_pixels"].sum()),
            "max_fire_pixels": int(grp["num_fire_pixels"].max()),
        }

    manifest = {
        "top_per_region": top_n,
        "total_images": len(selected),
        "created": datetime.now().isoformat(),
        "regions": region_summary,
        "images": [
            {
                "image_id": int(row.image_id),
                "file_name": row.file_name,
                "prefix": row.prefix,
                "date": row.date,
                "region_id": int(row.region_id),
                "num_fire_pixels": int(row.num_fire_pixels),
                "rank": int(row.rank),
            }
            for row in selected.itertuples()
        ],
    }

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved: {path}")
    return path


# ── Download ─────────────────────────────────────────────────────────


def download_vnp03(my_images: pd.DataFrame, output_dir: str) -> None:
    """Search LAADS DAAC and download VNP03IMG files for the given images."""
    import earthaccess
    earthaccess.login()

    os.makedirs(output_dir, exist_ok=True)

    needed_prefixes = set(my_images["prefix"].dropna())
    if not needed_prefixes:
        print("  No images to download!")
        return

    # Skip prefixes for files that already exist on disk
    existing = set()
    for fname in os.listdir(output_dir):
        if fname.startswith("VNP03IMG") and fname.endswith(".nc"):
            pfx = viirs_granule_prefix(fname)
            if pfx:
                existing.add(pfx)
    already = needed_prefixes & existing
    to_search = needed_prefixes - existing
    if already:
        print(f"  {len(already):,} already on disk, "
              f"{len(to_search):,} to search/download")
    if not to_search:
        print("  All files already downloaded!")
        return

    # Group by date for batch search
    by_date = group_prefixes_by_date(to_search)
    print(f"  Searching {len(to_search):,} granules "
          f"across {len(by_date):,} dates...")

    all_results: list = []
    found: set[str] = set()
    for date_str in tqdm(sorted(by_date.keys()), desc="Searching LAADS"):
        results = earthaccess.search_data(
            short_name="VNP03IMG",
            temporal=(date_str, date_str),
            count=-1,
        )
        for r in results:
            prefix = viirs_granule_prefix(result_filename(r))
            if prefix in to_search and prefix not in found:
                all_results.append(r)
                found.add(prefix)

    missing = to_search - found
    print(f"  Found {len(all_results):,} / {len(to_search):,} "
          f"on LAADS DAAC")
    if missing:
        print(f"  WARNING: {len(missing):,} granules not found on LAADS")

    if all_results:
        print(f"  Downloading {len(all_results):,} VNP03IMG files...")
        prefix_to_path = safe_download(
            all_results, output_dir,
            delete_invalid=True, max_pending_wait=-1,
        )
        print(f"  Downloaded/verified {len(prefix_to_path):,} files")


# ── Sample picking ───────────────────────────────────────────────────


def _prepare_sample_pool(
    manifest_path: str, h5_path: str, region_tif: str,
    *, min_fire_pixels: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the pool of candidate (fire, day_T) pairs for sampling.

    This is the expensive step (~3 min): loads HDF5, builds fire-image
    links, computes coverage, enriches with metadata and fire types.

    Parameters
    ----------
    min_fire_pixels : drop fires with fewer than this many total pixels
                      (from VNP14 stats.num_fire). 0 = no filter.

    Returns
    -------
    pairs : DataFrame — all candidate (fire, day_T) pairs with columns:
        fire_id, day_T, day_T1, region_id, t_full, t1_full, num_fire,
        n_tiles_est, fire_type, plus bbox columns.
    fire_image_df : DataFrame — (fire_id, image_id, image_date, file_name,
        downloaded) for supplemental counting.
    """
    from firecomp.core.fire_filter import (
        FireType, classify_fires_from_h5, fire_type_label,
    )
    from firecomp.dsrc.vnp14 import DEG_CELL_SIZE

    IMG_SIZE = 256
    PADDING = 16
    CROP = IMG_SIZE - 2 * PADDING  # 224 px effective tile

    t0 = time.time()
    manifest = _load_manifest(manifest_path)
    downloaded_ids = set(manifest["image_id"].values)

    # Also check files already on disk
    output_dir = os.path.dirname(manifest_path)
    on_disk = set()
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if fname.startswith("VNP03IMG") and fname.endswith(".nc"):
                pfx = viirs_granule_prefix(fname)
                if pfx:
                    on_disk.add(pfx)
    if on_disk:
        print(f"  {len(on_disk):,} VNP03IMG files on disk")

    print(f"  {len(downloaded_ids):,} images in manifest")

    h5 = h5py.File(h5_path, "r")
    fires_df = _load_fires(h5, region_tif)
    fire_image = _build_fire_image(h5, fires_df["fire_id"].values)

    # Image metadata
    img_ids = h5["images/id"][:]
    img_times = h5["images/start_time"][:].astype("datetime64[ms]")
    fnames_raw = h5["images/file_name"][:]
    fnames = [
        f.decode() if isinstance(f, bytes) else str(f) for f in fnames_raw
    ]
    h5.close()

    img_df = pd.DataFrame({
        "image_id": img_ids,
        "file_name": fnames,
        "image_date": pd.to_datetime(img_times).normalize(),
    })

    # ── 1. Full coverage: count total vs downloaded per (fire, day) ──
    fi = fire_image.merge(img_df, on="image_id")
    fi["prefix"] = fi["file_name"].apply(viirs_granule_prefix)
    fi["downloaded"] = (
        fi["image_id"].isin(downloaded_ids) | fi["prefix"].isin(on_disk)
    )

    fire_day_stats = (
        fi.groupby(["fire_id", "image_date"])
        .agg(n_total=("image_id", "size"),
             n_downloaded=("downloaded", "sum"))
        .reset_index()
    )
    fire_day_stats["n_downloaded"] = fire_day_stats["n_downloaded"].astype(int)
    fire_day_stats["fully_covered"] = (
        fire_day_stats["n_total"] == fire_day_stats["n_downloaded"]
    )
    fire_day_stats.rename(columns={"image_date": "day"}, inplace=True)

    n_any = int((fire_day_stats["n_downloaded"] > 0).sum())
    n_full = int(fire_day_stats["fully_covered"].sum())
    print(f"  Fire-days with any coverage:  {n_any:,}")
    print(f"  Fire-days with full coverage: {n_full:,}")

    # ── 2. Consecutive (T, T+1) pairs where T has any coverage ──
    has_any = fire_day_stats[fire_day_stats["n_downloaded"] > 0].copy()
    all_fire_days = fire_day_stats[["fire_id", "day"]].copy()

    fd = has_any[["fire_id", "day"]].copy()
    fd["day_plus1"] = fd["day"] + pd.Timedelta(days=1)

    pairs = fd.merge(
        all_fire_days,
        left_on=["fire_id", "day_plus1"],
        right_on=["fire_id", "day"],
        suffixes=("", "_T1"),
    )
    pairs = pairs[["fire_id", "day"]].rename(columns={"day": "day_T"})
    pairs = pairs.drop_duplicates()
    print(f"  {len(pairs):,} (fire, day_T) pairs with T+1 in database")

    # Attach full-coverage flags
    cov = fire_day_stats[["fire_id", "day", "fully_covered"]].copy()
    pairs = pairs.merge(
        cov.rename(columns={"day": "day_T", "fully_covered": "t_full"}),
        on=["fire_id", "day_T"],
    )
    pairs["day_T1"] = pairs["day_T"] + pd.Timedelta(days=1)
    pairs = pairs.merge(
        cov.rename(columns={"day": "day_T1", "fully_covered": "t1_full"}),
        on=["fire_id", "day_T1"],
    )

    n_both_full = (pairs["t_full"] & pairs["t1_full"]).sum()
    n_t_full = int(pairs["t_full"].sum())
    n_t1_full = int(pairs["t1_full"].sum())
    print(f"  T fully covered:     {n_t_full:,} / {len(pairs):,}")
    print(f"  T+1 fully covered:   {n_t1_full:,} / {len(pairs):,}")
    print(f"  Both fully covered:  {n_both_full:,} / {len(pairs):,}")

    # ── 3. Enrich with fire metadata ──
    pairs = pairs.merge(fires_df, on="fire_id")

    h5 = h5py.File(h5_path, "r")
    g = h5["stats"]
    bbox_df = pd.DataFrame({
        "fire_id": g["id"][:],
        "min_x": g["min_x"][:].astype(np.float64),
        "max_x": g["max_x"][:].astype(np.float64),
        "min_y": g["min_y"][:].astype(np.float64),
        "max_y": g["max_y"][:].astype(np.float64),
        "num_fire": g["num_fire"][:],
    })
    h5.close()

    pairs = pairs.merge(bbox_df, on="fire_id")

    # ── 3a. Filter small fires ──
    if min_fire_pixels > 0:
        n_before = pairs["fire_id"].nunique()
        pairs = pairs[pairs["num_fire"] >= min_fire_pixels].copy()
        n_after = pairs["fire_id"].nunique()
        print(f"  min_fire_pixels={min_fire_pixels}: "
              f"{n_before:,} → {n_after:,} fires "
              f"({n_before - n_after:,} dropped), "
              f"{len(pairs):,} pairs remain")

    # ── 3b. Classify fires ──
    fire_ids_arr, fire_types_arr = classify_fires_from_h5(h5_path)
    ftype_df = pd.DataFrame({
        "fire_id": fire_ids_arr, "fire_type": fire_types_arr,
    })
    pairs = pairs.merge(ftype_df, on="fire_id", how="left")
    for ft in FireType:
        n = int((pairs["fire_type"] == ft).sum())
        print(f"  {fire_type_label(ft):>12}: {n:,} pairs")

    # Tile count estimate
    px_per_deg = 1.0 / DEG_CELL_SIZE
    pairs["width_px"] = (pairs["max_x"] - pairs["min_x"]) * px_per_deg
    pairs["height_px"] = (pairs["max_y"] - pairs["min_y"]) * px_per_deg
    pairs["nx"] = np.maximum(1, (pairs["width_px"] / CROP).astype(int))
    pairs["ny"] = np.maximum(1, (pairs["height_px"] / CROP).astype(int))
    pairs["n_tiles_est"] = pairs["nx"] * pairs["ny"]

    dt = time.time() - t0
    print(f"  Pool ready: {len(pairs):,} pairs in {dt:.0f}s\n")

    return pairs, fi


def _select_from_pool(
    pairs: pd.DataFrame,
    num_samples: int = 50_000,
    cap_per_fire: int = 50,
    seed: int = 42,
    weight_exp: float = 0.5,
    max_static_pct: float = 0.20,
) -> pd.DataFrame:
    """Select samples from the prepared pool (fast, ~1s).

    Parameters
    ----------
    pairs        : output of ``_prepare_sample_pool``.
    num_samples  : target total (fire, day) pairs.
    cap_per_fire : max days per fire.
    seed         : random seed.
    weight_exp   : exponent for num_fire weighting.  0 = uniform,
                   0.5 = sqrt, 1.0 = linear.  Higher values bias
                   toward larger fires.  Default 0.5 (sqrt).
    max_static_pct : max fraction of static fires per region (default 20%).

    Returns
    -------
    DataFrame — selected samples.
    """
    from firecomp.core.fire_filter import FireType

    # ── Cap per fire — prefer fully-covered pairs ──
    rng = np.random.RandomState(seed)
    pool = pairs.copy()
    pool["_rand"] = rng.random(len(pool))
    pool["_both_full"] = pool["t_full"] & pool["t1_full"]
    pool = pool.sort_values(
        ["fire_id", "_both_full", "_rand"],
        ascending=[True, False, True],
    )
    pool["_fire_rank"] = pool.groupby("fire_id").cumcount() + 1
    pool = pool[pool["_fire_rank"] <= cap_per_fire].copy()

    # ── Balanced selection across regions (static capped) ──
    regions = sorted(r for r in pool["region_id"].unique() if r > 0)
    per_region = max(1, num_samples // len(regions))
    max_static = int(per_region * max_static_pct)

    selected_parts = []
    for rid in regions:
        rp = pool[pool["region_id"] == rid].copy()
        if len(rp) == 0:
            continue

        # Weighted sampling: weight = num_fire^weight_exp
        if weight_exp > 0:
            w = np.maximum(rp["num_fire"].values, 1).astype(float)
            w = w ** weight_exp
            rp["_weight"] = w / w.sum()
        else:
            rp["_weight"] = 1.0 / len(rp)

        # Static capped, non-static fills the rest
        static = rp[rp["fire_type"] == FireType.STATIC]
        non_static = rp[rp["fire_type"] != FireType.STATIC]

        n_static = min(max_static, len(static))
        n_non_static = min(per_region - n_static, len(non_static))

        picked_static = static.sample(
            n=n_static, weights="_weight", random_state=rng, replace=False,
        ) if n_static > 0 else static.iloc[:0]

        picked_non = non_static.sample(
            n=n_non_static, weights="_weight", random_state=rng, replace=False,
        ) if n_non_static > 0 else non_static.iloc[:0]

        selected_parts.append(pd.concat([picked_non, picked_static]))

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.sort_values(["region_id", "day_T"]).reset_index(drop=True)


def _find_supplemental(
    selected: pd.DataFrame, fire_image_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find images needed but not yet downloaded for selected samples.

    Uses the pre-built fire-image DataFrame (with 'downloaded' flag) to
    avoid re-opening HDF5.

    Returns DataFrame of supplemental images with columns:
    image_id, file_name, prefix, image_date (one row per unique image).
    """
    # (fire_id, day) slots needed: both day_T and day_T+1
    need_t = selected[["fire_id", "day_T"]].rename(
        columns={"day_T": "image_date"},
    )
    need_t1 = selected[["fire_id", "day_T"]].copy()
    need_t1["image_date"] = need_t1["day_T"] + pd.Timedelta(days=1)
    need_t1 = need_t1.drop(columns=["day_T"])
    need = pd.concat([need_t, need_t1]).drop_duplicates()

    # Join with fire-image links
    needed = fire_image_df.merge(need, on=["fire_id", "image_date"])
    supp = needed[~needed["downloaded"]].drop_duplicates(subset=["image_id"])
    return supp[["image_id", "file_name", "prefix", "image_date"]].copy()


def _count_supplemental(
    selected: pd.DataFrame, fire_image_df: pd.DataFrame,
) -> int:
    """Count unique images needed but not yet downloaded."""
    return len(_find_supplemental(selected, fire_image_df))


def _recompute_coverage(
    pairs: pd.DataFrame, fi: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute t_full/t1_full flags from current fi downloaded state.

    After marking additional images as downloaded in *fi* (e.g. veg
    supplemental), call this on static/crop pools so that
    ``_select_cheapest`` sees the updated coverage.
    """
    fire_day_cov = (
        fi.groupby(["fire_id", "image_date"])
        .agg(n_total=("image_id", "size"),
             n_dl=("downloaded", "sum"))
        .reset_index()
    )
    fire_day_cov["n_dl"] = fire_day_cov["n_dl"].astype(int)
    fire_day_cov["fully_covered"] = (
        fire_day_cov["n_total"] == fire_day_cov["n_dl"]
    )
    cov = fire_day_cov[["fire_id", "image_date", "fully_covered"]]

    out = pairs.drop(columns=["t_full", "t1_full"], errors="ignore")
    out = out.merge(
        cov.rename(columns={"image_date": "day_T", "fully_covered": "t_full"}),
        on=["fire_id", "day_T"], how="left",
    )
    out = out.merge(
        cov.rename(columns={"image_date": "day_T1",
                             "fully_covered": "t1_full"}),
        on=["fire_id", "day_T1"], how="left",
    )
    out["t_full"] = out["t_full"].fillna(False)
    out["t1_full"] = out["t1_full"].fillna(False)
    return out


def _summarise_selection(
    selected: pd.DataFrame, n_supp: int | None = None,
    label: str = "",
):
    """Print summary stats for a sample selection."""
    from firecomp.core.fire_filter import FireType, fire_type_label

    if label:
        print(f"\n  ── {label} ──")

    regions = sorted(r for r in selected["region_id"].unique() if r > 0)
    n_fires = selected["fire_id"].nunique()
    total_tiles = int(selected["n_tiles_est"].sum())
    n_both = int((selected["t_full"] & selected["t1_full"]).sum())
    n_need = len(selected) - n_both

    print(f"  {len(selected):,} samples, {n_fires:,} fires, "
          f"~{total_tiles:,} tiles")
    print(f"  Fully covered: {n_both:,}  |  Need supplemental: {n_need:,}")
    if n_supp is not None:
        supp_tb = n_supp * 100 / 1e6
        print(f"  Supplemental images to download: "
              f"{n_supp:,} (~{supp_tb:.2f} TB)")

    # Fire type breakdown
    for ft in FireType:
        n = int((selected["fire_type"] == ft).sum())
        pct = 100 * n / len(selected)
        print(f"    {fire_type_label(ft):>12}: {n:>7,} ({pct:4.1f}%)")

    # num_fire percentiles
    nf = selected["num_fire"].values
    p25, p50, p75 = np.percentile(nf, [25, 50, 75])
    print(f"  num_fire P25={p25:,.0f}  P50={p50:,.0f}  P75={p75:,.0f}")

    # Per-region table
    print(f"\n  {'Region':<12} {'Pairs':>7} {'Fires':>7} "
          f"{'Full':>7} {'Need':>7} {'~Tiles':>7} "
          f"{'P50':>7} {'veg%':>5}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*5}")
    for rid in regions:
        rg = selected[selected["region_id"] == rid]
        rname = REGION_NAMES.get(int(rid), f"R{rid}")
        rf = int((rg["t_full"] & rg["t1_full"]).sum())
        rn = len(rg) - rf
        p50r = np.median(rg["num_fire"].values)
        n_veg = int((rg["fire_type"] == FireType.VEGETATION).sum())
        vpct = 100 * n_veg / len(rg) if len(rg) else 0
        print(f"  {rname:<12} {len(rg):>7,} "
              f"{rg['fire_id'].nunique():>7,} "
              f"{rf:>7,} {rn:>7,} "
              f"{int(rg['n_tiles_est'].sum()):>7,} "
              f"{p50r:>7,.0f} {vpct:>4.0f}%")
    print()


def pick_samples(
    manifest_path: str, h5_path: str, region_tif: str,
    veg_per_region: int = 10_000,
    static_per_region: int = 2_000,
    crop_per_region: int = 2_000,
    cap_veg: int = 500,
    cap_static: int = 50,
    cap_crop: int = 50,
    min_fire_pixels: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick next-day samples: 3-tier selection (veg → static → crop).

    Selects vegetation pairs first (cheapest-first, minimising VNP03
    downloads), computes which supplemental images they need, then
    selects static and crop pairs that piggyback on those downloads
    before adding extra images.

    Parameters
    ----------
    veg_per_region    : target vegetation pairs per region.
    static_per_region : target static pairs per region (on top of veg).
    crop_per_region   : target crop pairs per region (on top of veg).
    cap_veg           : max sample-days per vegetation fire.
    cap_static        : max sample-days per static fire.
    cap_crop          : max sample-days per crop fire.
    min_fire_pixels   : drop fires with fewer total pixels (0 = disable).
    seed              : random seed for reproducibility.

    Returns
    -------
    (selected, supplemental) — combined selected samples DataFrame and
    combined supplemental images DataFrame.
    Also writes ``data/next_day_v3/samples.json``.
    """
    from firecomp.core.fire_filter import FireType

    pairs, fi = _prepare_sample_pool(
        manifest_path, h5_path, region_tif,
        min_fire_pixels=min_fire_pixels,
    )

    # ── Split by fire type ──
    veg_pool = pairs[pairs["fire_type"] == FireType.VEGETATION].copy()
    static_pool = pairs[pairs["fire_type"] == FireType.STATIC].copy()
    crop_pool = pairs[pairs["fire_type"] == FireType.CROP].copy()
    print(f"  Pools: {len(veg_pool):,} veg, {len(static_pool):,} static, "
          f"{len(crop_pool):,} crop\n")

    # ── 1. Select vegetation (cheapest-first) ──
    veg_sel = _select_cheapest(
        veg_pool, fi, veg_per_region, cap_per_fire=cap_veg,
    )
    veg_supp = _find_supplemental(veg_sel, fi)
    print(f"  Vegetation: {len(veg_sel):,} pairs, "
          f"{veg_sel['fire_id'].nunique():,} fires, "
          f"{len(veg_supp):,} supp images "
          f"({len(veg_supp) * 100 / 1e6:.2f} TB)")

    # ── 2. Update coverage with veg supplemental downloads ──
    fi_updated = fi.copy()
    if not veg_supp.empty:
        supp_ids = set(veg_supp["image_id"].values)
        fi_updated.loc[
            fi_updated["image_id"].isin(supp_ids), "downloaded",
        ] = True

    # ── 3. Select static (piggybacking on veg downloads) ──
    if len(static_pool) and static_per_region > 0:
        static_pool = _recompute_coverage(static_pool, fi_updated)
        static_sel = _select_cheapest(
            static_pool, fi_updated, static_per_region,
            cap_per_fire=cap_static,
        )
        static_supp = _find_supplemental(static_sel, fi_updated)
    else:
        static_sel = pairs.iloc[:0]
        static_supp = pd.DataFrame(
            columns=["image_id", "file_name", "prefix", "image_date"],
        )

    # Update coverage with static supplemental too
    if not static_supp.empty:
        supp_ids2 = set(static_supp["image_id"].values)
        fi_updated.loc[
            fi_updated["image_id"].isin(supp_ids2), "downloaded",
        ] = True

    print(f"  Static:     {len(static_sel):,} pairs, "
          f"{static_sel['fire_id'].nunique():,} fires, "
          f"{len(static_supp):,} extra supp "
          f"({len(static_supp) * 100 / 1e6:.2f} TB)")

    # ── 4. Select crop (piggybacking on veg + static downloads) ──
    if len(crop_pool) and crop_per_region > 0:
        crop_pool = _recompute_coverage(crop_pool, fi_updated)
        crop_sel = _select_cheapest(
            crop_pool, fi_updated, crop_per_region,
            cap_per_fire=cap_crop,
        )
        crop_supp = _find_supplemental(crop_sel, fi_updated)
    else:
        crop_sel = pairs.iloc[:0]
        crop_supp = pd.DataFrame(
            columns=["image_id", "file_name", "prefix", "image_date"],
        )

    print(f"  Crop:       {len(crop_sel):,} pairs, "
          f"{crop_sel['fire_id'].nunique():,} fires, "
          f"{len(crop_supp):,} extra supp "
          f"({len(crop_supp) * 100 / 1e6:.2f} TB)")

    # ── 5. Combined report ──
    selected = pd.concat(
        [veg_sel, static_sel, crop_sel], ignore_index=True,
    )
    all_supp = pd.concat(
        [veg_supp, static_supp, crop_supp], ignore_index=True,
    ).drop_duplicates(subset=["image_id"])
    n_total_supp = len(all_supp)

    print(f"\n  ── Combined ──")
    print(f"  {len(selected):,} pairs, "
          f"{selected['fire_id'].nunique():,} fires")
    print(f"  Supplemental: {n_total_supp:,} images "
          f"({n_total_supp * 100 / 1e6:.2f} TB)")

    # Per-region table
    regions = sorted(r for r in selected["region_id"].unique() if r > 0)
    print(f"\n  {'Region':<12} {'Veg':>7} {'Static':>7} {'Crop':>7} "
          f"{'Total':>7} {'Fires':>7}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 7} "
          f"{'-' * 7} {'-' * 7}")
    for rid in regions:
        rname = REGION_NAMES.get(int(rid), f"R{rid}")
        rv = veg_sel[veg_sel["region_id"] == rid]
        rs = static_sel[static_sel["region_id"] == rid]
        rc = crop_sel[crop_sel["region_id"] == rid]
        ra = selected[selected["region_id"] == rid]
        print(f"  {rname:<12} {len(rv):>7,} {len(rs):>7,} {len(rc):>7,} "
              f"{len(ra):>7,} {ra['fire_id'].nunique():>7,}")
    ra = selected
    print(f"  {'TOTAL':<12} {len(veg_sel):>7,} {len(static_sel):>7,} "
          f"{len(crop_sel):>7,} {len(ra):>7,} "
          f"{ra['fire_id'].nunique():>7,}")
    print()

    # ── Save ──
    out_path = os.path.join(config.data_dir, "next_day_v3", "samples.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    records = [
        {
            "fire_id": int(r.fire_id),
            "day_T": r.day_T.isoformat(),
            "region_id": int(r.region_id),
            "num_fire": int(r.num_fire),
            "n_tiles_est": int(r.n_tiles_est),
            "t_full": bool(r.t_full),
            "t1_full": bool(r.t1_full),
            "fire_type": int(r.fire_type),
        }
        for r in selected.itertuples()
    ]
    meta = {
        "num_samples": len(records),
        "seed": seed,
        "veg_per_region": veg_per_region,
        "static_per_region": static_per_region,
        "crop_per_region": crop_per_region,
        "cap_veg": cap_veg,
        "cap_static": cap_static,
        "cap_crop": cap_crop,
        "min_fire_pixels": min_fire_pixels,
        "samples": records,
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {out_path}")
    return selected, all_supp


def sweep_pick_samples(
    manifest_path: str, h5_path: str, region_tif: str,
    num_samples: int = 50_000,
    cap_per_fire: int = 50,
    seed: int = 42,
    weight_exps: list[float] | None = None,
    min_fire_pixels: int = 0,
) -> None:
    """Run pick_samples with multiple weight_exp values and compare.

    Loads the pool once, then runs selection for each weight_exp.
    Prints a comparison table with the key metrics.
    """
    from firecomp.core.fire_filter import FireType, fire_type_label

    if weight_exps is None:
        weight_exps = [0.0, 0.25, 0.5, 0.75, 1.0]

    pairs, fi = _prepare_sample_pool(
        manifest_path, h5_path, region_tif,
        min_fire_pixels=min_fire_pixels,
    )

    # Collect results
    results = []
    for we in weight_exps:
        sel = _select_from_pool(
            pairs, num_samples, cap_per_fire, seed, we,
        )
        n_supp = _count_supplemental(sel, fi)
        nf = sel["num_fire"].values
        p25, p50, p75 = np.percentile(nf, [25, 50, 75])
        n_both = int((sel["t_full"] & sel["t1_full"]).sum())

        results.append({
            "weight_exp": we,
            "n_samples": len(sel),
            "n_fires": sel["fire_id"].nunique(),
            "n_veg": int((sel["fire_type"] == FireType.VEGETATION).sum()),
            "n_static": int((sel["fire_type"] == FireType.STATIC).sum()),
            "n_crop": int((sel["fire_type"] == FireType.CROP).sum()),
            "p25": p25, "p50": p50, "p75": p75,
            "n_full": n_both,
            "n_supp_imgs": n_supp,
            "supp_tb": n_supp * 100 / 1e6,
        })

    # ── Comparison table ──
    print("\n" + "=" * 90)
    print("  SWEEP: weight_exp comparison")
    print("=" * 90)
    print(f"  {'wt_exp':>6}  {'samples':>7}  {'fires':>7}  "
          f"{'veg':>7}  {'static':>7}  {'crop':>6}  "
          f"{'P25':>6}  {'P50':>7}  {'P75':>8}  "
          f"{'supp_imgs':>9}  {'~TB':>5}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*6}  "
          f"{'-'*6}  {'-'*7}  {'-'*8}  "
          f"{'-'*9}  {'-'*5}")
    for r in results:
        print(f"  {r['weight_exp']:>6.2f}  {r['n_samples']:>7,}  "
              f"{r['n_fires']:>7,}  "
              f"{r['n_veg']:>7,}  {r['n_static']:>7,}  "
              f"{r['n_crop']:>6,}  "
              f"{r['p25']:>6,.0f}  {r['p50']:>7,.0f}  "
              f"{r['p75']:>8,.0f}  "
              f"{r['n_supp_imgs']:>9,}  {r['supp_tb']:>5.2f}")
    print()

    # Per-region P50 breakdown for each weight
    print(f"  {'Region':<12}", end="")
    for r in results:
        print(f"  P50@{r['weight_exp']:.2f}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in results:
        print(f"  {'-'*8}", end="")
    print()
    regions = sorted(r for r in pairs["region_id"].unique() if r > 0)
    # Re-run selections for per-region table
    sels = []
    for we in weight_exps:
        sels.append(_select_from_pool(
            pairs, num_samples, cap_per_fire, seed, we,
        ))
    for rid in regions:
        rname = REGION_NAMES.get(int(rid), f"R{rid}")
        print(f"  {rname:<12}", end="")
        for sel in sels:
            rg = sel[sel["region_id"] == rid]
            p50 = np.median(rg["num_fire"].values) if len(rg) else 0
            print(f"  {p50:>8,.0f}", end="")
        print()
    print()


def _count_supp_images(sub: pd.DataFrame, fi: pd.DataFrame) -> int:
    """Count unique VNP03 images not yet downloaded for a set of pairs."""
    if sub.empty:
        return 0
    need_t = sub[["fire_id", "day_T"]].rename(columns={"day_T": "image_date"})
    need_t1 = sub[["fire_id", "day_T"]].copy()
    need_t1["image_date"] = need_t1["day_T"] + pd.Timedelta(days=1)
    need_t1 = need_t1.drop(columns=["day_T"])
    need = pd.concat([need_t, need_t1]).drop_duplicates()
    needed = fi.merge(need, on=["fire_id", "image_date"])
    return needed.loc[~needed["downloaded"], "image_id"].nunique()


# Per-region vegetation pair budgets.  Derived from v2 dataset yield:
# regions with more cloud loss / preprocessing attrition get more pairs
# so that ~10K vegetation samples survive after build_lossmask filtering.
#
#   v2 yield     budget (pairs to pick)
#   ----------   ----------------------
#   C.NA  5,019  20,000   (50% survival → need 2×)
#   W.Eur 5,916  17,000
#   MENA  6,632  16,000
#   N.Asia6,772  15,000
#   Oce   6,895  15,000
#   S.Asia7,358  14,000
#   N.NA  8,613  12,000
#   S.Amer9,162  11,000
#   Africa10,032 10,000
#   E.Eur 3,219  15,000   (low yield is fire quality, not just clouds)
VEG_BUDGET: dict[int, int] = {
    1: 17_000,   # W.Europe
    2: 16_000,   # MENA
    3: 10_000,   # Africa
    4: 15_000,   # N.Asia
    5: 14_000,   # S.Asia
    6: 15_000,   # Oceania
    7: 12_000,   # N.NA
    8: 20_000,   # C.NA
    9: 11_000,   # S.America
    10: 15_000,  # E.Europe
}

STATIC_BUDGET = 2_000   # per region, on top of veg
CROP_BUDGET = 2_000     # per region, on top of veg


def _select_cheapest(pool: pd.DataFrame, fi: pd.DataFrame,
                     per_region: int | dict[int, int],
                     cap_per_fire: int = 50,
                     weight_exp: float = 0.5,
                     ) -> pd.DataFrame:
    """Select up to *per_region* pairs per region.

    per_region can be a flat int or a dict mapping region_id → budget.

    Uses Efraimidis-Spirakis weighted sampling: ``rand() / weight``
    where ``weight = num_fire^weight_exp * cost_mult``.  Larger fires
    get proportionally more representation (sqrt by default), and
    already-downloaded pairs get a mild boost (2× for both-full vs
    neither-full) — but fire size can easily overcome the cost penalty.
    Caps per-fire to avoid one fire dominating a region.
    """
    rng = np.random.RandomState(42)
    regions = sorted(r for r in pool["region_id"].unique() if r > 0)
    budgets = (per_region if isinstance(per_region, dict)
               else {r: per_region for r in regions})

    pool = pool.copy()

    # Fire-size weight
    w = np.maximum(pool["num_fire"].values, 1).astype(float)
    if weight_exp > 0:
        w = w ** weight_exp

    # Soft cost multiplier: mild preference for already-covered pairs.
    # both full → 1.0,  T full → 0.7,  neither → 0.5
    both_full = pool["t_full"].values & pool["t1_full"].values
    t_full = pool["t_full"].values
    cost_mult = np.where(both_full, 1.0, np.where(t_full, 0.7, 0.5))
    w = w * cost_mult

    # Efraimidis-Spirakis key: rand / weight → lower = picked first
    pool["_wrand"] = rng.random(len(pool)) / w

    # Cap per fire first (keep best-scoring pairs per fire)
    pool = pool.sort_values(["fire_id", "_wrand"])
    pool["_frank"] = pool.groupby("fire_id").cumcount()
    pool = pool[pool["_frank"] < cap_per_fire]

    parts = []
    for rid in regions:
        rp = pool[pool["region_id"] == rid].copy()
        rp = rp.sort_values("_wrand")
        budget = budgets.get(rid, 10_000)
        parts.append(rp.head(budget))

    if not parts:
        return pool.iloc[:0]
    sel = pd.concat(parts, ignore_index=True)
    return sel.drop(columns=["_wrand", "_frank"], errors="ignore")


def survey_pool(
    manifest_path: str, h5_path: str, region_tif: str,
    solidity_json: str | None = None,
    per_region: int = 0,
) -> None:
    """Survey the sample pool with combined filters and download budget.

    For each filter scenario:
    1. Select vegetation pairs up to VEG_BUDGET per region (variable,
       oversamples cloudy regions to target ~10K final samples).
    2. Add up to STATIC_BUDGET + CROP_BUDGET per region on top,
       preferring already-covered pairs.
    3. Report fires, pairs, download cost.

    If --per-region > 0, uses that flat number instead of VEG_BUDGET
    (and skips separate static/crop selection).
    """
    from firecomp.core.fire_filter import FireType, fire_type_label

    pairs, fi = _prepare_sample_pool(manifest_path, h5_path, region_tif)
    regions = sorted(r for r in pairs["region_id"].unique() if r > 0)

    use_variable = per_region <= 0
    veg_budgets = VEG_BUDGET if use_variable else {r: per_region for r in regions}

    # ── Load solidity/spread data ──
    sol_map: dict[int, dict] = {}
    if solidity_json:
        with open(solidity_json) as f:
            for entry in json.load(f):
                sol_map[entry["fire_id"]] = entry
        print(f"  Loaded solidity data for {len(sol_map):,} fires")

    if sol_map:
        pairs["solidity"] = pairs["fire_id"].map(
            lambda fid: sol_map.get(fid, {}).get("solidity", np.nan))
        pairs["spread_ratio"] = pairs["fire_id"].map(
            lambda fid: sol_map.get(fid, {}).get("spread_ratio", np.nan))
    else:
        pairs["solidity"] = np.nan
        pairs["spread_ratio"] = np.nan

    # Separate pools by fire type
    veg_pairs = pairs[pairs["fire_type"] == FireType.VEGETATION]
    static_pairs = pairs[pairs["fire_type"] == FireType.STATIC]
    crop_pairs = pairs[pairs["fire_type"] == FireType.CROP]

    # ── Define filter scenarios (applied to vegetation only) ──
    veg_scenarios: list[tuple[str, pd.DataFrame]] = []

    pixel_thresholds = [0, 10, 25, 50]
    for px in pixel_thresholds:
        sub = veg_pairs[veg_pairs["num_fire"] >= px] if px > 0 else veg_pairs
        veg_scenarios.append((f"px≥{px}", sub))

    if sol_map:
        for sol_th in [0.3, 0.5, 0.7]:
            sub = veg_pairs[(veg_pairs["num_fire"] >= 10) &
                            (veg_pairs["solidity"] >= sol_th)]
            veg_scenarios.append((f"px≥10+sol≥{sol_th}", sub))

        for sp_th in [0.3, 0.5, 0.7]:
            sub = veg_pairs[(veg_pairs["num_fire"] >= 10) &
                            (veg_pairs["spread_ratio"] >= sp_th)]
            veg_scenarios.append((f"px≥10+spr≥{sp_th}", sub))

        sub = veg_pairs[(veg_pairs["num_fire"] >= 10) &
                        (veg_pairs["solidity"] >= 0.5) &
                        (veg_pairs["spread_ratio"] >= 0.5)]
        veg_scenarios.append(("px≥10+sol≥.5+spr≥.5", sub))

    # ── Select cheapest veg pairs + add static/crop on top ──
    selected: list[tuple[str, pd.DataFrame]] = []
    for label, veg_pool in veg_scenarios:
        veg_sel = _select_cheapest(veg_pool, fi, veg_budgets)
        if use_variable:
            # Add static + crop with px≥10 and their own budgets
            st_pool = static_pairs[static_pairs["num_fire"] >= 10]
            cr_pool = crop_pairs[crop_pairs["num_fire"] >= 10]
            st_sel = _select_cheapest(st_pool, fi, STATIC_BUDGET)
            cr_sel = _select_cheapest(cr_pool, fi, CROP_BUDGET)
            sel = pd.concat([veg_sel, st_sel, cr_sel], ignore_index=True)
        else:
            sel = veg_sel
        selected.append((label, sel))

    # ── Print tables ──
    col_w = 18
    total_w = 14 + col_w * len(selected)
    budget_label = "variable VEG_BUDGET" if use_variable else f"{per_region:,}/region"
    print(f"\n{'=' * total_w}")
    print(f"  SURVEY: {budget_label} + "
          f"{STATIC_BUDGET:,} static + {CROP_BUDGET:,} crop/region, "
          f"cheapest-first")
    print(f"{'=' * total_w}")

    if use_variable:
        print(f"\n  Veg budgets: ", end="")
        for rid in regions:
            rname = REGION_NAMES.get(rid, f"R{rid}")
            print(f"{rname}={veg_budgets.get(rid, 10_000):,}  ", end="")
        print()

    # Header
    print(f"\n  {'Region':<12}", end="")
    for label, _ in selected:
        print(f"  {label:>{col_w - 2}}", end="")
    print()
    print(f"  {'-' * 12}", end="")
    for _ in selected:
        print(f"  {'-' * (col_w - 2)}", end="")
    print()

    # Per-region: fires / pairs
    for rid in [None] + regions:
        if rid is None:
            rname = "TOTAL"
        else:
            rname = REGION_NAMES.get(int(rid), f"R{rid}")

        print(f"  {rname:<12}", end="")
        for _, sel in selected:
            rsub = sel if rid is None else sel[sel["region_id"] == rid]
            n_fires = rsub["fire_id"].nunique()
            n_pairs = len(rsub)
            print(f"  {n_fires:>5}f {n_pairs:>6}p", end="")
        print()

    # Pool size (total veg available before budget cap)
    print()
    print(f"  {'Veg pool':<12}", end="")
    for label, pool in veg_scenarios:
        print(f"  {len(pool):>{col_w - 2},}", end="")
    print()

    # Download cost
    print(f"  {'VNP03 supp':<12}", end="")
    for _, sel in selected:
        sc = _count_supp_images(sel, fi)
        est_tb = sc * 100 / 1e6
        print(f"  {sc:>8,} ({est_tb:.1f}T)", end="")
    print()

    # Both-covered
    print(f"  {'Both full':<12}", end="")
    for _, sel in selected:
        n_full = int((sel["t_full"] & sel["t1_full"]).sum()) if len(sel) else 0
        print(f"  {n_full:>{col_w - 2},}", end="")
    print()

    # Fire type breakdown
    print()
    for ft in FireType:
        label = fire_type_label(ft)
        print(f"  {label:<12}", end="")
        for _, sel in selected:
            n = int((sel["fire_type"] == ft).sum())
            print(f"  {n:>{col_w - 2},}", end="")
        print()

    print()


# ── Coverage analysis ─────────────────────────────────────────────────


def analyze_coverage(
    manifest_path: str, h5_path: str, region_tif: str,
) -> pd.DataFrame:
    """Check which fire-day pairs have both T and T+1 coverage.

    Loads the manifest (downloaded image_ids) and the HDF5 fire database,
    then finds:
    - fire-days where both T and T+1 images are in the downloaded set
    - fire-days where only T is covered (need supplemental T+1 download)
    - per-region breakdown

    Returns DataFrame of valid next-day sample pairs with columns:
    fire_id, day_T, region_id, has_T, has_T1, covered.
    """
    t0 = time.time()
    manifest = _load_manifest(manifest_path)
    downloaded_ids = set(manifest["image_id"].values)

    h5 = h5py.File(h5_path, "r")
    fires_df = _load_fires(h5, region_tif)
    fire_image = _build_fire_image(h5, fires_df["fire_id"].values)

    # Get image dates
    img_ids = h5["images/id"][:]
    img_times = h5["images/start_time"][:].astype("datetime64[ms]")
    h5.close()

    img_dates = pd.DataFrame({
        "image_id": img_ids,
        "image_date": pd.to_datetime(img_times).normalize(),
    })

    # Join fire_image with dates → fire-day-image triples
    fi = fire_image.merge(img_dates, on="image_id")
    fi = fi.merge(fires_df[["fire_id", "region_id"]], on="fire_id")

    # Mark which images are in our downloaded set
    fi["downloaded"] = fi["image_id"].isin(downloaded_ids)

    # For each (fire_id, date): is there at least one downloaded image?
    fire_day_coverage = (
        fi.groupby(["fire_id", "image_date", "region_id"])["downloaded"]
        .any()
        .reset_index()
    )
    fire_day_coverage.columns = [
        "fire_id", "day", "region_id", "has_coverage",
    ]

    # Find consecutive day pairs (T, T+1)
    fdc = fire_day_coverage.copy()
    fdc["day_plus1"] = fdc["day"] + pd.Timedelta(days=1)

    # Self-join: T row matched with T+1 row for same fire
    pairs = fdc.merge(
        fdc[["fire_id", "day", "has_coverage"]],
        left_on=["fire_id", "day_plus1"],
        right_on=["fire_id", "day"],
        suffixes=("_T", "_T1"),
    )
    pairs = pairs.rename(columns={
        "day_T": "day_T",
        "has_coverage_T": "has_T",
        "has_coverage_T1": "has_T1",
    })
    pairs["covered"] = pairs["has_T"] & pairs["has_T1"]

    dt = time.time() - t0

    # ── Summary ──
    n_pairs = len(pairs)
    n_covered = int(pairs["covered"].sum())
    n_t_only = int((pairs["has_T"] & ~pairs["has_T1"]).sum())
    n_t1_only = int((~pairs["has_T"] & pairs["has_T1"]).sum())
    n_neither = int((~pairs["has_T"] & ~pairs["has_T1"]).sum())
    n_fires_covered = pairs.loc[pairs["covered"], "fire_id"].nunique()

    print(f"\n  Coverage analysis ({dt:.0f}s):")
    print(f"    Total fire-day pairs (T, T+1): {n_pairs:,}")
    print(f"    Both T and T+1 covered:        {n_covered:,} "
          f"({100*n_covered/max(n_pairs,1):.1f}%)")
    print(f"    Only T covered (need T+1):     {n_t_only:,}")
    print(f"    Only T+1 covered:              {n_t1_only:,}")
    print(f"    Neither covered:               {n_neither:,}")
    print(f"    Fires with >=1 covered pair:   {n_fires_covered:,}")

    # Per-region breakdown
    print(f"\n  {'Region':<12} {'Pairs':>8} {'Covered':>8} {'%':>6} "
          f"{'Need T+1':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for rid in sorted(pairs["region_id"].unique()):
        if rid <= 0:
            continue
        rg = pairs[pairs["region_id"] == rid]
        rc = int(rg["covered"].sum())
        rt = int((rg["has_T"] & ~rg["has_T1"]).sum())
        rname = REGION_NAMES.get(int(rid), f"R{rid}")
        pct = 100 * rc / max(len(rg), 1)
        print(f"  {rname:<12} {len(rg):>8,} {rc:>8,} {pct:>5.1f}% "
              f"{rt:>8,}")
    print()

    return pairs


def find_supplemental(
    pairs: pd.DataFrame, h5_path: str,
) -> pd.DataFrame:
    """Find VNP03IMG images needed to cover T+1 gaps.

    For fire-day pairs where T is covered but T+1 is not, look up
    which images observed the fire on T+1 and return them as a
    supplemental download list.

    Returns DataFrame with columns: image_id, fire_id, day_T1,
    region_id, file_name, prefix, date.
    """
    # Pairs where we have T but need T+1
    gaps = pairs[pairs["has_T"] & ~pairs["has_T1"]].copy()
    if gaps.empty:
        print("  No supplemental downloads needed — full coverage!")
        return pd.DataFrame()

    h5 = h5py.File(h5_path, "r")

    # Build fire_image links (reuse existing helper)
    fire_ids = h5["stats"]["id"][:]
    fire_image = _build_fire_image(h5, fire_ids)

    # Get image metadata
    img_ids = h5["images/id"][:]
    img_times = h5["images/start_time"][:].astype("datetime64[ms]")
    fnames_raw = h5["images/file_name"][:]
    fnames = [
        f.decode() if isinstance(f, bytes) else str(f) for f in fnames_raw
    ]
    h5.close()

    img_df = pd.DataFrame({
        "image_id": img_ids,
        "file_name": fnames,
        "image_date": pd.to_datetime(img_times).normalize(),
    })

    # Join fire_image with image dates
    fi = fire_image.merge(img_df, on="image_id")

    # For each gap: find images that cover this fire on day T+1
    gaps_key = gaps[["fire_id", "day_plus1", "region_id"]].copy()
    gaps_key = gaps_key.rename(columns={"day_plus1": "image_date"})

    supplemental = fi.merge(
        gaps_key, on=["fire_id", "image_date"],
    )

    # Deduplicate: one row per unique image needed
    supplemental = (
        supplemental
        .drop_duplicates(subset=["image_id"])
        .copy()
    )
    supplemental["prefix"] = supplemental["file_name"].apply(
        viirs_granule_prefix,
    )
    supplemental["date"] = supplemental["image_date"].dt.strftime("%Y-%m-%d")

    # How many gap pairs would this cover?
    covered_fires_days = supplemental[["fire_id", "image_date"]].drop_duplicates()
    n_gaps_coverable = len(gaps_key.merge(
        covered_fires_days, on=["fire_id", "image_date"],
    ))

    print(f"  Supplemental images needed: {len(supplemental):,}")
    print(f"  Unique prefixes: {supplemental['prefix'].nunique():,}")
    supp_tb = supplemental["prefix"].nunique() * 100 / 1e6
    print(f"  Estimated download: ~{supp_tb:.1f} TB")
    print(f"  Would cover {n_gaps_coverable:,} / {len(gaps):,} "
          f"T+1 gaps")

    return supplemental


# ── CLI ──────────────────────────────────────────────────────────────


def _resolve_paths(args):
    """Resolve H5 and region TIF paths from args or defaults."""
    h5 = args.h5 or config.vnp14_path
    tif = args.region_tif or os.path.join(
        config.data_dir, "regions", "wildfire_regions.tif",
    )
    return h5, tif


def _load_manifest(path: str) -> pd.DataFrame:
    """Load a previously saved manifest JSON into a DataFrame."""
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["images"])
    print(f"  Loaded manifest: {path}")
    top = data.get("top_per_region")
    if top:
        print(f"    {len(df):,} images, top {top}/region")
    else:
        print(f"    {len(df):,} images")
    return df


def fix_invalid_files(output_dir: str, top: int) -> None:
    """Validate VNP03IMG files referenced in manifests, delete corrupt ones.

    Tries to open each .nc file with h5py.  Files that fail to open are
    deleted, then ``download_vnp03`` is called to re-download them.
    """
    # ── Load both manifests ──
    primary = os.path.join(output_dir, f"manifest_top{top}.json")
    supplemental = os.path.join(output_dir, f"manifest_top{top}_supplemental.json")

    manifest_paths = [p for p in [primary, supplemental] if os.path.exists(p)]
    if not manifest_paths:
        print(f"  No manifests found for --top {top} in {output_dir}")
        return

    all_images = []
    referenced_prefixes = set()
    for path in manifest_paths:
        with open(path) as f:
            data = json.load(f)
        images = data.get("images", [])
        all_images.extend(images)
        for img in images:
            pfx = img.get("prefix")
            if not pfx and "file_name" in img:
                pfx = viirs_granule_prefix(img["file_name"])
            if pfx:
                referenced_prefixes.add(pfx)
        print(f"  {path}: {len(images):,} images")
    print(f"  Total referenced prefixes: {len(referenced_prefixes):,}")

    # ── Find referenced files on disk ──
    prefix_to_file: dict[str, str] = {}
    for fname in os.listdir(output_dir):
        if fname.startswith("VNP03IMG") and fname.endswith(".nc"):
            pfx = viirs_granule_prefix(fname)
            if pfx and pfx in referenced_prefixes:
                prefix_to_file[pfx] = os.path.join(output_dir, fname)

    print(f"  Referenced files on disk: {len(prefix_to_file):,}")
    missing = len(referenced_prefixes) - len(prefix_to_file)
    if missing:
        print(f"  Missing (not on disk):    {missing:,}")

    # ── Validate each file ──
    invalid_paths = []
    for pfx, path in sorted(prefix_to_file.items()):
        try:
            f = h5py.File(path, "r")
            # Quick sanity: must have at least one group
            if len(f.keys()) == 0:
                raise ValueError("empty HDF5")
            f.close()
        except Exception as e:
            size_mb = os.path.getsize(path) / 1e6
            print(f"    INVALID: {os.path.basename(path)} "
                  f"({size_mb:.0f} MB) — {e}")
            invalid_paths.append(path)

    print(f"\n  Valid:   {len(prefix_to_file) - len(invalid_paths):,}")
    print(f"  Invalid: {len(invalid_paths):,}")

    if not invalid_paths:
        if missing:
            print(f"\n  No corrupt files, but {missing:,} are missing.")
            print(f"  Re-download with:")
            print(f"    python -m firecomp.next_day.download --top {top}")
        else:
            print("\n  All files valid!")
        return

    # ── Delete invalid files ──
    freed = 0
    for path in invalid_paths:
        freed += os.path.getsize(path)
        os.remove(path)
    print(f"  Deleted {len(invalid_paths):,} corrupt files "
          f"({freed / 1e9:.1f} GB freed)")

    # ── Re-download ──
    # Build a DataFrame matching download_vnp03's expected format
    invalid_prefixes = {viirs_granule_prefix(os.path.basename(p))
                        for p in invalid_paths}
    redownload = [img for img in all_images
                  if img.get("prefix") in invalid_prefixes]
    if redownload:
        redownload_df = pd.DataFrame(redownload)
        print(f"\n  Re-downloading {len(redownload_df):,} files...")
        download_vnp03(redownload_df, output_dir)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Download VNP03IMG files for next-day prediction.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--manifest", action="store_true",
        help="Create manifest (image selection) and exit. Run this first.",
    )
    mode.add_argument(
        "--pick", action="store_true",
        help="Pick balanced (fire, day) samples from downloaded images.",
    )
    mode.add_argument(
        "--sweep", action="store_true",
        help="Sweep weight_exp values and compare sample distributions.",
    )
    mode.add_argument(
        "--pick-supplemental", action="store_true",
        help="Pick samples, then find and download missing T+1 images.",
    )
    mode.add_argument(
        "--analyze", action="store_true",
        help="Analyze T/T+1 coverage of the manifest. Run after download.",
    )
    mode.add_argument(
        "--supplemental", action="store_true",
        help="Find and download supplemental T+1 images for coverage gaps.",
    )
    mode.add_argument(
        "--survey-pool", action="store_true",
        help="Survey sample pool at various min_fire_pixels thresholds. "
             "No selection or download — just counts.",
    )
    mode.add_argument(
        "--fix-invalid", action="store_true",
        help="Validate referenced VNP03IMG files on disk, delete corrupt "
             "ones, and re-download them.",
    )
    parser.add_argument(
        "--manifest-path", default=None,
        help="Path to manifest JSON. Download jobs load from this instead "
             "of recomputing the selection.",
    )
    parser.add_argument(
        "--top", type=int, default=1000,
        help="Top N images per region to select (default: 1000)",
    )
    parser.add_argument(
        "--veg-per-region", type=int, default=10_000,
        help="Target vegetation pairs per region for --pick (default: 10000)",
    )
    parser.add_argument(
        "--static-per-region", type=int, default=2_000,
        help="Target static pairs per region for --pick (default: 2000)",
    )
    parser.add_argument(
        "--crop-per-region", type=int, default=2_000,
        help="Target crop pairs per region for --pick (default: 2000)",
    )
    parser.add_argument(
        "--cap-veg", type=int, default=500,
        help="Max sample days per vegetation fire (default: 500)",
    )
    parser.add_argument(
        "--cap-static", type=int, default=50,
        help="Max sample days per static fire (default: 50)",
    )
    parser.add_argument(
        "--cap-crop", type=int, default=50,
        help="Max sample days per crop fire (default: 50)",
    )
    # Legacy args kept for --sweep compatibility
    parser.add_argument(
        "--num-samples", type=int, default=50_000,
        help="Target pairs for --sweep (default: 50000)",
    )
    parser.add_argument(
        "--cap-per-fire", type=int, default=50,
        help="Cap per fire for --sweep (default: 50)",
    )
    parser.add_argument(
        "--weight-exp", type=float, default=0.5,
        help="Exponent for num_fire weighting in --sweep (default: 0.5)",
    )
    parser.add_argument(
        "--min-fire-pixels", type=int, default=10,
        help="Drop fires with fewer total pixels (default: 10). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "-ji", type=int, default=0,
        help="Job index for distributed execution (0-based)",
    )
    parser.add_argument(
        "-jn", type=int, default=1,
        help="Total number of jobs (default: 1 = no distribution)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {config.vnp03img_dir})",
    )
    parser.add_argument("--h5", help="Path to vnp14_fires HDF5")
    parser.add_argument("--region-tif", help="Path to wildfire_regions.tif")
    parser.add_argument(
        "--solidity-json", default=None,
        help="Path to solidity survey JSON (from solidity survey command). "
             "Used by --survey-pool to cross-reference solidity/spread.",
    )
    parser.add_argument(
        "--per-region", type=int, default=0,
        help="Flat veg budget per region for --survey-pool. "
             "0 (default) uses variable VEG_BUDGET targeting ~10K final samples.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or config.vnp03img_dir
    default_manifest = os.path.join(
        output_dir, f"manifest_top{args.top}.json",
    )

    # ── Create manifest mode ──
    if args.manifest:
        h5_path, region_tif = _resolve_paths(args)
        for path, label in [(h5_path, "H5"), (region_tif, "Region TIF")]:
            if not os.path.exists(path):
                sys.exit(f"ERROR: {label} not found: {path}")

        print(f"H5:  {h5_path}")
        print(f"TIF: {region_tif}")
        print(f"Out: {output_dir}")
        print(f"Top: {args.top}/region\n")

        selected = select_top_images(h5_path, region_tif, args.top)

        # Per-region summary
        print(f"\n  {'Region':<12} {'Images':>8} {'Max px':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*10}")
        for rid in sorted(selected["region_id"].unique()):
            rname = REGION_NAMES.get(int(rid), f"R{rid}")
            grp = selected[selected["region_id"] == rid]
            print(f"  {rname:<12} {len(grp):>8,} "
                  f"{grp['num_fire_pixels'].max():>10,}")
        total_tb = len(selected) * 100 / 1e6
        print(f"\n  Total: {len(selected):,} images ~ {total_tb:.1f} TB\n")

        save_manifest(selected, output_dir, args.top)
        return

    # ── All other modes require a manifest ──
    manifest_path = args.manifest_path or default_manifest
    if not os.path.exists(manifest_path):
        sys.exit(
            f"ERROR: Manifest not found: {manifest_path}\n"
            f"Create it first:  python -m firecomp.next_day.download "
            f"--manifest --top {args.top}"
        )

    # ── Fix-invalid mode ──
    if args.fix_invalid:
        fix_invalid_files(output_dir, args.top)
        return

    # ── Survey pool mode ──
    if args.survey_pool:
        h5_path, region_tif = _resolve_paths(args)
        for path, label in [(h5_path, "H5"), (region_tif, "Region TIF")]:
            if not os.path.exists(path):
                sys.exit(f"ERROR: {label} not found: {path}")

        survey_pool(manifest_path, h5_path, region_tif,
                    solidity_json=args.solidity_json,
                    per_region=args.per_region)
        return

    # ── Sweep mode ──
    if args.sweep:
        h5_path, region_tif = _resolve_paths(args)
        for path, label in [(h5_path, "H5"), (region_tif, "Region TIF")]:
            if not os.path.exists(path):
                sys.exit(f"ERROR: {label} not found: {path}")

        sweep_pick_samples(
            manifest_path, h5_path, region_tif,
            num_samples=args.num_samples,
            cap_per_fire=args.cap_per_fire,
            min_fire_pixels=args.min_fire_pixels,
        )
        return

    # ── Pick mode ──
    if args.pick or args.pick_supplemental:
        h5_path, region_tif = _resolve_paths(args)
        for path, label in [(h5_path, "H5"), (region_tif, "Region TIF")]:
            if not os.path.exists(path):
                sys.exit(f"ERROR: {label} not found: {path}")

        picked, supp = pick_samples(
            manifest_path, h5_path, region_tif,
            veg_per_region=args.veg_per_region,
            static_per_region=args.static_per_region,
            crop_per_region=args.crop_per_region,
            cap_veg=args.cap_veg,
            cap_static=args.cap_static,
            cap_crop=args.cap_crop,
            min_fire_pixels=args.min_fire_pixels,
        )

        if args.pick_supplemental and not supp.empty:
            supp_path = os.path.join(
                output_dir,
                f"manifest_top{args.top}_supplemental.json",
            )
            supp["date"] = supp["image_date"].dt.strftime("%Y-%m-%d")
            supp_list = [
                {"image_id": int(r.image_id), "prefix": r.prefix,
                 "date": r.date, "file_name": r.file_name}
                for r in supp.itertuples()
            ]
            with open(supp_path, "w") as f:
                json.dump({"images": supp_list,
                           "total_images": len(supp_list)}, f, indent=2)
            print(f"  Supplemental manifest: {supp_path}")
            print(f"\n  Download supplemental with:")
            print(f"    python -m firecomp.next_day.download "
                  f"--manifest-path {supp_path}")
        return

    # ── Analyze mode ──
    if args.analyze:
        h5_path, region_tif = _resolve_paths(args)
        for path, label in [(h5_path, "H5"), (region_tif, "Region TIF")]:
            if not os.path.exists(path):
                sys.exit(f"ERROR: {label} not found: {path}")
        analyze_coverage(manifest_path, h5_path, region_tif)
        return

    # ── Supplemental mode ──
    if args.supplemental:
        h5_path, region_tif = _resolve_paths(args)
        for path, label in [(h5_path, "H5"), (region_tif, "Region TIF")]:
            if not os.path.exists(path):
                sys.exit(f"ERROR: {label} not found: {path}")

        pairs = analyze_coverage(manifest_path, h5_path, region_tif)
        supp = find_supplemental(pairs, h5_path)
        if not supp.empty:
            # Save supplemental manifest
            supp_manifest = os.path.join(
                output_dir,
                f"manifest_top{args.top}_supplemental.json",
            )
            supp_list = [
                {"image_id": int(r.image_id), "prefix": r.prefix,
                 "date": r.date, "file_name": r.file_name}
                for r in supp.itertuples()
            ]
            with open(supp_manifest, "w") as f:
                json.dump({"images": supp_list,
                           "total_images": len(supp_list)}, f, indent=2)
            print(f"  Supplemental manifest: {supp_manifest}")

            # Download
            my_supp = supp.iloc[args.ji::args.jn].copy()
            print(f"\n  Job {args.ji}/{args.jn}: downloading "
                  f"{len(my_supp):,} / {len(supp):,} supplemental\n")
            download_vnp03(my_supp, output_dir)
        print("\nDone.")
        return

    # ── Download mode: load manifest and download ──
    selected = _load_manifest(manifest_path)

    my_images = selected.iloc[args.ji::args.jn].copy()
    print(f"  Job {args.ji}/{args.jn}: {len(my_images):,} / "
          f"{len(selected):,} images\n")

    download_vnp03(my_images, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
