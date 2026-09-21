"""
next_day/preprocess.py — Build v3 dataset from manifest-based samples.

Output lives in data/next_day_v3/ (flat — no subdirectory).

Three build phases (each adds fields to the same H5 shards):
    python -m firecomp.next_day.preprocess build-accum    --dataset-dir data/next_day [--ji 0 --jn 8]
    python -m firecomp.next_day.preprocess build-lossmask --dataset-dir data/next_day [--ji 0 --jn 8]
    python -m firecomp.next_day.preprocess build-fields   --dataset-dir data/next_day [--ji 0 --jn 8]

Plus existing:
    python -m firecomp.next_day.preprocess patch    --dataset-dir data/next_day --source-version latest --target-version v2
    python -m firecomp.next_day.preprocess distill  --source data/next_day --dest data/next_day_small --n-fires 100

Workflow: download.py picks (fire_id, day_T) pairs and downloads VNP03IMG;
build phases materialize H5 shards; patch creates filtered dataset versions.

Phase 1 (build-accum): iterate raw VNP14 fire pixels day-by-day,
    accumulate accum_t_{min,max,count} on picked days.
Phase 2 (build-lossmask): project VNP14 fire/cloud masks for loss,
    derive cur_mask + next_mask + new_fires from geolocated projections.
Phase 3 (build-fields): crop AE embeddings, ERA5 weather, GFS forecasts.

Phase 1 must run first (creates H5 shards + sample JSONs).
Phases 2 and 3 are independent and run in any order after Phase 1.
All phases are resume-friendly: they skip samples/fields already written.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date as Date
from pathlib import Path

import numpy as np

from firecomp.core.cli import Cli
from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import ALL_FIELDS, Sample, load_all_samples


# =============================================================================
# PickedSample — input from download.py --pick
# =============================================================================

@dataclass
class PickedSample:
    """One (fire_id, day_T) pair from download.py --pick."""
    fire_id: int
    day_T: Date
    region_id: int
    num_fire: int
    n_tiles_est: int
    t_full: bool
    t1_full: bool
    fire_type: int


def load_picked_samples(path: Path) -> list[PickedSample]:
    """Read samples.json produced by download.py --pick."""
    with open(path) as f:
        data = json.load(f)
    out = []
    for r in data["samples"]:
        day = datetime.fromisoformat(r["day_T"]).date()
        out.append(PickedSample(
            fire_id=r["fire_id"], day_T=day, region_id=r["region_id"],
            num_fire=r["num_fire"], n_tiles_est=r["n_tiles_est"],
            t_full=r["t_full"], t1_full=r["t1_full"],
            fire_type=r["fire_type"],
        ))
    return out


# =============================================================================
# ShardWriter — H5 append-mode writer
# =============================================================================

class ShardWriter:
    """Write fields to a single H5 shard in append mode (zstd compressed)."""

    def __init__(self, path: Path):
        import h5py
        import hdf5plugin  # noqa: F401 — registers zstd filter
        from firecomp.dsrc.dsrc import DSRC
        path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(path, "a")
        self._storage_opts = DSRC.get_storage_opts

    def write(self, sample_key: str, field_name: str, data: np.ndarray):
        """Write a per-sample field.  Creates/overwrites."""
        grp = self._h5.require_group(sample_key)
        if field_name in grp:
            del grp[field_name]
        grp.create_dataset(field_name, data=data,
                           **self._storage_opts(data))

    def write_tile(self, field_name: str, tile_key: str, data: np.ndarray):
        """Write a per-tile field (shared across timesteps).  Skips if exists."""
        grp = self._h5.require_group(field_name)
        if tile_key not in grp:
            grp.create_dataset(tile_key, data=data,
                               **self._storage_opts(data))

    def has_field(self, sample_key: str, field_name: str) -> bool:
        return sample_key in self._h5 and field_name in self._h5[sample_key]

    def close(self):
        self._h5.close()


# =============================================================================
# Fire metadata from VNP14 stats
# =============================================================================

@dataclass
class _FireMeta:
    fire_id: int
    start_date: Date
    end_date: Date
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def _load_fire_metadata(h5, fire_ids: list[int]) -> dict[int, _FireMeta]:
    """Load bbox and date range for specific fires from VNP14 stats."""
    g = h5["stats"]
    all_ids = g["id"][:]
    id_to_idx = {int(fid): i for i, fid in enumerate(all_ids)}

    starts = g["start_date"][:].astype("datetime64[ms]")
    ends = g["end_date"][:].astype("datetime64[ms]")
    min_xs = g["min_x"][:].astype(np.float64)
    min_ys = g["min_y"][:].astype(np.float64)
    max_xs = g["max_x"][:].astype(np.float64)
    max_ys = g["max_y"][:].astype(np.float64)

    result = {}
    for fid in fire_ids:
        idx = id_to_idx.get(fid)
        if idx is None:
            print(f"  WARNING: fire {fid} not in VNP14 stats, skipping")
            continue
        result[fid] = _FireMeta(
            fire_id=fid,
            start_date=starts[idx].item().date(),
            end_date=ends[idx].item().date(),
            min_x=float(min_xs[idx]), min_y=float(min_ys[idx]),
            max_x=float(max_xs[idx]), max_y=float(max_ys[idx]),
        )
    return result


# =============================================================================
# VNP14 observation helpers
# =============================================================================

def _is_clear_obs(mask):
    """True where VNP14 fire_mask has a clear classification.

    Clear = water (3), land (5), or fire (7-9).
    Excludes: not-processed (0-2), cloud (4), unclassified (6), no-coverage (255).
    """
    return (mask == 3) | (mask == 5) | ((mask >= 7) & (mask <= 9))


# VNP14 observation priority for merging multiple passes:
#   tier 0 = no data (0,1,2)       — no information
#   tier 1 = cloud/unclassified (4,6) — satellite looked but can't see
#   tier 2 = clear land/water (3,5)   — real surface observation, no fire
#   tier 3 = fire (7,8,9)            — highest priority, always wins
_VNP14_PRIORITY = np.zeros(256, dtype=np.uint8)
_VNP14_PRIORITY[4] = 1
_VNP14_PRIORITY[6] = 1
_VNP14_PRIORITY[3] = 2
_VNP14_PRIORITY[5] = 2
_VNP14_PRIORITY[7] = 3
_VNP14_PRIORITY[8] = 3
_VNP14_PRIORITY[9] = 3


def _merge_vnp14_obs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge two VNP14 observation arrays by priority, vectorized.

    Higher-tier observations win.  Within the same tier, take the max
    value (e.g. high-confidence fire 9 beats nominal fire 7).
    Returns a new array (does not modify inputs).
    """
    pa = _VNP14_PRIORITY[a]
    pb = _VNP14_PRIORITY[b]
    out = a.copy()
    b_wins = (pb > pa) | ((pb == pa) & (b > a))
    out[b_wins] = b[b_wins]
    return out


# =============================================================================
# Bbox and tiling helpers
# =============================================================================

def _compute_padded_bbox(min_x, min_y, max_x, max_y,
                         crop_size, padding, deg_cell_size):
    """Compute pixel bbox padded for 256x256 tiling."""
    from firecomp.helpers import geod_to_pix_bbox
    x0, y0, x1, y1 = geod_to_pix_bbox(
        [min_x, min_y, max_x, max_y], deg_cell_size=deg_cell_size)
    x1 += 1; y1 += 1
    pad_x = (crop_size - ((x1 - x0) % crop_size)) % crop_size
    pad_y = (crop_size - ((y1 - y0) % crop_size)) % crop_size
    return (x0 - (pad_x // 2 + padding), y0 - (pad_y // 2 + padding),
            x1 + (pad_x - pad_x // 2 + padding),
            y1 + (pad_y - pad_y // 2 + padding))


def _pix_to_geod(bbox, deg_cell_size):
    """Pixel bbox → geographic [lon_min, lat_min, lon_max, lat_max]."""
    return [bbox[0] * deg_cell_size - 180,
            bbox[1] * deg_cell_size - 90,
            bbox[2] * deg_cell_size - 180,
            bbox[3] * deg_cell_size - 90]


_ae_cache: dict[tuple, "AEEmbeddings"] = {}

def _get_ae_cached(year: int, **kwargs) -> "AEEmbeddings":
    """Cached AEEmbeddings by (year, kwargs). Avoids re-creating per fire."""
    from firecomp.dsrc.ae_embedding import AEEmbeddings
    key = (year, tuple((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in sorted(kwargs.items())))
    if key not in _ae_cache:
        _ae_cache[key] = AEEmbeddings(year=year, **kwargs)
    return _ae_cache[key]


def _gap_fill_detections(y, x, fids, shape):
    """Fill sub-pixel gaps in daily fire detections via morphological closing.

    Each VNP14 detection covers more than one grid cell, but we place it as a
    single point.  This leaves 1-2 pixel gaps between detections that should be
    contiguous.  A binary close (dilate then erode) with a 3x3 kernel fills
    those gaps without expanding the fire boundary.  The result is unioned with
    the original mask so no real detection is ever lost.

    Gap-filled pixels inherit the fire_id of their nearest original detection
    (via distance transform).

    Returns (fill_y, fill_x, fill_fids) for the NEW gap-filled pixels only
    (excludes originals).  Empty arrays if nothing was filled.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    mask = np.zeros(shape, dtype=bool)
    mask[y, x] = True
    struct = np.ones((3, 3), dtype=bool)  # 8-connected
    closed = binary_erosion(binary_dilation(mask, struct), struct)
    filled = closed & ~mask  # new pixels only (originals kept separately)

    if not filled.any():
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    _, nn_idx = distance_transform_edt(~mask, return_indices=True)
    fy, fx = np.where(filled)

    # Nearest-neighbor fire_id for gap-filled pixels
    fid_map = np.full(shape, -1, dtype=np.int32)
    fid_map[y, x] = fids
    fill_fids = fid_map[nn_idx[0][fy, fx], nn_idx[1][fy, fx]]

    return fy.astype(np.int32), fx.astype(np.int32), fill_fids


# =============================================================================
# Job slicing — temporal chunks for cache locality
# =============================================================================

def _group_by_fire(samples: list[PickedSample]) -> dict[int, list[PickedSample]]:
    groups: dict[int, list[PickedSample]] = defaultdict(list)
    for s in samples:
        groups[s.fire_id].append(s)
    return dict(groups)


def _job_slice_by_date(samples: list[PickedSample],
                       ji: int, jn: int) -> list[PickedSample]:
    """Split samples by date. Each job gets a contiguous date range.

    This ensures each VNP03 granule is loaded by exactly one job in Phase 2,
    avoiding redundant I/O across jobs that share overlapping fire dates.
    Phase 1 (accum_t) recomputes per-fire history as needed — it's fast.
    """
    sorted_samples = sorted(samples, key=lambda s: s.day_T)
    n = len(sorted_samples)
    start = (n * ji) // jn
    end = (n * (ji + 1)) // jn
    return sorted_samples[start:end]


# =============================================================================
# PreprocessConfig
# =============================================================================

@dataclass
class PreprocessConfig(NextDayConfig):
    """NextDayConfig + preprocessing knobs."""
    # --- build phases ---
    samples_path: str = "data/next_day_v3/samples.json"
    ji: int = 0                      # job index  (0..jn-1)
    jn: int = 1                      # total jobs
    seed: int = 42
    show: int = 0                    # save diagnostic images for N random samples

    # --- patch ---
    source_version: str = "latest"
    target_version: str = "v2"
    min_fire_days: int = 5
    min_sample_year: int = 0
    filter_bad_fires: bool = True
    require_canopy: bool = False
    filter_inconsistent: bool = False
    filter_region: str = ""
    exclude_region: str = ""
    filter_fire_size: str = ""
    exclude_fire_size: str = ""

    # --- distill ---
    source: str = ""
    dest: str = ""
    n_fires: int = 100
    pool_size: int = 1000
    n_files: int = 2


# =============================================================================
# main — CLI dispatcher
# =============================================================================

def main():
    cli = Cli(PreprocessConfig, prog="next_day.preprocess",
              description="Build v3 dataset from manifest-based samples.")
    cli.command("build-accum", _build_accum_cmd)
    cli.command("build-lossmask", _build_lossmask_cmd)
    cli.command("build-fields", _build_fields_cmd)
    cli.command("compute-stats", _compute_stats_cmd)
    cli.command("show", _show_cmd)
    cli.command("patch", _patch_cmd)
    cli.command("distill", _distill_cmd)
    cli.run()


def _build_accum_cmd(cfg: PreprocessConfig):
    """Phase 1: accumulate fire state, write accum_t + masks."""
    build_accum(cfg)
    if cfg.show > 0:
        _show_accum(cfg)

def _build_lossmask_cmd(cfg: PreprocessConfig):
    """Phase 2: project VNP03+VNP14 fire/cloud masks."""
    build_lossmask(cfg)
    if cfg.show > 0:
        _show_lossmask(cfg)

def _build_fields_cmd(cfg: PreprocessConfig):
    """Phase 3: crop AE embeddings + weather rasters."""
    build_fields(cfg)
    if cfg.show > 0:
        _show_fields(cfg)

def _patch_cmd(cfg: PreprocessConfig):
    """Create a filtered dataset version."""
    patch(cfg)

def _distill_cmd(cfg: PreprocessConfig):
    """Distill a small test dataset."""
    distill(cfg)


# =============================================================================
# Phase 1: build_accum — fire state accumulation (VNP14 only)
# =============================================================================

def build_accum(cfg: PreprocessConfig):
    """Iterate raw VNP14 fire pixels day-by-day, accumulate fire state,
    emit accum_t + masks on picked days only.

    Uses raw Fires.Pixels (geodetic coordinates, 1 detection = 1 grid cell)
    instead of Fires.Projection (BFS-expanded kernel-bloated grid coords).
    This gives accurate fire shapes for cur_mask / next_mask ground truth.

    Writes: accum_t_{min,max,count}, accum_t (compat), cur_mask,
    next_mask, new_fires, plus sample JSON with metadata.

    Resume: if the shard JSON already exists and is non-empty, skip entirely
    (accumulation state is in-memory so partial resume is not feasible).
    """
    import h5py
    from firecomp.config import config as path_config
    from firecomp.dsrc.vnp14 import DEG_CELL_SIZE, Fires

    # ── Resume check ──
    output_dir = Path(cfg.dataset_dir)
    shard_json = output_dir / f"dataset_{cfg.ji}.json"
    if shard_json.exists():
        try:
            with open(shard_json) as f:
                existing = json.load(f)
            if existing:
                print(f"Phase 1 already complete: {shard_json} has "
                      f"{len(existing)} samples, skipping.")
                return
        except (json.JSONDecodeError, IOError):
            pass  # corrupt/empty — rebuild

    # ── Load and slice (by date, not fire_id) ──
    picked = load_picked_samples(Path(cfg.samples_path))
    my_picked = _job_slice_by_date(picked, cfg.ji, cfg.jn)
    if not my_picked:
        print(f"Job {cfg.ji}/{cfg.jn}: no samples in this slice.")
        return
    my_samples = _group_by_fire(my_picked)
    my_fire_ids = list(my_samples.keys())
    n_sample_days = len(my_picked)
    print(f"Job {cfg.ji}/{cfg.jn}: {len(my_fire_ids)} fires, "
          f"{n_sample_days} sample-days, "
          f"dates {my_picked[0].day_T}..{my_picked[-1].day_T}")

    # ── Fire metadata from VNP14 ──
    vnp14_h5 = h5py.File(path_config.vnp14_path, "r")
    meta = Fires.load(vnp14_h5, "meta")
    global_start = meta.start_date.item().date()  # Date
    fire_meta = _load_fire_metadata(vnp14_h5, my_fire_ids)

    # ── Padded bboxes ──
    crop_size = cfg.img_size - 2 * cfg.padding
    bbox_map: dict[int, tuple] = {}
    for fid, fm in fire_meta.items():
        bbox_map[fid] = _compute_padded_bbox(
            fm.min_x, fm.min_y, fm.max_x, fm.max_y,
            crop_size, cfg.padding, DEG_CELL_SIZE)

    # ── Load raw fire pixels (no BFS expansion) ──
    # Date filter: only load pixels within our fires' active date range.
    # This includes ALL fire events globally on those days — the per-fire
    # bbox filter in the accumulation loop handles spatial selection.
    t_min_fire = min((fm.start_date - global_start).days
                     for fm in fire_meta.values())
    t_min_sample = min((ps.day_T - global_start).days
                       for samples in my_samples.values() for ps in samples)
    t_min = max(t_min_fire, t_min_sample - 100)
    t_max = max((ps.day_T - global_start).days
                for samples in my_samples.values() for ps in samples)
    print(f"  Need pixels for day range t={t_min}..{t_max} "
          f"({t_max - t_min + 1} days)")

    pix_grp = vnp14_h5["pixels"]

    # Load dates first, filter by range, then load rest
    all_date = pix_grp["date"][:]
    n_total = len(all_date)
    global_start_dt64 = np.datetime64(global_start)
    all_t = ((all_date.astype("datetime64[ms]") - global_start_dt64)
             // np.timedelta64(1, "D")).astype(np.int32)
    del all_date

    pix_mask = (all_t >= t_min) & (all_t <= t_max)
    pix_t = all_t[pix_mask]; del all_t

    all_x = pix_grp["x"][:]
    pix_x_geod = all_x[pix_mask]; del all_x
    all_y = pix_grp["y"][:]
    pix_y_geod = all_y[pix_mask]; del all_y
    all_fid = pix_grp["fire_id"][:]
    pix_fid = all_fid[pix_mask]; del all_fid, pix_mask

    # Convert geodetic → grid coordinates
    pix_gx = np.round((pix_x_geod + 180.0) / DEG_CELL_SIZE).astype(np.int32)
    pix_gy = np.round((pix_y_geod + 90.0) / DEG_CELL_SIZE).astype(np.int32)
    del pix_x_geod, pix_y_geod

    print(f"  Loaded {len(pix_fid):,} fire pixels in date range "
          f"(of {n_total:,} total)")

    # Sort by t for fast day-wise slicing
    sort_idx = np.argsort(pix_t)
    pix_gx = pix_gx[sort_idx]
    pix_gy = pix_gy[sort_idx]
    pix_fid = pix_fid[sort_idx]
    pix_t_sorted = pix_t[sort_idx]
    del pix_t, sort_idx

    # Build offset index: t → (start, end) slice into sorted arrays
    unique_t, t_starts = np.unique(pix_t_sorted, return_index=True)
    t_ends = np.append(t_starts[1:], len(pix_t_sorted))
    pixels_by_t: dict[int, tuple[int, int]] = {
        int(t): (int(s), int(e))
        for t, s, e in zip(unique_t, t_starts, t_ends)
    }
    del unique_t, t_starts, t_ends, pix_t_sorted
    print(f"  Pixel days indexed: {len(pixels_by_t)} unique days")

    # ── Temporal index: day t → active fire_ids ──
    # Iterate from fire start through max(picked day_T) per fire.
    fires_by_t: dict[int, set[int]] = defaultdict(set)
    for fid, samples in my_samples.items():
        fm = fire_meta.get(fid)
        if fm is None:
            continue
        max_picked = max(s.day_T for s in samples)
        min_picked = min(s.day_T for s in samples)
        t_start = max((fm.start_date - global_start).days,
                       (min_picked - global_start).days - 100)
        t_end = (max_picked - global_start).days
        for t in range(t_start, t_end + 1):
            fires_by_t[t].add(fid)

    # ── Quick lookups for picked days ──
    picked_dates: dict[int, set[Date]] = defaultdict(set)
    picked_lookup: dict[tuple[int, Date], PickedSample] = {}
    for fid, samples in my_samples.items():
        for ps in samples:
            picked_dates[fid].add(ps.day_T)
            picked_lookup[(fid, ps.day_T)] = ps

    # ── Accumulators ──
    accum_min:   dict[int, np.ndarray] = {}
    accum_max:   dict[int, np.ndarray] = {}
    accum_count: dict[int, np.ndarray] = {}
    accum_comp:  dict[int, np.ndarray] = {}

    # ── Output ──
    writer = ShardWriter(output_dir / f"dataset_{cfg.ji}.h5")
    finished: list[Sample] = []
    finished_meta: list[dict] = []

    t_range = sorted(fires_by_t.keys())
    print(f"  Day range: t={t_range[0]}..{t_range[-1]} "
          f"({len(t_range)} days with active fires)")
    n_emit_calls = 0

    # ── Day loop (using raw pixel index, no projection_by_t) ──
    for t in t_range:
        # Init new fires
        for fid in fires_by_t[t]:
            if fid not in accum_min:
                bbox = bbox_map[fid]
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                accum_min[fid]   = np.full((h, w), -1, dtype=np.int32)
                accum_max[fid]   = np.full((h, w), -1, dtype=np.int32)
                accum_count[fid] = np.zeros((h, w), dtype=np.int32)
                accum_comp[fid]  = np.full((h, w), -1, dtype=np.int32)

        # Get raw fire pixels for this day (sorted slice, no H5 read)
        t_slice = pixels_by_t.get(t)

        # Update accumulators
        if t_slice is not None:
            sl_s, sl_e = t_slice
            day_gx = pix_gx[sl_s:sl_e]
            day_gy = pix_gy[sl_s:sl_e]
            day_fid = pix_fid[sl_s:sl_e]

            for fid in fires_by_t[t]:
                bbox = bbox_map[fid]

                # Accumulate ALL fire pixels within the padded bbox,
                # not just this fire_id. The model should see all fires
                # in the region, not just the event that defined the sample.
                bbox_sel = ((day_gx >= bbox[0]) & (day_gx < bbox[2]) &
                            (day_gy >= bbox[1]) & (day_gy < bbox[3]))
                x = day_gx[bbox_sel] - bbox[0]
                y = day_gy[bbox_sel] - bbox[1]

                if len(y) > 0:
                    accum_comp[fid][y, x] = day_fid[bbox_sel]
                    # min: first detection only
                    is_new = accum_min[fid][y, x] == -1
                    if is_new.any():
                        accum_min[fid][y[is_new], x[is_new]] = t
                    # max: always update
                    accum_max[fid][y, x] = t
                    # count: increment
                    accum_count[fid][y, x] += 1

                    # Gap-fill: morphological closing on today's detections.
                    # Fills 1-2 pixel gaps between nearby fire pixels without
                    # expanding the boundary.  No count increment — these
                    # pixels were never actually detected by the sensor.
                    fy, fx, ff = _gap_fill_detections(
                        y, x, day_fid[bbox_sel], accum_min[fid].shape)
                    if len(fy) > 0:
                        accum_comp[fid][fy, fx] = ff
                        is_new_f = accum_min[fid][fy, fx] == -1
                        if is_new_f.any():
                            accum_min[fid][fy[is_new_f], fx[is_new_f]] = t
                        accum_max[fid][fy, fx] = t

        # Emit tiles for picked days (accum includes today's pixels)
        date = global_start + timedelta(days=t)
        n_emit_calls += sum(1 for fid in fires_by_t[t]
                            if date in picked_dates.get(fid, set()))
        for fid in fires_by_t[t]:
            if date in picked_dates.get(fid, set()):
                _emit_tiles(
                    writer, finished, finished_meta, cfg,
                    fid, picked_lookup[(fid, date)],
                    accum_min[fid], accum_max[fid],
                    accum_count[fid], accum_comp[fid],
                    bbox_map[fid], fire_meta[fid], DEG_CELL_SIZE)

        # Cleanup fires no longer needed
        next_active = fires_by_t.get(t + 1, set())
        for fid in list(fires_by_t[t]):
            if fid not in next_active:
                for d in (accum_min, accum_max, accum_count, accum_comp):
                    d.pop(fid, None)

    writer.close()
    vnp14_h5.close()
    _write_build_json(finished, finished_meta,
                      output_dir / f"dataset_{cfg.ji}.json")
    print(f"\nPhase 1 done: {len(finished)} samples from "
          f"{n_emit_calls} emit calls ({n_sample_days} sample-days) "
          f"→ {output_dir / f'dataset_{cfg.ji}.h5'}")


def _emit_tiles(writer, finished, finished_meta, cfg,
                fid, ps, saved_min, saved_max, saved_count, saved_comp,
                bbox, fire_meta, deg_cell_size):
    """Tile a fire's day_T bbox into 256x256 patches, skip empty, write accum_t.

    The tiling grid is centered on fire pixels known at day_T (not the
    fire's full-lifetime extent), so the crop window doesn't leak future
    spread direction.  The tile origin offset (tile_ox, tile_oy) is stored
    in the metadata so Phase 2 can extract tiles at the same positions.

    Only writes accumulation fields (input features). Ground-truth masks
    (cur_mask, next_mask, new_fires) are produced by build_lossmask using
    geolocated VNP14 fire_mask projections for better accuracy.
    """
    crop_size = cfg.img_size - 2 * cfg.padding

    # Flip to image coords (y=0 at top)
    s_min  = np.flip(saved_min, axis=0)
    s_max  = np.flip(saved_max, axis=0)
    s_cnt  = np.flip(saved_count, axis=0)
    s_comp = np.flip(saved_comp, axis=0)

    # Compute tight extent of TARGET-fire pixels at day_T, then pad for
    # tiling.  Filtering by fire_id (via accum_comp) is critical: the
    # accumulators include all fires in the padded bbox for context, but
    # the tile-placement bbox must depend only on the target fire's
    # extent — otherwise an unrelated fire on the edge would shift the
    # crop window and leak its location into the sample's tile placement.
    full_h, full_w = s_min.shape
    has_fire = s_comp == fid
    if not has_fire.any():
        return  # no target-fire pixels — skip (shouldn't happen)

    # Small fires: if the global bbox already fits in a single tile,
    # skip the tight-bbox computation — just use the full array.
    if full_w <= cfg.img_size and full_h <= cfg.img_size:
        ox, oy = 0, 0
        w, h = full_w, full_h
    else:
        fy, fx = np.where(has_fire)
        fire_x0, fire_x1 = int(fx.min()), int(fx.max()) + 1
        fire_y0, fire_y1 = int(fy.min()), int(fy.max()) + 1

        # Pad for tiling alignment (same logic as _compute_padded_bbox)
        fw, fh = fire_x1 - fire_x0, fire_y1 - fire_y0
        pad_x = (crop_size - (fw % crop_size)) % crop_size
        pad_y = (crop_size - (fh % crop_size)) % crop_size
        ox = max(0, fire_x0 - (pad_x // 2 + cfg.padding))
        oy = max(0, fire_y0 - (pad_y // 2 + cfg.padding))
        ex = min(full_w, fire_x1 + (pad_x - pad_x // 2 + cfg.padding))
        ey = min(full_h, fire_y1 + (pad_y - pad_y // 2 + cfg.padding))
        w, h = ex - ox, ey - oy

    nx = (w - 2 * cfg.padding) // crop_size
    ny = (h - 2 * cfg.padding) // crop_size
    if nx == 0 or ny == 0:
        return  # shouldn't happen after the single-tile fast path

    lon0 = bbox[0] * deg_cell_size - 180
    lat1 = bbox[3] * deg_cell_size - 90
    day_of_fire = (ps.day_T - fire_meta.start_date).days

    for i in range(nx):
        for j in range(ny):
            x0, y0 = ox + i * crop_size, oy + j * crop_size
            x1, y1 = x0 + cfg.img_size, y0 + cfg.img_size

            if x1 > full_w or y1 > full_h:
                continue

            # Skip tiles with no fire from THIS component
            if (s_comp[y0:y1, x0:x1] == fid).sum() == 0:
                continue

            sample = Sample(
                fire_id=fid, xi=i, yi=j,
                lon=lon0 + (x0 + cfg.img_size // 2) * deg_cell_size,
                lat=lat1 - (y0 + cfg.img_size // 2) * deg_cell_size,
                dt=ps.day_T.isoformat(),
                img_size=cfg.img_size,
                idx=len(finished),
            )
            key = str(sample.idx)

            # accum_t variants  (1, H, W) int32
            writer.write(key, "accum_t_min",
                         s_min[y0:y1, x0:x1][np.newaxis])
            writer.write(key, "accum_t_max",
                         s_max[y0:y1, x0:x1][np.newaxis])
            writer.write(key, "accum_t_count",
                         s_cnt[y0:y1, x0:x1][np.newaxis].astype(np.float32))
            # backward compat alias
            writer.write(key, "accum_t",
                         s_min[y0:y1, x0:x1][np.newaxis])

            finished.append(sample)
            finished_meta.append({
                "region_id": ps.region_id,
                "fire_type": ps.fire_type,
                "num_fire": ps.num_fire,
                "day_of_fire": day_of_fire,
                "tile_ox": ox,  # tiling grid origin (image coords, within global bbox)
                "tile_oy": oy,
            })


# =============================================================================
# Phase 2: build_lossmask — VNP03+VNP14 fire/cloud projection
# =============================================================================

def build_lossmask(cfg: PreprocessConfig):
    """Project VNP14 fire_mask for day_T and day_T+1, aggregate across passes.

    For each sample, we need the satellite observation mask for BOTH days:
    - day_T:   if cloud/missing → model input is uncertain
    - day_T+1: if cloud/missing → target label is uncertain
    Loss should be masked where either day lacks a clear observation.

    VNP14 fire_mask source values (per pixel):
        0:   not-processed
        1:   bowtie (scan gap — invalid geolocation, excluded by lat/lon filter)
        2:   missing input data
        3:   water
        4:   cloud (can't see ground)
        5:   clear land
        6:   unclassified (ambiguous)
        7-9: fire (low / nominal / high conf)

    All source values are kept as-is through projection.  The no-data
    sentinel is 255 (not a valid VNP14 value), so only pixels with NO
    source observation at all are treated as no-data during interpolation.
    This prevents fire values from bleeding through cloud/unclassified
    gaps in the projection kernel.

    After projection, output values are:
        0-2: not-processed / bowtie / missing (no reliable observation)
        3:   water
        4:   cloud
        5:   clear land
        6:   unclassified
        7-9: fire
        255: no coverage (no source pixel landed here)

    Day/night merge:
        Daytime and nighttime passes are accumulated separately, then
        merged using _merge_vnp14_obs (priority-based: fire > clear >
        cloud > no-data).  This ensures fire detections always win, and
        clear surface observations beat cloud/unclassified.

    Writes per sample:
        vnp14_t          (1, H, W) uint8 — best observation for day_T
        vnp14_t1         (1, H, W) uint8 — best observation for day_T+1
        cur_mask         (1, H, W) int32 — fire at day_T  (vnp14_t >= 7)
        next_mask        (1, H, W) int32 — fire at day_T+1 (vnp14_t1 >= 7) — TARGET
        new_fires        (1, H, W) int32 — next_mask & no prior fire (accum_t_min == -1)
        ignition_t1      (1, H, W) uint8 — unpredictable new fire starts on day_T+1
        fire_id_mask_t1  (1, H, W) uint8 — pixels belonging to this fire_id on T+1

    ignition_t1 uses Fires.Pixels.ignition from BFS clustering — pixels whose
    first detection wasn't reachable via temporal fire spread.  These are
    genuinely new ignition points that the model cannot predict from day_T state
    alone.  Uses raw geodetic coords (1 detection = 1 grid cell, no kernel
    expansion), so ignition pixels are not bloated.

    fire_id_mask_t1 marks which day_T+1 fire pixels belong to the sample's
    fire_id (BFS component).  Other fires in the patch are visible in
    cur_mask (input context) but excluded from loss.  Same raw geodetic
    coord placement as ignition_t1.

    Downstream loss rule (in dataset.py):
        observed = (mask >= 3) & (mask <= 9)
        other_fire = (vnp14_t1 >= 7) & ~fire_id_mask_t1
        loss_valid = observed_t & observed_t1 & ~ignition_t1 & ~other_fire
    """
    import h5py
    import hdf5plugin  # noqa: F401
    from firecomp.config import config as path_config
    from firecomp.dsrc.vnp14 import DEG_CELL_SIZE
    from firecomp.dsrc.viirs import prefix_to_date_str, extract_vnp14_metadata
    from firecomp.dsrc.vnp03img import read_fire_mask
    from firecomp.dsrc.projection import (
        project_patch_uint8, read_vnp03,
    )
    from firecomp.helpers import ram_paths

    # ── Load Phase 1 output ──
    output_dir = Path(cfg.dataset_dir)
    shard_json = output_dir / f"dataset_{cfg.ji}.json"
    samples, meta_list = _load_build_json(shard_json)
    print(f"Job {cfg.ji}: {len(samples)} samples from Phase 1")

    if not samples:
        return

    # ── Resume: identify already-written samples ──
    shard_h5_path = output_dir / f"dataset_{cfg.ji}.h5"
    writer = ShardWriter(shard_h5_path)
    done_idx: set[int] = set()
    for s in samples:
        key = str(s.idx)
        # Check fire_id_mask_t1 — last field written by _commit_lossmask_sample.
        # vnp14_t/t1 can be empty placeholders; fire_id_mask_t1 means full commit.
        if writer.has_field(key, "fire_id_mask_t1"):
            done_idx.add(s.idx)

    all_done = len(done_idx) == len(samples)
    if all_done:
        print(f"  All {len(samples)} samples have vnp14_t + vnp14_t1")
    elif done_idx:
        print(f"  Resuming: {len(done_idx)}/{len(samples)} already done")

    # ── Fire metadata for bboxes ──
    vnp14_h5 = h5py.File(path_config.vnp14_path, "r")
    fire_ids = list({s.fire_id for s in samples})
    fire_meta = _load_fire_metadata(vnp14_h5, fire_ids)

    crop_size = cfg.img_size - 2 * cfg.padding
    bbox_map: dict[int, tuple] = {}
    for fid, fm in fire_meta.items():
        bbox_map[fid] = _compute_padded_bbox(
            fm.min_x, fm.min_y, fm.max_x, fm.max_y,
            crop_size, cfg.padding, DEG_CELL_SIZE)

    # Precompute geodetic bboxes for early granule rejection
    geod_bbox_map: dict[int, tuple[float, float, float, float]] = {}
    for fid, pix_bbox in bbox_map.items():
        gb = _pix_to_geod(pix_bbox, DEG_CELL_SIZE)
        geod_bbox_map[fid] = (gb[0], gb[1], gb[2], gb[3])

    # ── Load fire pixels from Fires.Pixels ──
    # Raw geodetic coords → grid coords (1 detection = 1 cell, no expansion).
    # Build two lookups keyed by (fire_id, date_str) → list of (gx, gy):
    #   ignition_by_fid_date  — ignition=True pixels only (for ignition_t1)
    #   allpix_by_fid_date    — ALL pixels (for fire_id_mask_t1)
    fire_id_set = set(fire_ids)
    pix_grp = vnp14_h5["pixels"]

    all_fid = pix_grp["fire_id"][:]
    pix_mask = np.isin(all_fid, list(fire_id_set))
    pix_fid = all_fid[pix_mask]; del all_fid

    all_ign = pix_grp["ignition"][:]
    pix_ign = all_ign[pix_mask].astype(bool); del all_ign

    all_x = pix_grp["x"][:]
    pix_x = all_x[pix_mask]; del all_x
    all_y = pix_grp["y"][:]
    pix_y = all_y[pix_mask]; del all_y
    all_date = pix_grp["date"][:]
    pix_date = all_date[pix_mask]; del all_date
    del pix_mask

    # Convert geodetic → grid coordinates (all pixels)
    pix_gx = np.round((pix_x.astype(np.float64) + 180.0) / DEG_CELL_SIZE).astype(np.int32)
    pix_gy = np.round((pix_y.astype(np.float64) + 90.0) / DEG_CELL_SIZE).astype(np.int32)
    del pix_x, pix_y

    # Convert dates to ISO strings
    pix_date_str = np.array([
        str(d)[:10] for d in pix_date.astype("datetime64[ms]")
    ])
    del pix_date

    # Build both indexes in one pass
    ignition_by_fid_date: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    allpix_by_fid_date: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(pix_fid)):
        fid_i = int(pix_fid[i])
        date_i = pix_date_str[i]
        gx_i = int(pix_gx[i])
        gy_i = int(pix_gy[i])
        allpix_by_fid_date[(fid_i, date_i)].append((gx_i, gy_i))
        if pix_ign[i]:
            ignition_by_fid_date[(fid_i, date_i)].append((gx_i, gy_i))
    del pix_fid, pix_ign, pix_gx, pix_gy, pix_date_str

    print(f"  Pixel index: {len(allpix_by_fid_date)} (fire, date) entries, "
          f"{len(ignition_by_fid_date)} with ignitions")
    vnp14_h5.close()

    if not all_done:
        # ── Collect needed dates from TODO samples only ──
        # Samples already written don't need granule reads.
        todo_set = {s.idx for s in samples} - done_idx

        fids_by_date: dict[str, set[int]] = defaultdict(set)
        for s in samples:
            if s.idx not in todo_set:
                continue
            day_t = datetime.fromisoformat(s.dt).date()
            day_t1 = day_t + timedelta(days=1)
            fids_by_date[day_t.isoformat()].add(s.fire_id)
            fids_by_date[day_t1.isoformat()].add(s.fire_id)

        needed_dates = set(fids_by_date.keys())
        print(f"  Need observations for {len(needed_dates)} unique dates "
              f"(day_T + day_T+1)")

        # ── Index TODO samples by day_T+1 for streaming commit ──
        # Once all granules for date D are processed, samples with
        # day_T+1 == D can be written immediately.
        samples_by_t1: dict[str, list[int]] = defaultdict(list)
        idx_to_pos: dict[int, int] = {s.idx: i for i, s in enumerate(samples)}
        idx_to_meta: dict[int, dict] = {s.idx: meta_list[i]
                                         for i, s in enumerate(samples)}
        for s in samples:
            if s.idx not in todo_set:
                continue
            day_t1 = (datetime.fromisoformat(s.dt).date()
                      + timedelta(days=1)).isoformat()
            samples_by_t1[day_t1].append(s.idx)

        # ── Build VNP03 prefix → path lookup ──
        print("  Scanning VNP03IMG directory...")
        vnp03_by_prefix = _build_prefix_lookup(
            path_config.vnp03img_dir, "VNP03IMG")
        print(f"  {len(vnp03_by_prefix)} VNP03IMG files on disk")

        # ── Filter VNP03 prefixes to only our needed dates ──
        prefixes_by_date: dict[str, list[str]] = defaultdict(list)
        for prefix in vnp03_by_prefix:
            try:
                date_str = prefix_to_date_str(prefix)
            except Exception:
                continue
            if date_str in needed_dates:
                prefixes_by_date[date_str].append(prefix)

        n_granules = sum(len(v) for v in prefixes_by_date.values())
        print(f"  {n_granules} VNP03 granules on needed dates")

        # ── Accumulation buffers ──
        fire_day_buf: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        fire_night_buf: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        fire_buf: dict[int, dict[str, np.ndarray]] = defaultdict(dict)

        # Fire center longitudes for day/night determination
        fire_lon: dict[int, float] = {}
        for fid, fm in fire_meta.items():
            fire_lon[fid] = (fm.min_x + fm.max_x) / 2.0

        n_read = 0
        n_no_vnp14 = 0
        n_meta_skip = 0
        n_projected = 0
        n_committed = 0
        n_day_granules = 0
        n_night_granules = 0
        n_samples_total = len(samples)

        empty = np.zeros((1, cfg.img_size, cfg.img_size), dtype=np.uint8)

        sorted_dates = sorted(prefixes_by_date)

        for di, date_str in enumerate(sorted_dates):
            fids_today = fids_by_date[date_str]

            # ── Read and project all granules for this date ──
            for prefix in prefixes_by_date[date_str]:
                vnp03_path = vnp03_by_prefix[prefix]
                vnp14_path = _find_file_by_prefix(
                    path_config.vnp14_dir, "VNP14IMG", prefix)
                if vnp14_path is None:
                    n_no_vnp14 += 1
                    continue

                # Early reject via VNP14 metadata bbox (no data loading).
                # Axis-aligned bbox from file attributes — generous but
                # eliminates granules on other continents instantly.
                # 1° padding: metadata bbox may underestimate curved
                # high-latitude swaths (great-circle bulge).
                meta_bbox = extract_vnp14_metadata(vnp14_path)
                if meta_bbox is not None:
                    m_lonmin, m_latmin, m_lonmax, m_latmax = meta_bbox
                    m_lonmin -= 1.0; m_lonmax += 1.0
                    m_latmin = max(-90.0, m_latmin - 1.0)
                    m_latmax = min(90.0, m_latmax + 1.0)
                    any_overlap = False
                    for fid in fids_today:
                        gb = geod_bbox_map.get(fid)
                        if gb is None:
                            continue
                        if not (gb[2] < m_lonmin or gb[0] > m_lonmax or
                                gb[3] < m_latmin or gb[1] > m_latmax):
                            any_overlap = True
                            break
                    if not any_overlap:
                        n_meta_skip += 1
                        continue

                n_read += 1
                print(f"  [{n_read + n_meta_skip}/{n_granules}] "
                      f"{date_str} {prefix} "
                      f"({len(fids_today)} fires)", flush=True)

                try:
                    with ram_paths(vnp03_path, vnp14_path) as (rv03, rv14):
                        lat, lon = read_vnp03(rv03)
                        fire_mask = read_fire_mask(rv14)
                except Exception as e:
                    print(f"    ERROR reading: {e}")
                    continue

                if fire_mask.shape[0] != lat.shape[0]:
                    continue

                # Keep all VNP14 values as-is.  The projection uses
                # no_data_val=255, so only truly unobserved output pixels
                # get the no-data sentinel.  Cloud/unclassified values
                # participate in nearest-neighbor and prevent fire from
                # bleeding through gaps.

                # Pre-compute granule coverage for fast fire rejection
                gran_lat_min = float(lat.min())
                gran_lat_max = float(lat.max())
                gran_lon_min = float(lon.min())
                gran_lon_max = float(lon.max())

                for fid in fids_today:
                    if fid not in geod_bbox_map:
                        continue

                    geod_bbox = geod_bbox_map[fid]

                    # Quick reject: fire bbox doesn't overlap granule
                    if (geod_bbox[2] < gran_lon_min or
                        geod_bbox[0] > gran_lon_max or
                        geod_bbox[3] < gran_lat_min or
                        geod_bbox[1] > gran_lat_max):
                        continue

                    is_day = _is_daytime_pass(
                        prefix, fire_lon.get(fid, 0.0))
                    buf = fire_day_buf if is_day else fire_night_buf

                    patch = project_patch_uint8(
                        fire_mask, lat, lon, geod_bbox,
                        no_data_val=255)
                    if patch is None:
                        continue
                    if patch.ndim == 3:
                        patch = patch[0]
                    # 255 served its purpose during projection (C++ skips
                    # those source pixels, unvisited output → 255).  Now
                    # convert back to 0 (not-processed) so that priority
                    # merges and >= 7 fire checks work correctly.
                    patch[patch == 255] = 0
                    # project_patch_uint8 already outputs image coords
                    # (north at row 0) — do NOT flip.

                    # Accumulate: priority merge across passes
                    if date_str in buf[fid]:
                        buf[fid][date_str] = _merge_vnp14_obs(
                            buf[fid][date_str], patch)
                    else:
                        buf[fid][date_str] = patch.copy()

                    if is_day:
                        n_day_granules += 1
                    else:
                        n_night_granules += 1

            # ── Merge day/night for this date ──
            for fid in (set(fire_day_buf.keys()) |
                        set(fire_night_buf.keys())):
                day = fire_day_buf.get(fid, {}).pop(date_str, None)
                night = fire_night_buf.get(fid, {}).pop(date_str, None)
                if day is not None and night is not None:
                    fire_buf[fid][date_str] = _merge_vnp14_obs(day, night)
                elif day is not None:
                    fire_buf[fid][date_str] = day
                elif night is not None:
                    fire_buf[fid][date_str] = night

            # ── Commit samples whose day_T+1 == date_str ──
            # Both day_T and day_T+1 observations are now complete.
            to_commit = samples_by_t1.get(date_str, [])
            for idx in to_commit:
                s = samples[idx_to_pos[idx]]
                meta = idx_to_meta.get(idx, {})
                n_projected += _commit_lossmask_sample(
                    writer, s, fire_buf, bbox_map, cfg,
                    ignition_by_fid_date, allpix_by_fid_date,
                    empty, crop_size, meta)
            if to_commit:
                n_committed += len(to_commit)
                print(f"    committed {len(to_commit)} samples "
                      f"({n_committed}/{n_samples_total}, "
                      f"day_T+1={date_str})", flush=True)

            # ── Free buffers ≥2 days old (no longer needed) ──
            if di >= 1:
                old_date = sorted_dates[di - 1]
                for fid in list(fire_buf.keys()):
                    fire_buf[fid].pop(old_date, None)

        # ── Commit any remaining samples ──
        # (day_T+1 wasn't in prefixes_by_date, e.g. no granules that day)
        for idx in todo_set - done_idx:
            if not (writer.has_field(str(idx), "vnp14_t") and
                    writer.has_field(str(idx), "vnp14_t1")):
                s = samples[idx_to_pos[idx]]
                meta = idx_to_meta.get(idx, {})
                _commit_lossmask_sample(
                    writer, s, fire_buf, bbox_map, cfg,
                    ignition_by_fid_date, allpix_by_fid_date,
                    empty, crop_size, meta)

        print(f"  Granules read: {n_read}  (no VNP14: {n_no_vnp14},  "
              f"skipped by metadata bbox: {n_meta_skip})")
        print(f"  Day projections: {n_day_granules}  "
              f"Night projections: {n_night_granules}")

    # ── Filter out samples with too few valid loss pixels ──
    # Always runs, even on resume (in case previous run was interrupted
    # before filtering).  Samples below 10% loss_valid are useless for
    # training.  Removed from JSON; H5 groups left as harmless orphans.
    MIN_LOSS_VALID_FRAC = 0.10
    n_total = cfg.img_size * cfg.img_size
    h5 = writer._h5
    kept_samples = []
    kept_meta = []
    n_dropped = 0
    for s, m in zip(samples, meta_list):
        key = str(s.idx)
        if key not in h5:
            kept_samples.append(s)
            kept_meta.append(m)
            continue
        grp = h5[key]
        has_t = "vnp14_t" in grp
        has_t1 = "vnp14_t1" in grp
        if has_t and has_t1:
            mask_t = grp["vnp14_t"][0]
            mask_t1 = grp["vnp14_t1"][0]
            loss_valid = _is_clear_obs(mask_t) & _is_clear_obs(mask_t1)
            if "ignition_t1" in grp:
                loss_valid = loss_valid & (grp["ignition_t1"][0] == 0)
            frac = loss_valid.sum() / n_total
            if frac < MIN_LOSS_VALID_FRAC:
                n_dropped += 1
                continue
        kept_samples.append(s)
        kept_meta.append(m)

    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{len(samples)} samples with "
              f"loss_valid < {MIN_LOSS_VALID_FRAC:.0%}")
    # Always rewrite JSON (ensures filter is applied even on resume)
    _write_build_json(kept_samples, kept_meta, shard_json)

    writer.close()
    print(f"\nPhase 2 done: {len(kept_samples)} samples kept "
          f"({n_dropped} dropped for low coverage)")


def _commit_lossmask_sample(
    writer, s, fire_buf, bbox_map, cfg,
    ignition_by_fid_date, allpix_by_fid_date, empty, crop_size,
    meta=None,
) -> int:
    """Write vnp14_t/t1, cur/next_mask, new_fires, ignition_t1,
    fire_id_mask_t1 for one sample.

    Returns 1 if observation tiles were written, 0 otherwise (for counting).
    """
    if meta is None:
        meta = {}
    fid = s.fire_id
    key = str(s.idx)
    day_t = datetime.fromisoformat(s.dt).date().isoformat()
    day_t1 = (datetime.fromisoformat(s.dt).date()
              + timedelta(days=1)).isoformat()

    # Tile offset: tile_ox/oy from Phase 1 day_T bbox (0 for legacy data)
    tile_ox = meta.get("tile_ox", 0)
    tile_oy = meta.get("tile_oy", 0)
    x0 = tile_ox + s.xi * crop_size
    y0 = tile_oy + s.yi * crop_size
    x1 = x0 + cfg.img_size
    y1 = y0 + cfg.img_size

    # Write vnp14_t and vnp14_t1 observation masks
    tile_t = None
    tile_t1 = None
    n_obs = 0
    for field, date_key, is_t in [("vnp14_t", day_t, True),
                                   ("vnp14_t1", day_t1, False)]:
        buf = fire_buf.get(fid, {}).get(date_key)
        if buf is not None and y1 <= buf.shape[0] and x1 <= buf.shape[1]:
            tile = buf[y0:y1, x0:x1]
            writer.write(key, field,
                         tile[np.newaxis].astype(np.uint8))
            n_obs += 1
            if is_t:
                tile_t = tile
            else:
                tile_t1 = tile
        else:
            writer.write(key, field, empty)

    # Derive cur_mask, next_mask from VNP14 fire pixels (values >= 7)
    if tile_t is not None:
        cur_mask = (tile_t >= 7).astype(np.int32)
    else:
        cur_mask = np.zeros((cfg.img_size, cfg.img_size), dtype=np.int32)
    writer.write(key, "cur_mask", cur_mask[np.newaxis])

    if tile_t1 is not None:
        next_mask = (tile_t1 >= 7).astype(np.int32)
    else:
        next_mask = np.zeros((cfg.img_size, cfg.img_size), dtype=np.int32)
    writer.write(key, "next_mask", next_mask[np.newaxis])

    # new_fires: fire on T+1 that wasn't burning before (accum_t_min == -1)
    new_fires = next_mask.copy()
    h5 = writer._h5
    if key in h5 and "accum_t_min" in h5[key]:
        accum_min_tile = h5[key]["accum_t_min"][0]  # (H, W)
        new_fires[accum_min_tile != -1] = 0
    writer.write(key, "new_fires", new_fires[np.newaxis])

    # ignition_t1: unpredictable new fire starts on day_T+1.
    bbox = bbox_map.get(fid)
    ign_pixels = ignition_by_fid_date.get((fid, day_t1), [])
    ign_mask = np.zeros((cfg.img_size, cfg.img_size), dtype=np.uint8)
    if ign_pixels and bbox is not None:
        for gx, gy in ign_pixels:
            lx = gx - bbox[0]
            ly = gy - bbox[1]
            ly_img = (bbox[3] - bbox[1]) - 1 - ly
            tx = lx - x0
            ty = ly_img - y0
            if 0 <= tx < cfg.img_size and 0 <= ty < cfg.img_size:
                ign_mask[ty, tx] = 1
    writer.write(key, "ignition_t1", ign_mask[np.newaxis])

    # fire_id_mask_t1: boolean mask of pixels belonging to THIS fire_id on
    # day_T+1.  Used to exclude other fires from loss — the model is only
    # supervised on the target fire's spread + non-fire pixels.
    #
    # Projection expands each raw VNP14 detection into a small blob (kernel
    # ~3–5 px), so raw-pixel-position marking alone misses most of the blob.
    # Instead, compute connected components of the projected fire mask;
    # any component containing a raw detection of the target fire is
    # entirely the target.  Components without target raw detections stay
    # at 0 and are treated as "other fire" by the loss rule.
    fid_mask = np.zeros((cfg.img_size, cfg.img_size), dtype=np.uint8)
    fid_pixels = allpix_by_fid_date.get((fid, day_t1), [])
    if fid_pixels and bbox is not None and tile_t1 is not None:
        from scipy.ndimage import label as _cc_label
        fire_t1_mask = (tile_t1 >= 7) & (tile_t1 <= 9)
        cc, n_cc = _cc_label(fire_t1_mask, structure=np.ones((3, 3)))
        target_components: set[int] = set()
        for gx, gy in fid_pixels:
            lx = gx - bbox[0]
            ly = gy - bbox[1]
            ly_img = (bbox[3] - bbox[1]) - 1 - ly
            tx = lx - x0
            ty = ly_img - y0
            if 0 <= tx < cfg.img_size and 0 <= ty < cfg.img_size:
                c = int(cc[ty, tx])
                if c > 0:
                    target_components.add(c)
        if target_components:
            comp_mask = np.isin(cc, list(target_components))
            fid_mask[comp_mask] = 1
    writer.write(key, "fire_id_mask_t1", fid_mask[np.newaxis])

    return n_obs


def _build_prefix_lookup(directory: str, product: str) -> dict[str, str]:
    """List files in directory, build prefix → full_path dict."""
    from firecomp.dsrc.viirs import viirs_granule_prefix
    lookup: dict[str, str] = {}
    if not os.path.isdir(directory):
        return lookup
    for fname in os.listdir(directory):
        if not fname.startswith(product):
            continue
        prefix = viirs_granule_prefix(fname)
        if prefix:
            lookup[prefix] = os.path.join(directory, fname)
    return lookup


def _find_file_by_prefix(directory: str, product: str, prefix: str):
    """Find a file matching product.prefix.* using glob (no full dir scan)."""
    import glob as g
    pattern = os.path.join(directory, f"{product}.{prefix}.*")
    matches = g.glob(pattern)
    return matches[0] if matches else None


def _is_daytime_pass(prefix: str, fire_lon: float) -> bool:
    """Determine if a VIIRS granule is a daytime pass at a given longitude.

    Uses the UTC overpass time from the granule prefix plus the fire's
    longitude to compute approximate local solar time.  Daytime = roughly
    06:00–18:00 local solar time.

    Nighttime thermal-only passes can detect fire but *cannot* reliably
    distinguish cloud from clear land, so they should not contribute to
    loss mask aggregation (where cloud uncertainty is the signal).

    Args:
        prefix: Granule prefix "A{YYYYDDD}.{HHMM}.{VVV}"
        fire_lon: Fire center longitude (degrees, -180..180)

    Returns:
        True if the pass is daytime at the fire location.
    """
    hhmm = prefix[9:13]  # e.g. "0900"
    utc_hour = int(hhmm[:2]) + int(hhmm[2:]) / 60.0
    # Approximate local solar time (no DST, no equation of time — good enough)
    local_hour = (utc_hour + fire_lon / 15.0) % 24.0
    return 6.0 <= local_hour < 18.0



# =============================================================================
# Phase 3: build_fields — AE embeddings + weather rasters
# =============================================================================

def build_fields(cfg: PreprocessConfig):
    """Crop AE embeddings, ERA5 weather (day_T), GFS forecast (day_T+1)
    from existing rasters.  Pure windowed reads, fast.

    Resume-friendly: skips per-sample fields that already exist in the H5.
    """
    from firecomp.dsrc.all_dsrc import dsrc

    # ── Load Phase 1 output ──
    output_dir = Path(cfg.dataset_dir)
    shard_json = output_dir / f"dataset_{cfg.ji}.json"
    samples, meta_list = _load_build_json(shard_json)
    print(f"Job {cfg.ji}: {len(samples)} samples")

    if not samples:
        return

    # ── Fire metadata for bboxes ──
    import h5py
    from firecomp.config import config as path_config
    from firecomp.dsrc.vnp14 import DEG_CELL_SIZE

    vnp14_h5 = h5py.File(path_config.vnp14_path, "r")
    fire_ids = list({s.fire_id for s in samples})
    fire_meta = _load_fire_metadata(vnp14_h5, fire_ids)
    vnp14_h5.close()

    crop_size = cfg.img_size - 2 * cfg.padding
    bbox_map: dict[int, tuple] = {}
    for fid, fm in fire_meta.items():
        bbox_map[fid] = _compute_padded_bbox(
            fm.min_x, fm.min_y, fm.max_x, fm.max_y,
            crop_size, cfg.padding, DEG_CELL_SIZE)

    # ── Group samples + build meta lookup ──
    fire_samples: dict[int, list[Sample]] = defaultdict(list)
    idx_to_meta: dict[int, dict] = {}
    for i, s in enumerate(samples):
        fire_samples[s.fire_id].append(s)
        idx_to_meta[s.idx] = meta_list[i]

    writer = ShardWriter(output_dir / f"dataset_{cfg.ji}.h5")

    # ── Resume check: count already-done samples ──
    # Check the last expected field (gfs or weather) as a proxy for completion.
    _resume_field = "gfs" if cfg.include_gfs else "weather"
    n_already = sum(1 for s in samples
                    if writer.has_field(str(s.idx), _resume_field))
    if n_already == len(samples):
        writer.close()
        print(f"Phase 3 already complete: all {n_already} samples have "
              f"'{_resume_field}', skipping.")
        return
    if n_already > 0:
        print(f"  Resuming: {n_already}/{len(samples)} already done")

    stats_accum = _StatsAccumulator(cfg)
    n_weather, n_gfs, n_skipped = 0, 0, 0

    # Weather/GFS native resolution is far coarser than VIIRS 375m.
    # Store at native_size (32) per tile instead of full 256 — ~64x smaller.
    # dataset.py upsamples 32→256 at load time via load_field().
    weather_native = ALL_FIELDS["weather"].native_size   # 32
    weather_scale = weather_native / cfg.img_size         # 32/256 = 0.125

    for fid, f_samples in fire_samples.items():
        if fid not in bbox_map:
            continue
        bbox = bbox_map[fid]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        geod_bbox = _pix_to_geod(bbox, DEG_CELL_SIZE)

        # Reduced-resolution bbox dims for weather/GFS
        w_lo = max(1, round(w * weather_scale))
        h_lo = max(1, round(h * weather_scale))

        # AE embeddings: per-tile, written once per unique tile.
        # Year-matched: fire in year N uses AE from year N-1, clamped 2017-2024.
        first_sample = f_samples[0]
        t_date = datetime.fromisoformat(first_sample.dt)
        ae_year = max(2017, min(2024, t_date.year - 1))

        ae_pca_dsrc = _get_ae_cached(ae_year, pca_bands=5)
        ae_pca_5 = ae_pca_dsrc.get_sample(
            geod_bbox, t_date, x_size=w, y_size=h)
        ae_12345_dsrc = _get_ae_cached(ae_year, band_list=[1, 2, 3, 4, 5])
        ae_12345 = ae_12345_dsrc.get_sample(
            geod_bbox, t_date, x_size=w, y_size=h)

        for s in f_samples:
            meta = idx_to_meta.get(s.idx, {})
            tile_ox = meta.get("tile_ox", 0)
            tile_oy = meta.get("tile_oy", 0)
            x0 = tile_ox + s.xi * crop_size
            y0 = tile_oy + s.yi * crop_size
            x1 = x0 + cfg.img_size
            y1 = y0 + cfg.img_size
            key = str(s.idx)

            # Resume: skip if this sample already has its fields
            if writer.has_field(key, _resume_field):
                n_skipped += 1
                continue

            # AE embeddings — per-sample (day_T bbox means each sample
            # may have different tile offsets, so tile sharing is invalid).
            if ae_pca_5 is not None:
                patch = ae_pca_5[:, y0:y1, x0:x1]
                writer.write(key, "ae_pca_5", patch)
                stats_accum.update("ae_pca_5", patch)
            if ae_12345 is not None:
                patch = ae_12345[:, y0:y1, x0:x1]
                writer.write(key, "ae_12345", patch)
                stats_accum.update("ae_12345", patch)

            # Per-sample: ERA5 weather at day_T
            # Stored at reduced resolution (32x32 per tile) — upsampled at load.
            sample_date = datetime.fromisoformat(s.dt)
            lx0 = round(x0 * weather_scale)
            ly0 = round(y0 * weather_scale)
            lx1 = lx0 + weather_native
            ly1 = ly0 + weather_native
            if cfg.include_weather:
                weather = dsrc.era5_fire.get_sample(
                    geod_bbox, sample_date, x_size=w_lo, y_size=h_lo)
                if weather is not None:
                    patch = weather[:, ly0:ly1, lx0:lx1].astype(np.float32)
                    writer.write(key, "weather", patch)
                    stats_accum.update("weather", patch)
                    n_weather += 1

            # Per-sample: GFS forecast for day_T+1 weather.
            # GFS day01 from init_date forecasts init_date+1, so
            # init at day_T → day01 = day_T+1 weather.
            if cfg.include_gfs:
                gfs_date = sample_date
                gfs = dsrc.gfs_1day.get_sample(
                    geod_bbox, gfs_date, x_size=w_lo, y_size=h_lo)
                if gfs is not None:
                    patch = gfs[:, ly0:ly1, lx0:lx1].astype(np.float32)
                    writer.write(key, "gfs", patch)
                    stats_accum.update("gfs", patch)
                    n_gfs += 1

    # Stats are computed separately via `compute-stats` command
    # after all shards are built. This avoids shard-0 bias and
    # resume inconsistencies.

    writer.close()
    n_samples = len(samples) - n_skipped
    print(f"\nPhase 3 done: {n_samples} samples, "
          f"{n_weather} weather, {n_gfs} gfs")


# =============================================================================
# compute-stats — compute normalization stats from random 10% of samples
# =============================================================================

def _compute_stats_cmd(cfg: PreprocessConfig):
    """Compute normalization stats (mean/std) from a random 10% of samples."""
    compute_stats(cfg)


def compute_stats(cfg: PreprocessConfig, frac: float = 0.10):
    """Compute per-channel mean/std for normalizable fields.

    Reads from the built dataset (all shards), picks `frac` of samples
    uniformly at random, and computes Welford running stats on the
    dtype-transformed values (matching training time).

    Result: <dataset_dir>/stats.json
    """
    import hdf5plugin  # noqa: F401
    import h5py
    import random as rng

    from firecomp.next_day.dataset import ALL_FIELDS

    output_dir = Path(cfg.dataset_dir)
    h5_paths = sorted(output_dir.glob("dataset_*.h5"))
    json_paths = sorted(output_dir.glob("dataset_*.json"))
    if not h5_paths:
        print(f"No dataset_*.h5 files in {output_dir}. Run build phases first.")
        return
    if not json_paths:
        print(f"No dataset_*.json files in {output_dir}. Run build phases first.")
        return

    # Open all shards
    h5s = [h5py.File(p, "r") for p in h5_paths]

    # Collect (shard_idx, sample_idx) pairs from all JSON files
    all_keys: list[tuple[int, str]] = []
    tile_keys: dict[str, tuple[int, str]] = {}   # tile_key -> (shard_idx, fire_id_xi_yi)
    for file_idx, jp in enumerate(json_paths):
        with open(jp) as f:
            records = json.load(f)
        for rec in records:
            idx = str(rec["idx"])
            all_keys.append((file_idx, idx))
            # Build tile_key for per-tile field lookup
            tk = f"{rec.get('fire_id', '')}_{rec.get('xi', '')}_{rec.get('yi', '')}"
            if tk not in tile_keys:
                tile_keys[tk] = (file_idx, tk)

    if not all_keys:
        print("No samples found in JSON files.")
        for h5 in h5s:
            h5.close()
        return

    # Pick a random subset
    n_pick = max(1, int(len(all_keys) * frac))
    rng.seed(42)
    subset = rng.sample(all_keys, n_pick)
    print(f"Computing stats from {n_pick}/{len(all_keys)} samples "
          f"({frac:.0%}) across {len(h5_paths)} shards")

    # Fields that get normalized at training time
    stat_fields = {name: f for name, f in ALL_FIELDS.items() if f.normalize}
    accum = _StatsAccumulator(cfg)

    for i, (shard_idx, sample_key) in enumerate(subset):
        h5 = h5s[shard_idx]
        if sample_key not in h5:
            continue
        grp = h5[sample_key]

        for name, field in stat_fields.items():
            # Per-sample fields
            if field.storage == "per_sample" and name in grp:
                data = grp[name][:]
                if field.dtype_transform is not None:
                    data = field.dtype_transform(data)
                else:
                    data = data.astype(np.float64)
                accum.update(name, data)
            # Per-tile fields (stored in top-level group keyed by field name)
            elif field.storage == "per_tile" and name in h5:
                # Pick any tile key present in this shard
                tile_grp = h5[name]
                for tk in tile_grp:
                    data = tile_grp[tk][:]
                    if field.dtype_transform is not None:
                        data = field.dtype_transform(data)
                    else:
                        data = data.astype(np.float64)
                    accum.update(name, data)
                    break  # one tile per sample is enough for stats

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n_pick}]")

    for h5 in h5s:
        h5.close()

    stats_path = output_dir / "stats.json"
    accum.dump(stats_path)
    print(f"Wrote {stats_path} ({len(accum._totals)} fields)")


# =============================================================================
# patch — create a new dataset version by filtering an existing one
# =============================================================================

def patch(cfg: PreprocessConfig):
    """
    Create a new dataset version from an existing one by applying filters.

    Reads `source_version` (default: "latest" = base dir), applies the
    configured filters, writes a new `target_version` (default: "v2") with:
      - dataset_k.json  — filtered subset per shard (idx preserved)
      - dataset_k.h5    — symlink to source H5 (no duplication)
    """
    dataset_dir = Path(cfg.dataset_dir)

    if cfg.source_version == "latest":
        source_path = dataset_dir
    else:
        source_path = dataset_dir / cfg.source_version

    samples = load_all_samples(source_path)
    src_h5_files = sorted(source_path.glob("dataset_*.h5"))
    n_before = len(samples)
    print(f"Source: {source_path} ({n_before} samples, "
          f"{len(src_h5_files)} shard{'s' if len(src_h5_files) != 1 else ''})")

    if cfg.filter_bad_fires:
        samples = _filter_bad_fires(samples)
        print(f"  filter_bad_fires: {n_before} → {len(samples)}")

    if cfg.min_fire_days > 0:
        n = len(samples)
        samples = _filter_min_days(samples, cfg.min_fire_days)
        print(f"  min_fire_days={cfg.min_fire_days}: {n} → {len(samples)}")

    if cfg.min_sample_year > 0:
        n = len(samples)
        samples = [s for s in samples
                   if int(str(s.dt)[:4]) >= cfg.min_sample_year]
        print(f"  min_sample_year={cfg.min_sample_year}: {n} → {len(samples)}")

    if cfg.require_canopy:
        n = len(samples)
        import hdf5plugin  # noqa: F401
        import h5py
        from firecomp.core.dataset_utils import SampleStore
        store = SampleStore(source_path)
        samples = [s for s in samples
                   if "canopy_height" in
                   store._h5s[getattr(s, '_h5_file', 0)][str(s.idx)]]
        store.close()
        print(f"  require_canopy: {n} → {len(samples)}")

    if cfg.filter_inconsistent:
        n = len(samples)
        samples = _filter_inconsistent_fields(samples, source_path)
        print(f"  filter_inconsistent: {n} → {len(samples)}")

    if cfg.filter_region or cfg.exclude_region or \
       cfg.filter_fire_size or cfg.exclude_fire_size:
        n = len(samples)
        samples = _filter_region_and_size(
            samples, cfg.filter_region, cfg.exclude_region,
            cfg.filter_fire_size, cfg.exclude_fire_size)
        print(f"  region/size filter: {n} → {len(samples)}")

    if not samples:
        print("ERROR: No samples remain after filtering!")
        return

    target_path = dataset_dir / cfg.target_version
    target_path.mkdir(parents=True, exist_ok=True)
    _write_patch_output(source_path, target_path, src_h5_files,
                        samples, cfg.source_version)

    patch_meta = {
        "source_version": cfg.source_version,
        "target_version": cfg.target_version,
        "n_source": n_before,
        "n_target": len(samples),
        "filters": {
            "filter_bad_fires": cfg.filter_bad_fires,
            "min_fire_days": cfg.min_fire_days,
            "min_sample_year": cfg.min_sample_year,
            "require_canopy": cfg.require_canopy,
            "filter_inconsistent": cfg.filter_inconsistent,
            "filter_region": cfg.filter_region,
            "exclude_region": cfg.exclude_region,
            "filter_fire_size": cfg.filter_fire_size,
            "exclude_fire_size": cfg.exclude_fire_size,
        },
    }
    with open(target_path / "patch_meta.json", "w") as f:
        json.dump(patch_meta, f, indent=2)

    print(f"\nPatched: {n_before} → {len(samples)} samples")
    print(f"Version '{cfg.target_version}' written to {target_path}")


# =============================================================================
# distill — create a small fire-dense test dataset
# =============================================================================

def distill(cfg: PreprocessConfig):
    """
    Distill a test dataset by selecting fires with the most fire pixels.

    Strategy: randomly sample pool_size fires, keep top n_fires by VNP14
    pixel count.  Output is self-contained H5/JSON pairs.
    """
    import h5py
    from firecomp.config import config as path_config
    from firecomp.dsrc.vnp14 import Fires

    source_path = Path(cfg.source)
    dest_path = Path(cfg.dest)
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    samples = load_all_samples(source_path)
    src_h5_files = sorted(source_path.glob("dataset_*.h5"))

    fire_to_samples: dict[int, list] = defaultdict(list)
    for s in samples:
        fire_to_samples[s.fire_id].append(s)

    all_fire_ids = list(fire_to_samples.keys())
    print(f"Source: {source_path} ({len(samples)} samples, "
          f"{len(all_fire_ids)} fires, {len(src_h5_files)} shard"
          f"{'s' if len(src_h5_files) != 1 else ''})")

    pool_size = min(cfg.pool_size, len(all_fire_ids))
    rng = np.random.default_rng(cfg.seed)
    pool_ids = rng.choice(all_fire_ids, size=pool_size, replace=False).tolist()
    print(f"  Random pool: {pool_size} fires (seed={cfg.seed})")

    try:
        with h5py.File(path_config.vnp14_path, "r") as h5:
            stats = Fires.load(h5, "stats_1000")
        num_fire_by_id = dict(zip(stats.id.tolist(), stats.num_fire.tolist()))
    except (FileNotFoundError, OSError):
        print(f"  VNP14 not available; ranking by sample count per fire")
        num_fire_by_id = {fid: len(fire_to_samples[fid]) for fid in all_fire_ids}

    pool_ids.sort(key=lambda fid: num_fire_by_id.get(fid, 0), reverse=True)
    n_select = min(cfg.n_fires, len(pool_ids))
    selected_ids = pool_ids[:n_select]

    fire_counts = [num_fire_by_id.get(fid, 0) for fid in selected_ids]
    print(f"  Top {n_select} fires: min={min(fire_counts):,}  "
          f"max={max(fire_counts):,}  "
          f"mean={sum(fire_counts) // len(fire_counts):,}")

    selected_set = set(selected_ids)
    selected_samples = [s for s in samples if s.fire_id in selected_set]
    selected_samples.sort(key=lambda s: (s.fire_id, s.dt, s.xi, s.yi))
    print(f"  {len(selected_samples)} samples from {len(selected_set)} fires")

    dest_path.mkdir(parents=True, exist_ok=True)
    _write_distill_output(src_h5_files, dest_path, selected_samples,
                          n_files=cfg.n_files)
    print(f"\nDistilled dataset → {dest_path}")


# =============================================================================
# Stats accumulator — running mean/std per field
# =============================================================================

class _StatsAccumulator:
    """Welford-style running mean/std per field, dumped to stats.json."""

    def __init__(self, cfg: NextDayConfig):
        self._totals: dict[str, dict] = {}

    def update(self, field_name: str, data: np.ndarray):
        """data: (C, H, W). Accumulate per-channel mean / std."""
        if field_name not in self._totals:
            C = data.shape[0]
            self._totals[field_name] = {
                "n": 0, "mean": np.zeros(C, dtype=np.float64),
                "m2": np.zeros(C, dtype=np.float64),
            }
        s = self._totals[field_name]
        flat = data.reshape(data.shape[0], -1).astype(np.float64)
        for ch in range(flat.shape[0]):
            n_new = flat.shape[1]
            new_mean = flat[ch].mean()
            new_var = flat[ch].var()
            n_old = s["n"]
            n_total = n_old + n_new
            delta = new_mean - s["mean"][ch]
            s["mean"][ch] = (s["mean"][ch] * n_old + new_mean * n_new) / n_total
            s["m2"][ch] += new_var * n_new + delta**2 * n_old * n_new / n_total
        s["n"] += flat.shape[1]

    def dump(self, path: Path):
        out = {}
        for name, s in self._totals.items():
            std = np.sqrt(s["m2"] / max(s["n"], 1))
            out[name] = {
                "mean": s["mean"].tolist(),
                "std": np.maximum(std, 1e-6).tolist(),
            }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


# =============================================================================
# Patch filters — pure functions on sample lists
# =============================================================================

def _filter_inconsistent_fields(samples: list[Sample],
                                source_path: Path) -> list[Sample]:
    """Remove samples whose H5 fields have unexpected shapes."""
    from collections import Counter
    import hdf5plugin  # noqa: F401
    import h5py

    h5_path = (source_path / "dataset_0.h5").resolve()
    with h5py.File(h5_path, "r") as h5:
        field_shapes: dict[str, Counter] = defaultdict(Counter)
        for s in samples:
            grp = h5[str(s.idx)]
            for field_name in grp:
                field_shapes[field_name][grp[field_name].shape] += 1

        expected_shapes: dict[str, tuple] = {}
        for field_name, shapes in field_shapes.items():
            expected_shapes[field_name] = shapes.most_common(1)[0][0]

        inconsistent_fields = {
            name for name, shapes in field_shapes.items() if len(shapes) > 1
        }
        if inconsistent_fields:
            for name in sorted(inconsistent_fields):
                majority_shape, majority_count = field_shapes[name].most_common(1)[0]
                total = sum(field_shapes[name].values())
                print(f"    {name}: expected {majority_shape} "
                      f"({majority_count}/{total} match)")

        kept = []
        for s in samples:
            grp = h5[str(s.idx)]
            ok = True
            for field_name in grp:
                if field_name not in inconsistent_fields:
                    continue
                if grp[field_name].shape != expected_shapes[field_name]:
                    ok = False
                    break
            if ok:
                kept.append(s)

    return kept


def _filter_bad_fires(samples: list[Sample]) -> list[Sample]:
    """Remove samples from fires that fail quality checks."""
    import h5py
    from firecomp.config import config as path_config
    from firecomp.dsrc.vnp14 import Fires

    with h5py.File(path_config.vnp14_path, "r") as h5:
        stats = Fires.load(h5, "stats_100")
    bad_mask = ~Fires.filter_fires(stats, 0.7, 17.0, 1.0)
    bad_ids = set(stats.id[bad_mask].tolist())
    return [s for s in samples if s.fire_id not in bad_ids]


def _filter_min_days(samples: list[Sample], min_days: int) -> list[Sample]:
    """Remove samples from fires with fewer than min_days timesteps."""
    fire_dates: dict[int, set[str]] = defaultdict(set)
    for s in samples:
        fire_dates[s.fire_id].add(str(s.dt)[:10])
    bad_ids = {fid for fid, dates in fire_dates.items()
               if len(dates) < min_days}
    return [s for s in samples if s.fire_id not in bad_ids]


def _filter_region_and_size(
    samples: list[Sample],
    filter_region: str, exclude_region: str,
    filter_fire_size: str, exclude_fire_size: str,
) -> list[Sample]:
    """Filter by region name and/or fire size bucket."""
    import h5py
    from firecomp.config import config as path_config
    from firecomp.dsrc.vnp14 import Fires

    fire_stats: dict[int, int] = {}
    fire_regions: dict[int, int] = {}
    region_names: dict[int, str] = {}
    try:
        with h5py.File(path_config.vnp14_path, "r") as h5:
            stats = Fires.load(h5, "stats_100")
            fire_stats = dict(zip(stats.id.tolist(), stats.num_fire.tolist()))
    except Exception as e:
        print(f"  Warning: Could not load fire stats: {e}")

    try:
        from firecomp.core.regions import RegionRaster
        rr = RegionRaster.load()
        region_names = rr.names

        fire_centroids: dict[int, tuple[float, float]] = {}
        for s in samples:
            fire_centroids.setdefault(s.fire_id, (s.lon, s.lat))
        fids = np.array(list(fire_centroids.keys()))
        lons = np.array([fire_centroids[f][0] for f in fids])
        lats = np.array([fire_centroids[f][1] for f in fids])
        fire_regions = {int(f): int(r)
                       for f, r in zip(fids, rr.lookup_coords(lons, lats))}
    except Exception as e:
        print(f"  Warning: Could not load regions: {e}")

    name_to_id = {name: rid for rid, name in region_names.items()}

    def _to_ids(name):
        return {name_to_id[name]} if name and name in name_to_id else set()

    include_regions = _to_ids(filter_region)
    exclude_regions = _to_ids(exclude_region)
    include_sizes = {filter_fire_size} if filter_fire_size else set()
    exclude_sizes = {exclude_fire_size} if exclude_fire_size else set()

    SIZE_THRESHOLDS = [(100, "<100"), (1000, "100-1000"),
                       (10000, "1000-10000")]

    def _size_bucket(num_fire: int) -> str:
        for threshold, name in SIZE_THRESHOLDS:
            if num_fire < threshold:
                return name
        return ">10000"

    out = []
    for s in samples:
        rid = fire_regions.get(s.fire_id, 0)
        if include_regions and rid not in include_regions:
            continue
        if exclude_regions and rid in exclude_regions:
            continue
        bucket = _size_bucket(fire_stats.get(s.fire_id, 0))
        if include_sizes and bucket not in include_sizes:
            continue
        if exclude_sizes and bucket in exclude_sizes:
            continue
        out.append(s)
    return out


# =============================================================================
# I/O helpers
# =============================================================================

def _write_build_json(samples: list[Sample], meta_list: list[dict],
                      path: Path):
    """Write Phase 1 output JSON: sample fields + extra metadata."""
    records = []
    for s, meta in zip(samples, meta_list):
        d = {k: v for k, v in s.__dict__.items() if not k.startswith("_")}
        d.update(meta)
        records.append(d)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


def _load_build_json(path: Path) -> tuple[list[Sample], list[dict]]:
    """Read Phase 1 JSON back into Sample list + metadata list."""
    with open(path) as f:
        records = json.load(f)
    samples = []
    meta_list = []
    sample_fields = {"fire_id", "xi", "yi", "lon", "lat", "dt",
                     "img_size", "idx"}
    for d in records:
        sample_d = {k: v for k, v in d.items() if k in sample_fields}
        samples.append(Sample(**sample_d))
        meta_d = {k: v for k, v in d.items() if k not in sample_fields}
        meta_list.append(meta_d)
    return samples, meta_list


def _write_samples_json(samples: list[Sample], path: Path,
                        include_idx: bool = True):
    """Write sample list to JSON (used by patch/distill)."""
    raw = []
    for s in samples:
        d = {k: v for k, v in s.__dict__.items() if not k.startswith("_")}
        if not include_idx:
            d.pop("idx", None)
        raw.append(d)
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def _write_patch_output(source_path: Path, target_path: Path,
                        src_h5_files: list[Path], samples: list[Sample],
                        source_version: str):
    """Symlink each source H5 shard and write per-shard filtered JSONs."""
    by_shard: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_shard[getattr(s, '_h5_file', 0)].append(s)

    for k, src_h5 in enumerate(src_h5_files):
        shard_samples = by_shard.get(k, [])
        if not shard_samples:
            continue
        _symlink_file(target_path / f"dataset_{k}.h5",
                      source_path / f"dataset_{k}.h5",
                      source_version)
        _write_samples_json(shard_samples, target_path / f"dataset_{k}.json")
        print(f"  dataset_{k}: {len(shard_samples)} samples")


def _symlink_file(link: Path, source_file: Path, source_version: str):
    """Create a relative symlink from link → source_file."""
    if source_version == "latest":
        rel_target = Path("..") / source_file.name
    else:
        rel_target = Path("..") / source_version / source_file.name
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            print(f"  WARNING: {link} exists and is not a symlink, skipping")
            return
    link.symlink_to(rel_target)


# =============================================================================
# Distill helpers
# =============================================================================

def _write_distill_output(src_h5_files: list[Path], dest_path: Path,
                          samples: list[Sample], n_files: int):
    """Copy selected sample H5 groups into N new self-contained H5 files."""
    import h5py
    import hdf5plugin  # noqa: F401

    src_h5s = [h5py.File(p, "r") for p in src_h5_files]

    n = len(samples)
    chunk_size = (n + n_files - 1) // n_files
    groups = [samples[i * chunk_size:(i + 1) * chunk_size]
              for i in range(n_files)]
    groups = [g for g in groups if g]

    for k, group in enumerate(groups):
        h5_out = dest_path / f"dataset_{k}.h5"
        with h5py.File(h5_out, "w") as dst:
            if k == 0:
                for src in src_h5s:
                    if "statistics" in src:
                        src.copy("statistics", dst)
                        break

            for new_idx, s in enumerate(group):
                src = src_h5s[getattr(s, '_h5_file', 0)]
                src_key = str(s.idx)
                dst_key = str(new_idx)
                if src_key in src:
                    src.copy(src_key, dst, dst_key)
                s.idx = new_idx

            if k == 0:
                # Copy per-tile fields (e.g. vnp02_terrain) that are
                # stored in top-level groups keyed by tile_key.
                # AE fields (ae_pca_5, ae_12345) are now per-sample and
                # already copied above with the sample group.
                tile_keys = {s.tile_key for s in samples}
                for field in ("vnp02_terrain",):
                    for src in src_h5s:
                        if field not in src:
                            continue
                        if field not in dst:
                            dst.create_group(field)
                        for tk in tile_keys:
                            if tk in src[field] and tk not in dst[field]:
                                src.copy(f"{field}/{tk}", dst[field], tk)

        _write_samples_json(group, dest_path / f"dataset_{k}.json")
        print(f"  dataset_{k}.h5: {len(group)} samples "
              f"({h5_out.stat().st_size / 1e6:.1f} MB)")

    for h5 in src_h5s:
        h5.close()


# =============================================================================
# show — combined diagnostic grid (all 3 phases in one image)
# =============================================================================

def _show_cmd(cfg: PreprocessConfig):
    """Combined diagnostic: all 3 phases side-by-side for each sample."""
    if cfg.show <= 0:
        cfg.show = 5  # default to 5 if not specified
    _show_combined(cfg)


def _show_combined(cfg: PreprocessConfig):
    """Diagnostic grid: one row per phase (accum, lossmask, fields).

    Each sample produces one image with 3 rows:
        Row 0: Phase 1 — accum_t_min, accum_t_max, accum_t_count
        Row 1: Phase 2 — vnp14_t, vnp14_t1, cur_mask, next_mask, ignition_t1, loss_valid
        Row 2: Phase 3 — ae_pca_5, weather[0..5] or gfs[0..4]
    """
    import matplotlib.pyplot as plt
    from firecomp.config import FLOAT32_NODATA

    result = _pick_show_samples(cfg)
    if result is None:
        return
    pairs, h5 = result

    out_dir = Path(cfg.dataset_dir) / "show"
    out_dir.mkdir(exist_ok=True)

    def _obs_rgb(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        rgb[mask >= 7] = [1, 0.3, 0]         # fire = red/orange
        rgb[mask == 5] = [0.2, 0.6, 0.2]     # land = green
        rgb[mask == 4] = [0.7, 0.7, 0.7]     # cloud = gray
        rgb[mask == 3] = [0, 0.7, 0.7]       # water = cyan
        rgb[mask == 6] = [0.5, 0.5, 0.3]     # unclassified = khaki
        # 0-2 = black (not-processed/bowtie/missing)
        # 255 = black (no coverage)
        return rgb

    n_cols = 6  # max columns (Phase 2 has 6)

    for sample, grp in pairs:
        fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10))

        # ── Row 0: Phase 1 (accum) ──
        accum_fields = ["accum_t_min", "accum_t_max", "accum_t_count"]
        for ci, fname in enumerate(accum_fields):
            ax = axes[0, ci]
            if fname in grp:
                data = grp[fname][0]
                masked = np.ma.masked_where(data == -1, data)
                ax.imshow(masked, cmap="inferno", interpolation="nearest")
                ax.set_title(fname, fontsize=8)
            else:
                ax.set_title(f"{fname} (n/a)", fontsize=8)
            ax.axis("off")
        for ci in range(len(accum_fields), n_cols):
            axes[0, ci].axis("off")

        # ── Row 1: Phase 2 (lossmask) ──
        # Col 0: vnp14_t
        has_t = "vnp14_t" in grp
        has_t1 = "vnp14_t1" in grp
        mask_t = grp["vnp14_t"][0] if has_t else None
        mask_t1 = grp["vnp14_t1"][0] if has_t1 else None

        if has_t:
            axes[1, 0].imshow(_obs_rgb(mask_t), interpolation="nearest")
            n_obs = _is_clear_obs(mask_t).sum()
            axes[1, 0].set_title(f"vnp14_t ({n_obs})", fontsize=8)
        else:
            axes[1, 0].set_title("vnp14_t (n/a)", fontsize=8)
        axes[1, 0].axis("off")

        # Col 1: vnp14_t1
        if has_t1:
            axes[1, 1].imshow(_obs_rgb(mask_t1), interpolation="nearest")
            n_obs = _is_clear_obs(mask_t1).sum()
            axes[1, 1].set_title(f"vnp14_t1 ({n_obs})", fontsize=8)
        else:
            axes[1, 1].set_title("vnp14_t1 (n/a)", fontsize=8)
        axes[1, 1].axis("off")

        # Col 2: cur_mask
        if "cur_mask" in grp:
            axes[1, 2].imshow(grp["cur_mask"][0], cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[1, 2].set_title(f"cur_mask ({grp['cur_mask'][0].sum()}px)",
                                 fontsize=8)
        else:
            axes[1, 2].set_title("cur_mask (n/a)", fontsize=8)
        axes[1, 2].axis("off")

        # Col 3: next_mask
        if "next_mask" in grp:
            axes[1, 3].imshow(grp["next_mask"][0], cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[1, 3].set_title(f"next_mask ({grp['next_mask'][0].sum()}px)",
                                 fontsize=8)
        else:
            axes[1, 3].set_title("next_mask (n/a)", fontsize=8)
        axes[1, 3].axis("off")

        # Col 4: ignition_t1
        ign_t1 = None
        if "ignition_t1" in grp:
            ign_t1 = grp["ignition_t1"][0]
            axes[1, 4].imshow(ign_t1, cmap="hot", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[1, 4].set_title(f"ignition ({ign_t1.sum()}px)", fontsize=8)
        else:
            axes[1, 4].set_title("ignition (n/a)", fontsize=8)
        axes[1, 4].axis("off")

        # Col 5: loss_valid
        if has_t and has_t1:
            loss_valid = _is_clear_obs(mask_t) & _is_clear_obs(mask_t1)
            if ign_t1 is not None:
                loss_valid = loss_valid & (ign_t1 == 0)
            pct = 100 * loss_valid.mean()
            axes[1, 5].imshow(loss_valid.astype(np.float32), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
            axes[1, 5].set_title(f"loss_valid ({pct:.0f}%)", fontsize=8)
        else:
            axes[1, 5].set_title("loss_valid (n/a)", fontsize=8)
        axes[1, 5].axis("off")

        # ── Row 2: Phase 3 (fields) ──
        col = 0
        # AE pseudo-RGB (per-sample in v3, per-tile fallback for v2)
        tk = sample.tile_key
        for ae_name in ("ae_pca_5", "ae_12345"):
            if col >= n_cols:
                break
            if ae_name in grp:
                data = grp[ae_name][...]
            elif ae_name in h5 and tk in h5[ae_name]:
                data = h5[ae_name][tk][...]
            else:
                continue
            if data.shape[0] >= 3:
                rgb = data[:3].transpose(1, 2, 0).astype(np.float32)
                valid = rgb > FLOAT32_NODATA + 1
                for c in range(3):
                    ch = rgb[:, :, c]
                    v = ch[valid[:, :, c]]
                    if len(v) > 0:
                        ch -= v.min()
                        ch /= (v.max() - v.min() + 1e-8)
                    ch[~valid[:, :, c]] = 0
                axes[2, col].imshow(np.clip(rgb, 0, 1),
                                    interpolation="nearest")
            else:
                axes[2, col].imshow(data[0], cmap="viridis",
                                    interpolation="nearest")
            axes[2, col].set_title(ae_name, fontsize=8)
            axes[2, col].axis("off")
            col += 1

        # Weather / GFS: show first few channels until columns full
        for w_name in ("weather", "gfs"):
            if col >= n_cols:
                break
            if w_name in grp:
                data = grp[w_name][...]
                for ch_i in range(data.shape[0]):
                    if col >= n_cols:
                        break
                    ch = data[ch_i].astype(np.float32)
                    masked = np.ma.masked_where(
                        ch <= FLOAT32_NODATA + 1, ch)
                    axes[2, col].imshow(masked, cmap="viridis",
                                        interpolation="nearest")
                    axes[2, col].set_title(f"{w_name}[{ch_i}]", fontsize=8)
                    axes[2, col].axis("off")
                    col += 1

        for ci in range(col, n_cols):
            axes[2, ci].axis("off")

        # Row labels
        for ri, label in enumerate(["Phase 1: accum",
                                     "Phase 2: lossmask",
                                     "Phase 3: fields"]):
            axes[ri, 0].set_ylabel(label, fontsize=9, rotation=90,
                                    labelpad=10)

        fig.suptitle(f"fire={sample.fire_id}  dt={sample.dt}  "
                     f"tile=({sample.xi},{sample.yi})  idx={sample.idx}",
                     fontsize=11)
        fig.tight_layout()
        path = out_dir / f"combined_{sample.idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")

    h5.close()


# =============================================================================
# --show diagnostics — save sample images after each build phase
# =============================================================================

def _pick_show_samples(cfg: PreprocessConfig):
    """Load shard, pick N random samples.  Returns (pairs, h5) or None."""
    import h5py
    import hdf5plugin  # noqa: F401

    shard_json = Path(cfg.dataset_dir) / f"dataset_{cfg.ji}.json"
    samples, _ = _load_build_json(shard_json)
    if not samples:
        return None

    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.show, len(samples))
    chosen = rng.choice(len(samples), size=n, replace=False)

    h5 = h5py.File(Path(cfg.dataset_dir) / f"dataset_{cfg.ji}.h5", "r")
    pairs = [(samples[i], h5[str(samples[i].idx)]) for i in chosen]
    return pairs, h5


def _show_accum(cfg: PreprocessConfig):
    """Diagnostic images for Phase 1: accum_t + masks grid."""
    import matplotlib.pyplot as plt

    result = _pick_show_samples(cfg)
    if result is None:
        return
    pairs, h5 = result

    out_dir = Path(cfg.dataset_dir) / "show"
    out_dir.mkdir(exist_ok=True)

    for sample, grp in pairs:
        fields = ["accum_t_min", "accum_t_max", "accum_t_count"]
        present = [f for f in fields if f in grp]
        n = len(present)
        if n == 0:
            continue

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, fname in zip(axes, present):
            data = grp[fname][0]  # (1, H, W) → (H, W)
            if "accum_t" in fname:
                # Mask out -1 (no fire) for better contrast
                masked = np.ma.masked_where(data == -1, data)
                ax.imshow(masked, cmap="inferno", interpolation="nearest")
            else:
                ax.imshow(data, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
            ax.set_title(fname, fontsize=10)
            ax.axis("off")

        fig.suptitle(f"fire={sample.fire_id}  dt={sample.dt}  "
                     f"tile=({sample.xi},{sample.yi})", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"accum_{sample.idx}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  --show: {path}")

    h5.close()


def _show_lossmask(cfg: PreprocessConfig):
    """Diagnostic: vnp14_t, vnp14_t1, cur/next mask, ignition_t1, loss_valid."""
    import matplotlib.pyplot as plt

    result = _pick_show_samples(cfg)
    if result is None:
        return
    pairs, h5 = result

    out_dir = Path(cfg.dataset_dir) / "show"
    out_dir.mkdir(exist_ok=True)

    def _obs_rgb(mask):
        """Colorize observation mask: fire=red, land=green, water=cyan,
        cloud=gray, unclassified=khaki, black=no observation."""
        rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        rgb[mask >= 7] = [1, 0.3, 0]         # fire = red/orange
        rgb[mask == 5] = [0.2, 0.6, 0.2]     # land = green
        rgb[mask == 4] = [0.7, 0.7, 0.7]     # cloud = gray
        rgb[mask == 3] = [0, 0.7, 0.7]       # water = cyan
        rgb[mask == 6] = [0.5, 0.5, 0.3]     # unclassified = khaki
        # 0-2 = black (not-processed/bowtie/missing)
        # 255 = black (no coverage)
        return rgb

    for sample, grp in pairs:
        has_t = "vnp14_t" in grp
        has_t1 = "vnp14_t1" in grp
        if not has_t and not has_t1:
            continue

        fig, axes = plt.subplots(1, 6, figsize=(26, 4))

        # Col 0: vnp14_t (day_T observation)
        if has_t:
            mask_t = grp["vnp14_t"][0]
            axes[0].imshow(_obs_rgb(mask_t), interpolation="nearest")
            n_obs = _is_clear_obs(mask_t).sum()
            axes[0].set_title(f"vnp14_t  ({n_obs} obs)", fontsize=9)
        else:
            axes[0].set_title("vnp14_t  (missing)")
        axes[0].axis("off")

        # Col 1: vnp14_t1 (day_T+1 observation)
        if has_t1:
            mask_t1 = grp["vnp14_t1"][0]
            axes[1].imshow(_obs_rgb(mask_t1), interpolation="nearest")
            n_obs = _is_clear_obs(mask_t1).sum()
            axes[1].set_title(f"vnp14_t1  ({n_obs} obs)", fontsize=9)
        else:
            axes[1].set_title("vnp14_t1  (missing)")
        axes[1].axis("off")

        # Col 2: cur_mask (fire at day_T)
        if "cur_mask" in grp:
            axes[2].imshow(grp["cur_mask"][0], cmap="gray", vmin=0, vmax=1,
                           interpolation="nearest")
            n_fire = grp["cur_mask"][0].sum()
            axes[2].set_title(f"cur_mask ({n_fire}px)", fontsize=9)
        axes[2].axis("off")

        # Col 3: next_mask (fire at day_T+1)
        if "next_mask" in grp:
            axes[3].imshow(grp["next_mask"][0], cmap="gray", vmin=0, vmax=1,
                           interpolation="nearest")
            n_fire = grp["next_mask"][0].sum()
            axes[3].set_title(f"next_mask ({n_fire}px)", fontsize=9)
        axes[3].axis("off")

        # Col 4: ignition_t1 (unpredictable new fire starts)
        ign_t1 = None
        if "ignition_t1" in grp:
            ign_t1 = grp["ignition_t1"][0]
            axes[4].imshow(ign_t1, cmap="hot", vmin=0, vmax=1,
                           interpolation="nearest")
            n_ign = ign_t1.sum()
            axes[4].set_title(f"ignition_t1 ({n_ign}px)", fontsize=9)
        else:
            axes[4].set_title("ignition_t1 (n/a)")
        axes[4].axis("off")

        # Col 5: loss_valid = observed_t & observed_t1 & ~ignition_t1
        if has_t and has_t1:
            loss_valid = _is_clear_obs(mask_t) & _is_clear_obs(mask_t1)
            if ign_t1 is not None:
                loss_valid = loss_valid & (ign_t1 == 0)
            loss_valid = loss_valid.astype(np.float32)
            axes[5].imshow(loss_valid, cmap="gray", vmin=0, vmax=1,
                           interpolation="nearest")
            pct = 100 * loss_valid.mean()
            axes[5].set_title(f"loss_valid ({pct:.0f}%)", fontsize=9)
        else:
            axes[5].set_title("loss_valid (n/a)")
        axes[5].axis("off")

        fig.suptitle(f"fire={sample.fire_id}  dt={sample.dt}  "
                     f"tile=({sample.xi},{sample.yi})", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"lossmask_{sample.idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  --show: {path}")

    h5.close()


def _show_fields(cfg: PreprocessConfig):
    """Diagnostic image for Phase 3: AE terrain + weather channels.

    AE embeddings: pseudo-RGB from first 3 channels (similar value ranges).
    Weather/GFS: per-channel individual plots with nodata masking and
    per-channel normalization (different physical quantities per channel).
    """
    import matplotlib.pyplot as plt
    from firecomp.config import FLOAT32_NODATA

    result = _pick_show_samples(cfg)
    if result is None:
        return
    pairs, h5 = result

    out_dir = Path(cfg.dataset_dir) / "show"
    out_dir.mkdir(exist_ok=True)

    # AE fields → pseudo-RGB;  weather/gfs → per-channel
    ae_names = {"ae_pca_5", "ae_12345"}

    for sample, grp in pairs:
        # Collect (name, data) pairs
        # AE: per-sample (v3) with per-tile fallback (v2)
        all_fields = []
        tk = sample.tile_key
        for name in ("ae_pca_5", "ae_12345"):
            if name in grp:
                all_fields.append((name, grp[name][...]))
            elif name in h5 and tk in h5[name]:
                all_fields.append((name, h5[name][tk][...]))

        for name in ("weather", "gfs"):
            if name in grp:
                all_fields.append((name, grp[name][...]))
        if not all_fields:
            continue

        # Count columns: AE fields get 1 col each, weather/gfs get 1 per channel
        n_cols = 0
        for name, data in all_fields:
            n_cols += 1 if name in ae_names else data.shape[0]

        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
        if n_cols == 1:
            axes = [axes]

        col = 0
        for name, data in all_fields:
            if name in ae_names:
                # AE embeddings: pseudo-RGB from ch0-2
                ax = axes[col]
                if data.shape[0] >= 3:
                    rgb = data[:3].transpose(1, 2, 0).astype(np.float32)
                    valid = rgb > FLOAT32_NODATA + 1
                    for c in range(3):
                        ch = rgb[:, :, c]
                        v = ch[valid[:, :, c]]
                        if len(v) > 0:
                            ch -= v.min()
                            ch /= (v.max() - v.min() + 1e-8)
                        ch[~valid[:, :, c]] = 0
                    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
                else:
                    ax.imshow(data[0], cmap="viridis", interpolation="nearest")
                ax.set_title(name, fontsize=9)
                ax.axis("off")
                col += 1
            else:
                # Weather/GFS: one subplot per channel, nodata-masked
                for ch_i in range(data.shape[0]):
                    ax = axes[col]
                    ch = data[ch_i].astype(np.float32)
                    masked = np.ma.masked_where(ch <= FLOAT32_NODATA + 1, ch)
                    ax.imshow(masked, cmap="viridis", interpolation="nearest")
                    ax.set_title(f"{name}[{ch_i}]", fontsize=9)
                    ax.axis("off")
                    col += 1

        fig.suptitle(f"fire={sample.fire_id}  dt={sample.dt}  "
                     f"tile=({sample.xi},{sample.yi})", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"fields_{sample.idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  --show: {path}")

    h5.close()


# =============================================================================

if __name__ == "__main__":
    main()
