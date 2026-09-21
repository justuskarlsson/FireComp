"""
next_day/solidity.py — Fire shape & spread analysis.

Three subcommands:

  top-n     Original analysis: metrics for N largest vegetation fires per region.
  survey    Parallel scan of ALL vegetation fires, threshold summary table.
            With --n-samples renders fire event grids per region.
  test-set  Solidity over the fire events present in the deterministic test
            split (8-connected components) — values correlate with test F1.
  patches   Solidity on actual 256x256 dataset patches (what the model sees).
            Also renders N random samples per region as visual grids.

Usage:
    python -m firecomp.next_day.solidity top-n
    python -m firecomp.next_day.solidity top-n --top-n 200 --min-pixels 50

    python -m firecomp.next_day.solidity survey --workers 16
    python -m firecomp.next_day.solidity survey --workers 8 --min-pixels 50
    python -m firecomp.next_day.solidity survey --n-samples 5

    python -m firecomp.next_day.solidity test-set --workers 16

    python -m firecomp.next_day.solidity patches
    python -m firecomp.next_day.solidity patches --dataset-dir data/next_day_v3 --n-samples 5
    python -m firecomp.next_day.solidity patches --no-viz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull, cKDTree
from tqdm import tqdm

from firecomp.config import config
from firecomp.core.fire_filter import FireType, classify_fires_from_h5
from firecomp.core.regions import RegionRaster
from firecomp.dsrc.vnp14 import Fires
from firecomp.next_day.config import NextDayConfig


# ── main ────────────────────────────────────────────────────────────────

def main():
    """Dispatch to top-n or survey subcommand."""
    args = _parse_args()
    args.func(args)


# ── top-n (original) ────────────────────────────────────────────────────

def cmd_top_n(args):
    """Compute solidity for top-N vegetation fires per region, write JSON."""
    fire_ids, fire_types = classify_fires_from_h5()
    veg_ids = set(fire_ids[fire_types == FireType.VEGETATION].tolist())
    print(f"Vegetation fires: {len(veg_ids):,}", file=sys.stderr)

    rr = RegionRaster.load()

    with h5py.File(config.vnp14_path, "r") as h5:
        stats: Fires.Stats = Fires.load(h5, "stats")
        region_ids = rr.lookup_fires(stats)

        by_region: dict[int, list[int]] = defaultdict(list)
        for i, (fid, rid, npx) in enumerate(
            zip(stats.id, region_ids, stats.num_fire)
        ):
            if rid > 0 and int(fid) in veg_ids and npx >= args.min_pixels:
                by_region[int(rid)].append(i)

        results = []
        for rid in sorted(by_region):
            idxs = by_region[rid]
            idxs.sort(key=lambda i: stats.num_fire[i], reverse=True)
            idxs = idxs[: args.top_n]

            for i in idxs:
                fid = int(stats.id[i])
                proj = Fires.load(h5, "projection_by_fire", str(fid))
                if proj is None:
                    continue
                m = _compute_metrics(proj)
                m["fire_id"] = fid
                m["region_id"] = int(rid)
                m["region_name"] = rr.names.get(int(rid), f"unknown_{rid}")
                results.append(m)

            rname = rr.names.get(int(rid), f"unknown_{rid}")
            print(f"  {rname:20s}: {len(idxs)} fires", file=sys.stderr)

    _write_json(results, args.output)


# ── survey (parallel, all fires) ────────────────────────────────────────

def cmd_survey(args):
    """Parallel solidity+spread scan of all vegetation fires."""
    # 1. Build work list (single-threaded, fast)
    work, region_names = _build_work_list(args.min_pixels)
    print(f"Fires to process: {len(work):,}", file=sys.stderr)

    # 2. Parallel compute — each worker opens its own H5
    vnp14_path = str(config.vnp14_path)
    with Pool(args.workers, initializer=SurveyWorker.init,
              initargs=(vnp14_path,)) as pool:
        results = []
        it = pool.imap_unordered(SurveyWorker.process, work, chunksize=64)
        for r in tqdm(it, total=len(work), desc="survey", file=sys.stderr):
            if r is not None:
                results.append(r)

    print(f"\nComputed metrics for {len(results):,} fires", file=sys.stderr)

    # 3. Write full results JSON
    _write_json(results, args.output)

    # 4. Print threshold summary tables
    _print_threshold_table(results, region_names)
    _print_spread_table(results, region_names)

    # 5. Viz: N fire events per region, stratified by spread_ratio
    if args.n_samples > 0:
        from firecomp.core.plotting import save_figure

        viz_dir = str(Path(args.output).parent / "survey_viz")
        by_region: dict[int, list[dict]] = defaultdict(list)
        for r in results:
            by_region[r["region_id"]].append(r)

        with h5py.File(vnp14_path, "r") as h5:
            for rid in sorted(by_region):
                fires = by_region[rid]
                picked = _pick_stratified(fires, args.n_samples,
                                          key="spread_ratio")
                rname = region_names.get(rid, f"region_{rid}")
                fig = _render_fire_events(h5, picked, rname)
                out = f"{viz_dir}/{rname.replace('.', '').replace(' ', '_')}"
                save_figure(fig, out, formats=("png",))

        print(f"Viz saved to {viz_dir}/", file=sys.stderr)


def _build_work_list(min_pixels: int) -> tuple[list[tuple], dict[int, str]]:
    """Identify all vegetation fires with >= min_pixels detections.

    Returns (work_items, region_names) where each work item is
    (fire_id, region_id, region_name, num_fire).
    """
    fire_ids, fire_types = classify_fires_from_h5()
    veg_ids = set(fire_ids[fire_types == FireType.VEGETATION].tolist())
    print(f"Vegetation fires (all sizes): {len(veg_ids):,}", file=sys.stderr)

    rr = RegionRaster.load()
    region_names = dict(rr.names)

    with h5py.File(config.vnp14_path, "r") as h5:
        stats: Fires.Stats = Fires.load(h5, "stats")

    region_ids = rr.lookup_fires(stats)

    work = []
    for fid, rid, npx in zip(stats.id, region_ids, stats.num_fire):
        fid, rid, npx = int(fid), int(rid), int(npx)
        if rid > 0 and fid in veg_ids and npx >= min_pixels:
            rname = region_names.get(rid, f"unknown_{rid}")
            work.append((fid, rid, rname, npx))

    return work, region_names


class SurveyWorker:
    """Multiprocessing worker — each process gets its own H5 handle."""

    @staticmethod
    def init(vnp14_path: str):
        global _h5
        _h5 = h5py.File(vnp14_path, "r")

    @staticmethod
    def process(item: tuple) -> dict | None:
        """Compute solidity + spread coherence for one fire."""
        fire_id, region_id, region_name, num_fire = item
        proj = Fires.load(_h5, "projection_by_fire", str(fire_id))
        if proj is None:
            return None

        m = _compute_metrics(proj)
        m["fire_id"] = fire_id
        m["region_id"] = region_id
        m["region_name"] = region_name
        m["num_fire"] = num_fire
        m["n_days"] = int(len(np.unique(proj.t)))
        return m


# ── test-set (fires present in the deterministic test split) ────────────

def cmd_test_set(args):
    """Solidity over the fire events that appear in the test split.

    Filters to the deterministic test-split ``fire_id`` values so the
    solidity values correlate with the test-set F1 results in the transfer
    table.  Each fire's full ``projection_by_fire`` is loaded and its
    solidity + number of 8-connected components computed.  Output schema
    matches ``solidity.json`` so ``tables.table_solidity`` consumes it.
    """
    fid_region, region_names = _test_fire_regions(args.dataset_dir)
    print(f"Test-split fire events: {len(fid_region):,}", file=sys.stderr)

    work = [(fid, rid, region_names.get(rid, f"unknown_{rid}"))
            for fid, rid in fid_region.items()]

    vnp14_path = str(config.vnp14_path)
    with Pool(args.workers, initializer=TestSetWorker.init,
              initargs=(vnp14_path,)) as pool:
        results = []
        it = pool.imap_unordered(TestSetWorker.process, work, chunksize=64)
        for r in tqdm(it, total=len(work), desc="test-set", file=sys.stderr):
            if r is not None:
                results.append(r)

    print(f"\nComputed metrics for {len(results):,} fires", file=sys.stderr)
    _write_json(results, args.output)


def _test_fire_regions(dataset_dir: str) -> tuple[dict[int, int], dict[int, str]]:
    """Map each unique test-split ``fire_id`` to its region id.

    Returns ``(fire_id -> region_id, region_id -> region_name)``.
    """
    from firecomp.next_day.dataset import NextDayDataset

    cfg = NextDayConfig(dataset_dir=dataset_dir)
    ds = NextDayDataset(cfg)

    fid_region: dict[int, int] = {}
    for s in ds.test_samples:
        if s.region_id > 0:
            fid_region.setdefault(int(s.fire_id), int(s.region_id))

    rr = RegionRaster.load()
    return fid_region, dict(rr.names)


class TestSetWorker:
    """Multiprocessing worker — solidity + 8-connected components per fire."""

    @staticmethod
    def init(vnp14_path: str):
        global _ts_h5
        _ts_h5 = h5py.File(vnp14_path, "r")

    @staticmethod
    def process(item: tuple) -> dict | None:
        fire_id, region_id, region_name = item
        proj = Fires.load(_ts_h5, "projection_by_fire", str(fire_id))
        if proj is None:
            return None

        # Minimal bounding-box binary mask for the whole fire event.
        x, y = proj.x, proj.y
        x0, y0 = int(x.min()), int(y.min())
        w = int(x.max()) - x0 + 1
        h = int(y.max()) - y0 + 1
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y - y0, x - x0] = 1

        m = _compute_metrics_from_mask(mask, connectivity=2)
        m["fire_id"] = fire_id
        m["region_id"] = region_id
        m["region_name"] = region_name
        return m


# ── patches (dataset-level, what the model sees) ───────────────────────

def cmd_patches(args):
    """Compute solidity on actual 256x256 dataset patches, per-region summary.

    Computes metrics on the full accum_t AND on just the sample's fire_id
    component (via VNP14 projection_by_fire).  Also renders N random samples
    per region as visual grids (unless --no-viz).
    """
    import random as rng
    from firecomp.core.dataset_utils import SampleStore
    from firecomp.next_day.dataset import Sample

    store = SampleStore(Path(args.dataset_dir), sample_cls=Sample)
    all_samples = store.load_samples()
    print(f"Loaded {len(all_samples):,} samples from {args.dataset_dir}",
          file=sys.stderr)

    # VNP14 path for fire_id filtering
    vnp14_path = str(config.vnp14_path)

    # Build work list — include lon/lat for coordinate mapping
    h5_paths = [str(p) for p in store._h5_paths]
    work = []
    for s in all_samples:
        work.append((
            h5_paths[getattr(s, '_h5_file', 0)],
            s.idx,
            s.fire_id,
            s.region_id,
            getattr(s, 'fire_type', -1),
            s.dt,
            s.lon,
            s.lat,
        ))

    # Parallel compute — workers get both dataset H5s and VNP14
    with Pool(args.workers, initializer=PatchWorker.init,
              initargs=(vnp14_path,)) as pool:
        results = []
        it = pool.imap_unordered(PatchWorker.process, work, chunksize=64)
        for r in tqdm(it, total=len(work), desc="patches", file=sys.stderr):
            if r is not None:
                results.append(r)

    print(f"\nComputed patch metrics for {len(results):,} samples",
          file=sys.stderr)

    # Write full results JSON
    _write_json(results, args.output)

    # Collect region names
    region_names = {}
    try:
        rr = RegionRaster.load()
        region_names = dict(rr.names)
    except Exception:
        for r in results:
            region_names.setdefault(r["region_id"], f"region_{r['region_id']}")

    # Print tables: full-patch and fire_id-only
    print("\n=== Full patch (all fire pixels) ===", file=sys.stderr)
    _print_threshold_table(results, region_names, day_key=None)
    _print_component_table(results, region_names)

    print("\n=== Fire-ID only (main fire component) ===", file=sys.stderr)
    _print_threshold_table(results, region_names, day_key=None,
                           sol_key="fire_solidity")
    _print_fire_frac_table(results, region_names)

    # Viz: N random samples per region
    if not args.no_viz:
        from firecomp.core.plotting import save_figure
        rng.seed(args.seed)
        by_region: dict[int, list] = defaultdict(list)
        for s in all_samples:
            by_region[s.region_id].append(s)

        viz_dir = str(Path(args.output).parent / "solidity_viz")
        with h5py.File(vnp14_path, "r") as h5_vnp:
            for rid in sorted(by_region):
                samples = by_region[rid]
                picked = rng.sample(samples,
                                    min(args.n_samples * 5, len(samples)))
                chosen = []
                for s in picked:
                    if len(chosen) >= args.n_samples:
                        break
                    raw = store.load(s)
                    akey = "accum_t_min" if "accum_t_min" in raw else "accum_t"
                    accum = raw[akey].squeeze()
                    if (accum >= 0).sum() > 0:
                        fmask = _fire_id_mask(h5_vnp, s.fire_id,
                                              s.lon, s.lat)
                        chosen.append((s, raw, fmask))
                if not chosen:
                    continue
                rname = region_names.get(rid, f"region_{rid}")
                fig = _render_region_grid(chosen, rname, args.padding)
                out = f"{viz_dir}/{rname.replace('.', '').replace(' ', '_')}"
                save_figure(fig, out, formats=("png",))

        print(f"Viz saved to {viz_dir}/", file=sys.stderr)

    store.close()


# ── fire-ID mask helper ────────────────────────────────────────────────

def _fire_id_mask(
    h5_vnp14, fire_id: int, lon: float, lat: float, img_size: int = 256,
) -> np.ndarray | None:
    """Binary mask of fire_id's pixels within a 256x256 dataset patch.

    Maps VNP14 global pixel coordinates to local patch coordinates using
    the patch center (lon, lat).  Each pixel is DEG_CELL_SIZE apart.

    Returns (img_size, img_size) uint8 array or None if projection missing.
    """
    proj = Fires.load(h5_vnp14, "projection_by_fire", str(fire_id))
    if proj is None:
        return None

    D = DEG_CELL_SIZE
    half = img_size // 2  # 128

    # VNP14 global pixel coords → lon/lat → local patch coords
    fire_lon = proj.x.astype(np.float64) * D - 180.0
    fire_lat = proj.y.astype(np.float64) * D - 90.0

    local_col = np.round((fire_lon - lon) / D).astype(np.int32) + half
    local_row = np.round((lat - fire_lat) / D).astype(np.int32) + half

    in_patch = ((local_col >= 0) & (local_col < img_size) &
                (local_row >= 0) & (local_row < img_size))

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    if in_patch.any():
        mask[local_row[in_patch], local_col[in_patch]] = 1
    return mask


class PatchWorker:
    """Multiprocessing worker for patch-level solidity (full + fire_id)."""

    @staticmethod
    def init(vnp14_path: str | None = None):
        global _patch_h5_cache, _vnp14_h5
        _patch_h5_cache = {}
        _vnp14_h5 = h5py.File(vnp14_path, "r") if vnp14_path else None

    @staticmethod
    def _get_h5(h5_path: str):
        global _patch_h5_cache
        if h5_path not in _patch_h5_cache:
            import hdf5plugin  # noqa: F401
            _patch_h5_cache[h5_path] = h5py.File(h5_path, "r")
        return _patch_h5_cache[h5_path]

    @staticmethod
    def process(item: tuple) -> dict | None:
        """Compute solidity on accum_t (full) + fire_id only."""
        (h5_path, sample_idx, fire_id, region_id,
         fire_type, dt, lon, lat) = item
        try:
            h5 = PatchWorker._get_h5(h5_path)
            grp = h5[str(sample_idx)]

            # Full accum_t mask (all fire in the patch)
            if "accum_t_min" in grp:
                accum = grp["accum_t_min"][:].squeeze()
            else:
                accum = grp["accum_t"][:].squeeze()

            full_mask = (accum >= 0).astype(np.uint8)
            full_px = int(full_mask.sum())

            if full_px == 0:
                return {
                    "fire_id": fire_id, "region_id": region_id,
                    "fire_type": fire_type, "dt": dt,
                    "solidity": 0.0, "n_components": 0,
                    "num_pixels": 0, "mean_component_size": 0.0,
                    "fire_solidity": 0.0, "fire_n_components": 0,
                    "fire_num_pixels": 0, "fire_frac": 0.0,
                }

            m = _compute_metrics_from_mask(full_mask)
            m["fire_id"] = fire_id
            m["region_id"] = region_id
            m["fire_type"] = fire_type
            m["dt"] = dt

            # Fire-ID only metrics (via VNP14 projection)
            global _vnp14_h5
            if _vnp14_h5 is not None:
                fmask = _fire_id_mask(_vnp14_h5, fire_id, lon, lat)
                if fmask is not None and fmask.sum() > 0:
                    fm = _compute_metrics_from_mask(fmask)
                    m["fire_solidity"] = fm["solidity"]
                    m["fire_n_components"] = fm["n_components"]
                    m["fire_num_pixels"] = fm["num_pixels"]
                    m["fire_frac"] = round(fm["num_pixels"] / full_px, 4)
                else:
                    m["fire_solidity"] = 0.0
                    m["fire_n_components"] = 0
                    m["fire_num_pixels"] = 0
                    m["fire_frac"] = 0.0
            return m
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}", file=sys.stderr)
            return None


# ── viz helpers (used by patches) ───────────────────────────────────────

DEG_CELL_SIZE = 375 / 111_320  # ~0.00337 deg per pixel


def _render_region_grid(
    chosen: list[tuple],  # [(Sample, raw_dict, fire_id_mask|None), ...]
    region_name: str,
    padding: int,
) -> "plt.Figure":
    """Render one region's grid.

    Columns: accum_t (all) | fire_id only | satellite | loss_mask | next_mask
    """
    import matplotlib.pyplot as plt
    from firecomp.core.plotting import (
        Cmaps, Style, add_satellite_basemap, imshow_tensor,
    )
    from firecomp.next_day.dataset import compute_loss_mask

    nrows = len(chosen)
    ncols = 5  # accum_t | fire_id | satellite | loss_mask | next_mask

    try:
        import cartopy.crs as ccrs
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    cell_w, cell_h = 3.2, 3.5
    fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows + 1.2),
                     constrained_layout=True)
    proj = ccrs.PlateCarree() if has_cartopy else None

    gs = fig.add_gridspec(nrows, ncols)

    for row, item in enumerate(chosen):
        sample, raw = item[0], item[1]
        fid_mask = item[2] if len(item) > 2 else None

        # Full accum_t
        accum_key = "accum_t_min" if "accum_t_min" in raw else "accum_t"
        accum = raw[accum_key].squeeze().astype(np.float32)
        all_fire = (accum >= 0).astype(np.uint8)
        n_all = int(all_fire.sum())

        # Fire-ID metrics
        if fid_mask is not None and fid_mask.sum() > 0:
            fm = _compute_metrics_from_mask(fid_mask)
            f_sol = fm["solidity"]
            f_comp = fm["n_components"]
            f_px = fm["num_pixels"]
            f_frac = f_px / max(n_all, 1)
        else:
            f_sol, f_comp, f_px, f_frac = 0.0, 0, 0, 0.0

        cur = raw["cur_mask"].squeeze()
        nxt = raw["next_mask"].squeeze()

        title_0 = (
            f"fire {sample.fire_id}   {sample.dt[:10]}\n"
            f"all: {n_all}px   fire_id: {f_px}px ({f_frac:.0%})\n"
            f"sol={f_sol:.2f}  comp={f_comp}"
        )

        # --- Col 0: accum_t (all fires) ---
        ax0 = fig.add_subplot(gs[row, 0])
        cmap_accum = Cmaps.fire_spread.copy()
        cmap_accum.set_bad(color="#f0f0f0")
        accum_disp = accum.copy()
        accum_disp[accum < 0] = np.nan
        imshow_tensor(ax0, accum_disp, cmap=cmap_accum, nodata=np.nan)
        ax0.set_title(title_0, fontsize=8, pad=Style.title_pad, loc="left")

        # --- Col 1: fire_id only (masked accum_t) ---
        ax1 = fig.add_subplot(gs[row, 1])
        if fid_mask is not None and fid_mask.sum() > 0:
            fid_accum = accum.copy()
            fid_accum[fid_mask == 0] = np.nan  # hide non-fire_id pixels
            fid_accum[accum < 0] = np.nan
            imshow_tensor(ax1, fid_accum, cmap=cmap_accum, nodata=np.nan)
        else:
            ax1.text(0.5, 0.5, "no fire_id\nprojection",
                     ha="center", va="center", transform=ax1.transAxes,
                     fontsize=10, color="#999999")
            ax1.axis("off")
        if row == 0:
            ax1.set_title("fire_id only", fontsize=Style.title_fontsize,
                          pad=Style.title_pad)

        # --- Col 2: satellite basemap with fire overlay ---
        if has_cartopy:
            ax2 = fig.add_subplot(gs[row, 2], projection=proj)
            half_deg = 128 * DEG_CELL_SIZE
            lon, lat = sample.lon, sample.lat
            extent = [lon - half_deg, lon + half_deg,
                      lat - half_deg, lat + half_deg]
            try:
                add_satellite_basemap(ax2, extent)
            except Exception:
                pass
            # Overlay cur_mask (orange) and next_mask (yellow)
            fire_rgba = np.zeros((*cur.shape, 4), dtype=np.uint8)
            fire_rgba[cur > 0] = [255, 80, 0, 180]
            fire_rgba[nxt > 0] = [255, 255, 0, 150]
            ax2.imshow(fire_rgba, extent=extent, origin="upper",
                       transform=ccrs.PlateCarree(), zorder=5)
            if row == 0:
                ax2.set_title("satellite", fontsize=Style.title_fontsize,
                              pad=Style.title_pad)
        else:
            ax2 = fig.add_subplot(gs[row, 2])
            fire_rgb = np.zeros((*cur.shape, 3), dtype=np.uint8)
            fire_rgb[cur > 0] = [255, 80, 0]
            fire_rgb[nxt > 0] = [255, 255, 0]
            imshow_tensor(ax2, fire_rgb,
                          title="satellite" if row == 0 else None)
        ax2.axis("off")

        # --- Col 3: loss mask ---
        loss_mask = compute_loss_mask(raw, padding=padding)
        ax3 = fig.add_subplot(gs[row, 3])
        imshow_tensor(ax3, loss_mask, cmap="gray", vmin=0, vmax=1,
                      title="loss mask" if row == 0 else None)

        # --- Col 4: next_mask (target) ---
        ax4 = fig.add_subplot(gs[row, 4])
        imshow_tensor(ax4, nxt, cmap="Reds", vmin=0, vmax=1,
                      title="next_mask" if row == 0 else None)

    fig.suptitle(f"{region_name} — patches (fire_id vs all)",
                 fontsize=Style.suptitle_fontsize, fontweight="bold")
    return fig


# ── threshold summary table ─────────────────────────────────────────────

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _print_threshold_table(results: list[dict], region_names: dict[int, str],
                           day_key: str | None = "n_days",
                           sol_key: str = "solidity"):
    """Print region x solidity-threshold table.

    Args:
        sol_key: dict key for solidity (default "solidity", use
                 "fire_solidity" for fire-ID-only metrics).
        day_key: if set, shows "count(days_d)".  None = count only.
    """
    # Filter to results that have the requested key
    valid = [r for r in results if sol_key in r]
    if not valid:
        return

    by_region: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_region[r["region_id"]].append(r)

    has_days = day_key is not None

    def _cell(items: list[dict]) -> str:
        n = len(items)
        if has_days:
            d = sum(f.get(day_key, 0) for f in items)
            return f"{n:,}({d:,}d)"
        return f"{n:,}"

    label = "Solidity" if sol_key == "solidity" else "Fire solidity"
    hdr_thresh = "".join(f"{'>='+str(t):>14s}" for t in THRESHOLDS)
    print(f"\n{label:>14s}{hdr_thresh}")
    print(f"{'Region':>14s}{hdr_thresh}")
    print("-" * (14 + 14 * len(THRESHOLDS)))

    for rid in sorted(by_region):
        fires = by_region[rid]
        rname = region_names.get(rid, f"?{rid}")
        cells = [_cell([f for f in fires if f.get(sol_key, 0) >= t])
                 for t in THRESHOLDS]
        row = "".join(f"{c:>14s}" for c in cells)
        print(f"{rname:>14s}{row}")

    # Totals row
    cells = [_cell([f for f in valid if f.get(sol_key, 0) >= t])
             for t in THRESHOLDS]
    row = "".join(f"{c:>14s}" for c in cells)
    print("-" * (14 + 14 * len(THRESHOLDS)))
    print(f"{'TOTAL':>14s}{row}")


# ── component-count summary table ───────────────────────────────────────

COMP_BUCKETS = [1, 2, 5, 10, 20, 50, 100]


def _print_component_table(results: list[dict], region_names: dict[int, str]):
    """Print region x n_components bucket table (patch-level)."""
    by_region: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        if r["n_components"] > 0:  # skip empty patches
            by_region[r["region_id"]].append(r)

    # Buckets: <=1, <=2, <=5, <=10, <=20, <=50, <=100, >100
    labels = [f"<={b}" for b in COMP_BUCKETS] + [f">{COMP_BUCKETS[-1]}"]
    hdr = "".join(f"{l:>10s}" for l in labels) + f"{'median':>10s}{'mean':>10s}"
    print(f"\n{'Region':>14s}{hdr}")
    print("-" * (14 + 10 * (len(labels) + 2)))

    def _row(items: list[dict]) -> str:
        comps = [r["n_components"] for r in items]
        cells = []
        for b in COMP_BUCKETS:
            n = sum(1 for c in comps if c <= b)
            pct = 100 * n / len(comps) if comps else 0
            cells.append(f"{pct:.0f}%")
        n_over = sum(1 for c in comps if c > COMP_BUCKETS[-1])
        pct_over = 100 * n_over / len(comps) if comps else 0
        cells.append(f"{pct_over:.0f}%")
        med = float(np.median(comps)) if comps else 0
        avg = float(np.mean(comps)) if comps else 0
        cells.append(f"{med:.0f}")
        cells.append(f"{avg:.1f}")
        return "".join(f"{c:>10s}" for c in cells)

    for rid in sorted(by_region):
        rname = region_names.get(rid, f"?{rid}")
        print(f"{rname:>14s}{_row(by_region[rid])}")

    print("-" * (14 + 10 * (len(labels) + 2)))
    all_non_empty = [r for r in results if r["n_components"] > 0]
    print(f"{'TOTAL':>14s}{_row(all_non_empty)}")


# ── fire_frac summary table ────────────────────────────────────────────

FRAC_BUCKETS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def _print_fire_frac_table(results: list[dict], region_names: dict[int, str]):
    """Print fire_frac distribution: what fraction of patch pixels belong
    to the sample's fire_id."""
    valid = [r for r in results if "fire_frac" in r and r["fire_frac"] > 0]
    if not valid:
        print("\n(no fire_frac data — VNP14 unavailable?)", file=sys.stderr)
        return

    by_region: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_region[r["region_id"]].append(r)

    labels = [f"<={b}" for b in FRAC_BUCKETS[1:]]
    hdr = "".join(f"{l:>10s}" for l in labels)
    hdr += f"{'median':>10s}{'mean':>10s}"
    print(f"\n{'fire_frac':>14s}{hdr}")
    print(f"{'Region':>14s}{hdr}")
    print("-" * (14 + 10 * (len(labels) + 2)))

    def _row(items: list[dict]) -> str:
        fracs = [r["fire_frac"] for r in items]
        cells = []
        for b in FRAC_BUCKETS[1:]:
            n = sum(1 for f in fracs if f <= b)
            pct = 100 * n / len(fracs) if fracs else 0
            cells.append(f"{pct:.0f}%")
        med = float(np.median(fracs)) if fracs else 0
        avg = float(np.mean(fracs)) if fracs else 0
        cells.append(f"{med:.2f}")
        cells.append(f"{avg:.2f}")
        return "".join(f"{c:>10s}" for c in cells)

    for rid in sorted(by_region):
        rname = region_names.get(rid, f"?{rid}")
        print(f"{rname:>14s}{_row(by_region[rid])}")

    print("-" * (14 + 10 * (len(labels) + 2)))
    print(f"{'TOTAL':>14s}{_row(valid)}")


# ── spread_ratio summary table ─────────────────────────────────────────

SPREAD_THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


def _print_spread_table(results: list[dict], region_names: dict[int, str]):
    """Print region × spread_ratio threshold table (survey-level)."""
    # Only fires that have the spread_ratio field
    valid = [r for r in results if "spread_ratio" in r]
    if not valid:
        return

    by_region: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_region[r["region_id"]].append(r)

    has_days = "n_days" in valid[0]

    def _cell(items: list[dict]) -> str:
        n = len(items)
        if has_days:
            d = sum(f.get("n_days", 0) for f in items)
            return f"{n:,}({d:,}d)"
        return f"{n:,}"

    hdr = "".join(f"{'>='+str(t):>14s}" for t in SPREAD_THRESHOLDS)
    print(f"\n{'Spread ratio':>14s}{hdr}")
    print(f"{'Region':>14s}{hdr}")
    print("-" * (14 + 14 * len(SPREAD_THRESHOLDS)))

    for rid in sorted(by_region):
        fires = by_region[rid]
        rname = region_names.get(rid, f"?{rid}")
        cells = [_cell([f for f in fires if f["spread_ratio"] >= t])
                 for t in SPREAD_THRESHOLDS]
        row = "".join(f"{c:>14s}" for c in cells)
        print(f"{rname:>14s}{row}")

    cells = [_cell([f for f in valid if f["spread_ratio"] >= t])
             for t in SPREAD_THRESHOLDS]
    row = "".join(f"{c:>14s}" for c in cells)
    print("-" * (14 + 14 * len(SPREAD_THRESHOLDS)))
    print(f"{'TOTAL':>14s}{row}")


# ── survey viz helpers ─────────────────────────────────────────────────

def _pick_stratified(
    fires: list[dict], n: int, key: str = "spread_ratio",
) -> list[dict]:
    """Pick N fires spanning the full range of *key* (quantile sampling)."""
    fires = sorted(fires, key=lambda f: f.get(key, 0))
    if len(fires) <= n:
        return fires
    # Sample at evenly-spaced quantiles
    idxs = np.linspace(0, len(fires) - 1, n, dtype=int)
    return [fires[i] for i in idxs]


def _rasterize_fire(proj: Fires.Projection, pad: int = 2):
    """Rasterize a fire event colored by first detection day.

    Returns (raster, geo_extent) where raster has NaN=background and
    pixel values = day number (0-based), and geo_extent =
    [lon_min, lon_max, lat_min, lat_max].
    """
    x, y, t = proj.x, proj.y, proj.t
    x0 = int(x.min()) - pad
    y0 = int(y.min()) - pad
    w = int(x.max()) - x0 + 1 + pad
    h = int(y.max()) - y0 + 1 + pad

    raster = np.full((h, w), np.nan, dtype=np.float32)
    for xi, yi, ti in zip(x.tolist(), y.tolist(), t.tolist()):
        px, py = xi - x0, yi - y0
        if np.isnan(raster[py, px]) or ti < raster[py, px]:
            raster[py, px] = float(ti)

    # Normalise to 0-based days
    valid = raster[np.isfinite(raster)]
    if valid.size > 0:
        raster[np.isfinite(raster)] -= valid.min()

    # Geographic extent (VNP14 pixel coords → lon/lat)
    lon_min = x0 * DEG_CELL_SIZE - 180.0
    lat_min = y0 * DEG_CELL_SIZE - 90.0
    lon_max = (x0 + w) * DEG_CELL_SIZE - 180.0
    lat_max = (y0 + h) * DEG_CELL_SIZE - 90.0

    return raster, [lon_min, lon_max, lat_min, lat_max]


def _render_fire_events(
    h5,
    fires: list[dict],
    region_name: str,
) -> "plt.Figure":
    """Render fire events: rows = fires (sorted by spread_ratio),
    cols = [detection-day raster, satellite basemap]."""
    import matplotlib.pyplot as plt
    from firecomp.core.plotting import (
        Cmaps, Style, add_satellite_basemap, imshow_tensor,
    )

    try:
        import cartopy.crs as ccrs
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    nrows = len(fires)
    ncols = 2 if has_cartopy else 1
    cell_w, cell_h = 5.0, 4.5
    fig = plt.figure(
        figsize=(cell_w * ncols, cell_h * nrows + 1.2),
        constrained_layout=True,
    )
    proj_geo = ccrs.PlateCarree() if has_cartopy else None
    gs = fig.add_gridspec(nrows, ncols)

    cmap = Cmaps.fire_spread.copy()
    cmap.set_bad(color="#f0f0f0")

    for row, fdict in enumerate(fires):
        fire_id = fdict["fire_id"]
        proj = Fires.load(h5, "projection_by_fire", str(fire_id))
        if proj is None:
            continue

        raster, geo_ext = _rasterize_fire(proj)

        sol = fdict.get("solidity", 0)
        sr = fdict.get("spread_ratio", 0)
        nc = fdict.get("n_components", 0)
        ni = fdict.get("n_new_ignitions", 0)
        nd = fdict.get("n_days", 0)
        npx = fdict.get("num_pixels", 0)

        title = (
            f"fire {fire_id}   {nd}d   {npx}px\n"
            f"solidity={sol:.2f}  spread={sr:.2f}  "
            f"comp={nc}  ign={ni}"
        )

        # --- Col 0: fire raster (detection day) ---
        ax0 = fig.add_subplot(gs[row, 0])
        imshow_tensor(ax0, raster, cmap=cmap, nodata=np.nan)
        ax0.set_title(title, fontsize=9, pad=Style.title_pad, loc="left")

        # --- Col 1: satellite basemap with fire overlay ---
        if has_cartopy and ncols > 1:
            ax1 = fig.add_subplot(gs[row, 1], projection=proj_geo)
            try:
                # Pad extent slightly for context
                dlon = geo_ext[1] - geo_ext[0]
                dlat = geo_ext[3] - geo_ext[2]
                buf = max(dlon, dlat) * 0.15
                padded = [
                    geo_ext[0] - buf, geo_ext[1] + buf,
                    geo_ext[2] - buf, geo_ext[3] + buf,
                ]
                add_satellite_basemap(ax1, padded)

                # Overlay fire raster
                valid = raster[np.isfinite(raster)]
                vmax = valid.max() if valid.size > 0 else 1
                cmap_ov = cmap.copy()
                cmap_ov.set_bad(alpha=0)
                ax1.imshow(
                    raster, cmap=cmap_ov, vmin=0, vmax=vmax,
                    extent=geo_ext, origin="lower",
                    transform=ccrs.PlateCarree(),
                    interpolation="nearest", alpha=0.85, zorder=2,
                )
            except Exception:
                pass
            ax1.axis("off")

    fig.suptitle(
        f"{region_name} — fire events (by spread_ratio)",
        fontsize=Style.suptitle_fontsize, fontweight="bold",
    )
    return fig


# ── per-fire metrics (shared) ───────────────────────────────────────────

def _compute_metrics_from_mask(mask: np.ndarray, *, connectivity: int = 1) -> dict:
    """Compute solidity + components from a 2D binary mask.

    Used by both projection-based (survey) and patch-based (patches) code.

    connectivity: 1 = 4-connected (scipy default), 2 = 8-connected
                  (diagonal neighbours count as one component).
    """
    num_pixels = int(mask.sum())

    structure = np.ones((3, 3), dtype=int) if connectivity == 2 else None
    _labelled, n_comp = label(mask, structure=structure)

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 3:
        solidity = 1.0
    else:
        try:
            hull = ConvexHull(coords)
            solidity = num_pixels / hull.volume
        except Exception:
            solidity = 1.0

    return {
        "solidity": round(float(solidity), 4),
        "n_components": int(n_comp),
        "num_pixels": int(num_pixels),
        "mean_component_size": round(num_pixels / max(n_comp, 1), 1),
    }


def _compute_metrics(proj: Fires.Projection) -> dict:
    """Compute shape + spread metrics for one fire event.

    Returns dict with spatial metrics (solidity, n_components, num_pixels,
    mean_component_size) and temporal spread metrics (spread_ratio,
    n_new_ignitions, mean_spread_dist, max_spread_dist).
    """
    x, y = proj.x, proj.y

    # Minimal bounding-box binary mask
    x0, y0 = int(x.min()), int(y.min())
    w = int(x.max()) - x0 + 1
    h = int(y.max()) - y0 + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y - y0, x - x0] = 1

    m = _compute_metrics_from_mask(mask)
    m.update(_compute_spread_metrics(proj))
    return m


def _compute_spread_metrics(proj: Fires.Projection, k: int = 3) -> dict:
    """Temporal spread coherence from fire event projection.

    For each unique pixel position, finds when it first appeared (earliest
    detection day) and checks whether a nearby pixel was already burning.

    Wildfires spread contiguously → high spread_ratio (> 0.8).
    Crop / agricultural burns pop up independently → low spread_ratio (< 0.3).

    Args:
        proj: Fire event projection with x, y, t arrays.
        k: Spatial radius (pixels) within which a predecessor counts
           as "nearby" (default 3 ≈ 1.1 km at 375 m VIIRS resolution).

    Returns:
        spread_ratio:      fraction of non-first-timestep unique pixels
                           with a predecessor within *k* cells.
        n_new_ignitions:   timesteps where new pixels appear without a
                           nearby predecessor (excluding first timestep).
        mean_spread_dist:  mean nearest-predecessor distance (pixels).
        max_spread_dist:   max nearest-predecessor distance (pixels).
    """
    x, y, t = proj.x, proj.y, proj.t

    if len(x) == 0:
        return _EMPTY_SPREAD.copy()

    # First detection time per unique (x, y) position
    pixel_ft: dict[tuple[int, int], int] = {}
    for xi, yi, ti in zip(x.tolist(), y.tolist(), t.tolist()):
        key = (xi, yi)
        if key not in pixel_ft or ti < pixel_ft[key]:
            pixel_ft[key] = ti

    n_unique = len(pixel_ft)
    if n_unique <= 1:
        return {"spread_ratio": 1.0, "n_new_ignitions": 0,
                "mean_spread_dist": 0.0, "max_spread_dist": 0.0}

    # Group unique pixels by first detection time
    by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (px, py), ft in pixel_ft.items():
        by_time[ft].append((px, py))

    times = sorted(by_time.keys())
    if len(times) <= 1:
        # All unique pixels appeared on the same day → trivially coherent
        return {"spread_ratio": 1.0, "n_new_ignitions": 0,
                "mean_spread_dist": 0.0, "max_spread_dist": 0.0}

    # Accumulate predecessors, query nearest for each new timestep
    prev = np.array(by_time[times[0]], dtype=np.float64)
    n_close = 0
    n_after = 0
    n_ign = 0
    dists_all: list[np.ndarray] = []

    for ts in times[1:]:
        new = np.array(by_time[ts], dtype=np.float64)
        n_after += len(new)

        tree = cKDTree(prev)
        dd, _ = tree.query(new)
        close = dd <= k
        n_close += int(close.sum())
        dists_all.append(dd)

        if not close.all():
            n_ign += 1

        prev = np.vstack([prev, new])

    dists = np.concatenate(dists_all)
    return {
        "spread_ratio": round(float(n_close / max(n_after, 1)), 4),
        "n_new_ignitions": int(n_ign),
        "mean_spread_dist": round(float(dists.mean()), 2),
        "max_spread_dist": round(float(dists.max()), 2),
    }


_EMPTY_SPREAD = {
    "spread_ratio": 0.0, "n_new_ignitions": 0,
    "mean_spread_dist": 0.0, "max_spread_dist": 0.0,
}


# ── IO helpers ──────────────────────────────────────────────────────────

def _write_json(data: list[dict], path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(data)} entries to {out}", file=sys.stderr)


# ── CLI ─────────────────────────────────────────────────────────────────

DEFAULT_DATASET_DIR = NextDayConfig.dataset_dir
DEFAULT_TOP_N_OUT = "data/runs/solidity.json"
DEFAULT_SURVEY_OUT = "data/runs/solidity_survey.json"
DEFAULT_PATCHES_OUT = "data/runs/solidity_patches.json"
DEFAULT_TEST_SET_OUT = "data/runs/solidity_test.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="solidity",
        description="Fire solidity analysis.",
    )
    sub = p.add_subparsers(dest="command")
    sub.required = True

    # top-n (original)
    sp_top = sub.add_parser("top-n", help="Top-N largest fires per region.")
    sp_top.add_argument("--top-n", type=int, default=100)
    sp_top.add_argument("--min-pixels", type=int, default=100)
    sp_top.add_argument("--output", default=DEFAULT_TOP_N_OUT)
    sp_top.set_defaults(func=cmd_top_n)

    # survey (parallel, all fires)
    sp_sur = sub.add_parser("survey",
                            help="Parallel scan of all fires (solidity + spread).")
    sp_sur.add_argument("--workers", type=int, default=8,
                        help="Number of parallel processes (default: 8).")
    sp_sur.add_argument("--min-pixels", type=int, default=10,
                        help="Skip fires smaller than this (default: 10).")
    sp_sur.add_argument("--output", default=DEFAULT_SURVEY_OUT,
                        help=f"Output JSON path (default: {DEFAULT_SURVEY_OUT}).")
    sp_sur.add_argument("--n-samples", type=int, default=0,
                        help="Fire events to visualize per region (0 = no viz).")
    sp_sur.set_defaults(func=cmd_survey)

    # test-set (fires present in the deterministic test split)
    sp_test = sub.add_parser(
        "test-set",
        help="Solidity over test-split fire events (8-connected components).")
    sp_test.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR,
                         help=f"Dataset directory (default: {DEFAULT_DATASET_DIR}).")
    sp_test.add_argument("--workers", type=int, default=16,
                         help="Number of parallel processes (default: 16).")
    sp_test.add_argument("--output", default=DEFAULT_TEST_SET_OUT,
                         help=f"Output JSON path (default: {DEFAULT_TEST_SET_OUT}).")
    sp_test.set_defaults(func=cmd_test_set)

    # patches (dataset-level, actual 256x256 patches + viz)
    sp_pat = sub.add_parser("patches",
                            help="Solidity on actual dataset patches + viz.")
    sp_pat.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR,
                        help=f"Path to dataset directory (default: {DEFAULT_DATASET_DIR}).")
    sp_pat.add_argument("--workers", type=int, default=16,
                        help="Number of parallel processes (default: 16).")
    sp_pat.add_argument("--output", default=DEFAULT_PATCHES_OUT,
                        help=f"Output JSON path (default: {DEFAULT_PATCHES_OUT}).")
    sp_pat.add_argument("--n-samples", type=int, default=3,
                        help="Random samples per region to visualize (default: 3).")
    sp_pat.add_argument("--padding", type=int, default=16,
                        help="Loss mask border padding (default: 16).")
    sp_pat.add_argument("--seed", type=int, default=0,
                        help="Random seed for sample selection (default: 0).")
    sp_pat.add_argument("--no-viz", action="store_true",
                        help="Skip visualization, only compute metrics.")
    sp_pat.set_defaults(func=cmd_patches)

    return p.parse_args()


if __name__ == "__main__":
    main()
