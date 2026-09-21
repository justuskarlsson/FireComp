"""
next_day/run_figures.py — Load data and generate paper figure PDFs.

Each figure is a self-contained function registered via @figure("key").
Run all figures or select specific ones by name.

Run:
    python -m firecomp.next_day.run_figures                     # all figures
    python -m firecomp.next_day.run_figures inputs               # just inputs
    python -m firecomp.next_day.run_figures fire_types regions   # two figures
    python -m firecomp.next_day.run_figures --list               # show available
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import NextDayDataset
from firecomp.next_day.figures import (
    fig_cluster_lc,
    fig_fire_types,
    fig_input_grid,
    fig_input_group,
    fig_num_fire_histograms,
    fig_prediction_basemap,
    fig_prediction_samples,
    fig_region_extents,
    fig_region_samples,
    fig_sample_density,
    fig_size_and_region_f1,
    assign_size_bucket,
)


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

_FIGURES: dict[str, Callable] = {}


def figure(key: str):
    """Register a figure generator.  Functions receive (ds, args, out_dir)."""
    def decorator(fn: Callable) -> Callable:
        _FIGURES[key] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Channel group config (used by gen_inputs)
# ---------------------------------------------------------------------------

INPUT_GROUPS: dict[str, list[str]] = {
    "Fire State": ["accum_t", "accum_t_min", "accum_t_max", "accum_t_count",
                    "cur_mask"],
    "Terrain": [
        "PCA 1", "PCA 2", "PCA 3", "PCA 4", "PCA 5",
        "AE 1", "AE 2", "AE 3", "AE 4", "AE 5",
        "I1", "I2", "I3", "I4", "I5",
        "Canopy Height",
        "Pos sin(lat)", "Pos cos(lat)", "Pos sin(lon)", "Pos cos(lon)",
    ],
    "Weather": [
        "Current · VPD", "Current · Soil Moist.", "Current · Soil Ratio",
        "Current · Wind Mag.", "Current · Wind sin", "Current · Wind cos",
        "Current · VPD pct", "Current · Soil Moist. pct", "Current · Soil pct",
        "Current · Wind Mag. pct", "Current · Wind sin pct", "Current · Wind cos pct",
        "Forecast · Temp Max", "Forecast · RH Min",
        "Forecast · U Wind Max", "Forecast · V Wind Max",
        "Forecast · Precip Sum",
    ],
}

CMAPS: dict[str, str] = {
    "accum_t": "YlOrRd",
    "accum_t_min": "YlOrRd",
    "accum_t_max": "YlOrRd",
    "accum_t_count": "YlOrRd",
    "cur_mask": "Reds",
    "Current · VPD": "YlOrRd",
    "Current · Soil Moist.": "YlGnBu",
    "Current · Wind sin": "bwr",
    "Current · Wind cos": "bwr",
    "Forecast · U Wind Max": "bwr",
    "Forecast · V Wind Max": "bwr",
    "Forecast · Precip Sum": "YlGnBu",
    "Pos sin(lat)": "bwr",
    "Pos cos(lat)": "bwr",
    "Pos sin(lon)": "bwr",
    "Pos cos(lon)": "bwr",
}

NODATA: dict[str, float] = {
    "accum_t": -1.0, "accum_t_min": -1.0, "accum_t_max": -1.0,
}


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


@figure("inputs")
def gen_inputs(ds: NextDayDataset, args, out_dir: Path):
    """Individual PNGs per channel + a combined grid figure."""
    from firecomp.core.plotting import imshow_tensor, save_figure

    names = ds.channel_names

    # Pick a well-progressed, actively-burning patch of the 2024 Bolivia fire
    # (fire_id 6761 — the largest South American fire of the 2024 season).
    # "Progressed": footprint in the upper half of this fire's range.
    # Among those, take the day with the most active fire so all input
    # channels look meaningful.
    BOLIVIA_FIRE_ID = 6761
    print(f"  Scanning splits for Bolivia fire {BOLIVIA_FIRE_ID} patches...")
    candidates = []  # (split, idx, footprint_px, cur_px)
    for split in ("train", "val", "test"):
        for i, sample in enumerate(getattr(ds, f"{split}_samples")):
            if sample.fire_id != BOLIVIA_FIRE_ID:
                continue
            accum = ds._store.load_field(sample, "accum_t")[0]
            cur = ds._store.load_field(sample, "cur_mask")[0]
            candidates.append((split, i, int((accum >= 0).sum()),
                               int((cur > 0.5).sum())))
    if not candidates:
        raise RuntimeError(f"no dataset samples for fire {BOLIVIA_FIRE_ID}")

    max_foot = max(c[2] for c in candidates)
    progressed = [c for c in candidates if c[2] >= 0.5 * max_foot] or candidates
    best_split, best_idx, foot, cur_px = max(progressed, key=lambda c: c[3])
    print(f"  Selected {best_split}[{best_idx}] "
          f"(footprint={foot}px, active={cur_px}px)")

    x, sample = ds.load_sample(best_split, best_idx)
    all_channels = {name: x[i] for i, name in enumerate(names)}
    print(f"  fire_id={sample.fire_id} dt={sample.dt} "
          f"({len(names)} channels)")

    # Load extra accum_t variants + next_mask target
    extra_fields = ["accum_t_min", "accum_t_max", "accum_t_count"]
    for field_name in extra_fields:
        if field_name not in all_channels:
            try:
                raw = ds._store.load_field(sample, field_name)
                all_channels[field_name] = raw[0]
            except (KeyError, Exception):
                pass

    target = ds._store.load_field(sample, "next_mask")
    all_channels["next_mask"] = target[0].astype(np.float32)

    # ── Individual PNGs (subfolder) ──
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    all_for_grid: dict[str, np.ndarray] = {}
    n_saved = 0
    for group_name, candidates in INPUT_GROUPS.items():
        for lbl in candidates:
            if lbl not in all_channels:
                continue
            nd = NODATA.get(lbl)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            imshow_tensor(ax, all_channels[lbl], title=lbl, cmap="viridis",
                          pct=98.0, nodata=nd)
            slug = lbl.lower().replace(" ", "_").replace("·", "").replace("(", "").replace(")", "")
            slug = slug.replace(".", "").strip("_")
            save_figure(fig, str(inputs_dir / slug), formats=("png",),
                        log=False)
            plt.close(fig)
            all_for_grid[lbl] = all_channels[lbl]
            n_saved += 1

    # Include next_mask in the grid
    all_for_grid["next_mask"] = all_channels["next_mask"]

    print(f"  saved {n_saved} channel PNGs to {inputs_dir}/")

    # ── Combined grid figure ── (7 cols × 3 rows = 21 channels, compact)
    fig = fig_input_grid(
        all_for_grid,
        ncols=7,
        nodata=NODATA,
        out_path=str(out_dir / "inputs_grid"),
    )
    plt.close(fig)
    print(f"  saved: inputs_grid.pdf + .png")


@figure("regions")
def gen_regions(ds: NextDayDataset, args, out_dir: Path):
    """Single world map showing the spatial extent of each study region."""
    from firecomp.core.regions import Regions, RegionRaster

    regions = Regions()
    rr = RegionRaster.load()

    region_names = {rid: regions.id_to_name(rid) for rid in regions.ids()}
    print(f"  {len(region_names)} regions")

    fig = fig_region_extents(
        region_names,
        raster=rr.raster,
        title="Study regions",
        out_path=str(out_dir / "regions"),
    )
    plt.close(fig)
    print(f"  saved: regions.pdf")


@figure("fire_types")
def gen_fire_types(ds: NextDayDataset, args, out_dir: Path):
    """3×3 grid: static / crop burn / wildfire detection rasters."""
    import h5py
    from firecomp.config import config
    from firecomp.core.fire_stats import (
        FireClassification,
        MIN_FIRE_COUNT,
        TAME_IGNITION_THRESH,
        TAME_T_RATIO_THRESH,
        TAME_XY_THRESH,
    )
    from firecomp.dsrc.vnp14 import Fires

    N_EXAMPLES = 3
    MIN_DEG_APART = 30.0  # geographic spread between selected fires

    # --- load VNP14 + classify ---
    with h5py.File(config.vnp14_path, "r") as h5:
        stats: Fires.Stats = Fires.load(h5, "stats")
        proj: Fires.Projection = Fires.load(h5, "projection")

    is_tame = FireClassification.identify_tame(stats)
    fire_areas = FireClassification.compute_spatial_areas(proj, stats)
    areas_km2 = FireClassification.areas_to_km2(fire_areas, stats)
    is_wild = FireClassification.identify_wild(stats, is_tame, areas_km2)

    has_enough = stats.num_fire >= MIN_FIRE_COUNT
    is_oil = has_enough & (stats.t_ratio > TAME_T_RATIO_THRESH)
    is_crop = (has_enough
               & (stats.avg_xy_neighbors <= TAME_XY_THRESH)
               & (stats.ignition_ratio * 100 >= TAME_IGNITION_THRESH)
               & ~is_oil)

    # --- pick best examples per type (most detections, geographically spread) ---
    type_masks = [("Static", is_oil), ("Crop Burn", is_crop), ("Wildfire", is_wild)]
    type_labels: list[str] = []
    panels: list[list[np.ndarray]] = []
    fire_labels: list[list[str]] = []
    fire_geo: list[list[dict]] = []
    fire_stats_list: list[list[dict]] = []

    for label, mask in type_masks:
        candidates = np.where(mask)[0]
        candidates = candidates[np.argsort(stats.num_fire[candidates])[::-1]]
        print(f"  {label}: {len(candidates)} fires")

        # Pick geographically spread fires
        selected = _pick_spread(candidates, stats, N_EXAMPLES, MIN_DEG_APART)

        row_panels: list[np.ndarray] = []
        row_labels: list[str] = []
        row_geo: list[dict] = []
        row_stats: list[dict] = []
        for i in selected:
            fid = int(stats.id[i])
            raster = _rasterise_fire(proj, fid)
            h, w = raster.shape

            # Duration in days
            d0 = str(stats.start_date[i])[:10]
            d1 = str(stats.end_date[i])[:10]
            from datetime import date as _date
            duration = (_date.fromisoformat(d1) - _date.fromisoformat(d0)).days

            # Geo extent (lat/lon degrees)
            geo = {
                "lon_min": float(stats.min_x[i]),
                "lon_max": float(stats.max_x[i]),
                "lat_min": float(stats.min_y[i]),
                "lat_max": float(stats.max_y[i]),
            }

            # Classification metrics
            fs = {
                "t_ratio": float(stats.t_ratio[i]),
                "xy_neighbors": float(stats.avg_xy_neighbors[i]),
                "ign_ratio": float(stats.ignition_ratio[i] * 100),
            }

            print(f"    fire {fid}: {h}×{w} px, {duration} days, "
                  f"({geo['lat_min']:.1f}°, {geo['lon_min']:.1f}°) "
                  f"t={fs['t_ratio']:.1f} xy={fs['xy_neighbors']:.1f} "
                  f"ign={fs['ign_ratio']:.1f}%")

            row_panels.append(raster)
            row_labels.append(f"{duration} days")
            row_geo.append(geo)
            row_stats.append(fs)

        while len(row_panels) < N_EXAMPLES:
            row_panels.append(np.full((64, 64), np.nan))
            row_labels.append("—")
            row_geo.append({})
            row_stats.append({})

        type_labels.append(label)
        panels.append(row_panels)
        fire_labels.append(row_labels)
        fire_geo.append(row_geo)
        fire_stats_list.append(row_stats)

    fig = fig_fire_types(
        panels, type_labels, fire_labels,
        fire_geo=fire_geo,
        fire_stats=fire_stats_list,
        row_max_days=[None, 60, None],
        title="Fire types — detection spread",
        out_path=str(out_dir / "fire_types"),
    )
    plt.close(fig)
    print(f"  saved: fire_types.pdf")


@figure("cluster_lc")
def gen_cluster_lc(ds: NextDayDataset, args, out_dir: Path):
    """World map: each 0.1° cell coloured by its dominant cluster's LC class."""
    from osgeo import gdal

    from firecomp.core.clusters import ClusterPriors
    from firecomp.core.regions import RASTER_COLS, RASTER_ROWS

    priors = ClusterPriors.load(year=args.cluster_year)
    n_clusters = priors.n_clusters
    dom_lc = priors.cluster_stats["dominant_lc_class"]  # (n_clusters,) str
    print(f"  Loaded {n_clusters} clusters for year {args.cluster_year}")

    # Build name → index mapping for the unique LC classes present
    unique_names = sorted(set(str(n) for n in dom_lc if str(n)))
    name_to_idx = {n: i for i, n in enumerate(unique_names)}
    print(f"  LC classes present: {', '.join(unique_names)}")

    # Downsample cluster raster to 0.1° via mode
    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=(-180, -90, 180, 90),
        width=RASTER_COLS,
        height=RASTER_ROWS,
        resampleAlg="mode",
        srcNodata=255,
        dstNodata=255,
    )
    out_ds = gdal.Warp("", priors.cluster_raster_path, options=warp_opts)
    dominant = out_ds.ReadAsArray().astype(np.uint8)
    out_ds = None

    # Map cluster ID → LC class index
    valid = (dominant != 255) & (dominant < n_clusters)
    lc_map = np.full(dominant.shape, -1, dtype=np.int8)
    for cid in range(n_clusters):
        name = str(dom_lc[cid])
        if name in name_to_idx:
            mask = valid & (dominant == cid)
            lc_map[mask] = name_to_idx[name]

    n_valid = int(valid.sum())
    print(f"  {n_valid:,} / {dominant.size:,} cells with valid cluster data")

    fig = fig_cluster_lc(
        lc_map, unique_names,
        title=f"Dominant land-cover class ({args.cluster_year})",
        out_path=str(out_dir / "cluster_lc"),
    )
    plt.close(fig)
    print(f"  saved: cluster_lc.pdf")


@figure("sample_density")
def gen_sample_density(ds: NextDayDataset, args, out_dir: Path):
    """2×2 heatmap: sample density per 3° cell, years grouped in pairs.

    Loads samples.json + VNP14 stats directly (no full dataset needed).
    Each sample's position is the centre of its fire's bounding box.
    """
    import json
    import h5py
    from firecomp.config import config
    from firecomp.dsrc.vnp14 import Fires

    resolution = 3.0
    nlat = int(180 / resolution)
    nlon = int(360 / resolution)

    # --- load samples.json ---
    samples_path = Path(args.samples_path)
    with open(samples_path) as f:
        samples_data = json.load(f)
    samples = samples_data["samples"]
    print(f"  {len(samples)} samples from {samples_path}")

    # --- load VNP14 stats for fire positions ---
    with h5py.File(config.vnp14_path, "r") as h5:
        stats: Fires.Stats = Fires.load(h5, "stats")

    # Build fire_id → index lookup
    fid_to_idx: dict[int, int] = {}
    for i, fid in enumerate(stats.id):
        fid_to_idx[int(fid)] = i

    # Fire centre positions (degrees)
    lons = (stats.min_x + stats.max_x) / 2
    lats = (stats.min_y + stats.max_y) / 2

    # --- bin into yearly grids ---
    yearly_counts: dict[int, np.ndarray] = {}
    n_missing = 0
    for s in samples:
        fid = s["fire_id"]
        idx = fid_to_idx.get(fid)
        if idx is None:
            n_missing += 1
            continue

        year = int(s["day_T"][:4])
        if year not in yearly_counts:
            yearly_counts[year] = np.zeros((nlat, nlon), dtype=np.int32)

        col = int((float(lons[idx]) + 180) / resolution)
        row = int((float(lats[idx]) + 90) / resolution)
        col = min(col, nlon - 1)
        row = min(row, nlat - 1)
        yearly_counts[year][row, col] += 1

    if n_missing:
        print(f"  warning: {n_missing} samples with unknown fire_id")

    for year in sorted(yearly_counts):
        n_cells = (yearly_counts[year] > 0).sum()
        n_samples = yearly_counts[year].sum()
        print(f"  {year}: {n_samples:,} samples in {n_cells:,} cells")

    # --- group years into consecutive pairs ---
    all_years = sorted(yearly_counts)
    group_counts: dict[str, np.ndarray] = {}
    i = 0
    while i < len(all_years):
        if i + 1 < len(all_years):
            y1, y2 = all_years[i], all_years[i + 1]
            label = f"{y1}–{y2}"
            combined = yearly_counts[y1] + yearly_counts[y2]
            i += 2
        else:
            label = str(all_years[i])
            combined = yearly_counts[all_years[i]]
            i += 1
        group_counts[label] = combined
        n_cells = (combined > 0).sum()
        print(f"  group {label}: {int(combined.sum()):,} samples in {n_cells:,} cells")

    fig = fig_sample_density(
        group_counts,
        resolution=resolution,
        title=f"Sample density ({int(resolution)}° grid)",
        out_path=str(out_dir / "sample_density"),
    )
    plt.close(fig)
    print(f"  saved: sample_density.pdf")


@figure("num_fire_hist")
def gen_num_fire_hist(ds: NextDayDataset, args, out_dir: Path):
    """Per-region log-binned histograms of per-sample num_fire.

    Uses train+val+test from the built dataset with fire_type=all
    (same population as tab:fire-type-region), not the pick-list
    samples.json or the vegetation-only training default.
    """
    from firecomp.core.regions import Regions
    from firecomp.next_day.config import NextDayConfig

    cfg = NextDayConfig(
        dataset_dir=args.dataset_dir or NextDayConfig.dataset_dir,
        fire_type="all",
    )
    ds = NextDayDataset(cfg)

    samples = ds.train_samples + ds.val_samples + ds.test_samples
    print(f"  {len(samples):,} samples from dataset splits (fire_type=all)")

    regions = Regions()
    by_region: dict[str, list[int]] = {}
    n_unassigned = 0
    for s in samples:
        if s.region_id == 0:
            n_unassigned += 1
            continue
        by_region.setdefault(regions.id_to_name(s.region_id), []).append(
            s.num_fire)
    if n_unassigned:
        print(f"  skipped {n_unassigned:,} samples with region_id=0")

    # Stable display order: sort by region id
    ordered: dict[str, np.ndarray] = {}
    for rid in regions.ids():
        name = regions.id_to_name(rid)
        if name in by_region:
            arr = np.array(by_region[name])
            ordered[name] = arr
            print(f"  {name}: n={arr.size:,}  median={int(np.median(arr))}  "
                  f"max={int(arr.max())}")

    fig = fig_num_fire_histograms(
        ordered,
        title="num_fire per sample, by region (log bins)",
        out_path=str(out_dir / "num_fire_hist"),
    )
    plt.close(fig)
    print(f"  saved: num_fire_hist.pdf + .png")


@figure("size_and_region_f1")
def gen_size_and_region_f1(ds: NextDayDataset, args, out_dir: Path):
    """Per-pixel F1 scatter by region × fire-size bucket.

    Requires --checkpoint (uses the first one).
    Runs inference on ALL test samples, accumulates TP/FP/FN pixel
    counts per (region, fire-size bucket) group, then computes one
    pooled pixel-level F1 per group.

    Generates two figures:
      - size_and_region_f1.pdf — next_mask (standard loss mask)
      - size_and_region_f1_nf.pdf — nf_equiv (burned pixels excluded)
    """
    import json
    import torch
    from firecomp.core.checkpoint import BestCheckpoint
    from firecomp.core.metrics import Metrics
    from firecomp.core.regions import Regions
    from firecomp.core.torch_utils import get_device
    from firecomp.models.segmentation_models import model_factory

    if not args.checkpoint:
        print("  skipped: no --checkpoint provided")
        return

    ckpt_path = args.checkpoint[0]
    device = get_device()
    ckpt = BestCheckpoint(Path(ckpt_path).parent).load(map_location=device)
    cfg = ckpt["cfg"]
    threshold = ckpt["threshold"]
    tag = cfg.tag or Path(ckpt_path).parent.name
    print(f"  checkpoint: {ckpt_path}  (tag={tag})")

    model = model_factory[cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=cfg.encoder_name,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build fire_id → num_fire from samples.json
    fid_to_numfire: dict[int, int] = {}
    samples_path = Path(args.samples_path)
    if samples_path.exists():
        with open(samples_path) as f:
            for s in json.load(f)["samples"]:
                fid_to_numfire[s["fire_id"]] = s["num_fire"]
        print(f"  loaded {len(fid_to_numfire)} fire sizes from {samples_path}")
    else:
        print(f"  warning: {samples_path} not found, using target pixel count")

    regions = Regions()
    cur_mask_ch: int | None = None
    accum_ch: int | None = None

    # Accumulators per (region, bucket) group — both targets
    nm_tp: dict[tuple[str, str], int] = Counter()
    nm_fp: dict[tuple[str, str], int] = Counter()
    nm_fn: dict[tuple[str, str], int] = Counter()
    nf_tp: dict[tuple[str, str], int] = Counter()
    nf_fp: dict[tuple[str, str], int] = Counter()
    nf_fn: dict[tuple[str, str], int] = Counter()
    group_n: dict[tuple[str, str], int] = Counter()

    # Run inference on all test samples, accumulate per-sample TP/FP/FN
    print(f"  running inference on {len(ds.test_samples)} test samples...")
    with torch.no_grad():
        for batch in ds.test(batch_size=cfg.batch_size):
            pred = torch.sigmoid(model(batch.x))
            pred_cpu = pred.cpu()
            y_cpu = batch.y.cpu()
            mask_cpu = batch.loss_mask.cpu()

            # next_mask metrics (standard)
            m_nm = Metrics(pred_cpu, y_cpu, mask_cpu, threshold=threshold)

            # nf_equiv metrics — exclude burned pixels (accum_t_min >= -0.1)
            if cur_mask_ch is None:
                cur_mask_ch = batch.channel_names.index("cur_mask")
                accum_ch = batch.channel_names.index("accum_t_min")
            burned = (batch.x[:, accum_ch:accum_ch + 1].cpu() >= -0.1).float()
            nf_mask = mask_cpu * (1.0 - burned)
            m_nf = Metrics(pred_cpu, y_cpu, nf_mask, threshold=threshold)

            for i in range(pred.shape[0]):
                sample = batch.samples[i]
                if sample.region_id == 0:
                    continue

                nf = fid_to_numfire.get(sample.fire_id)
                if nf is None:
                    nf = int((y_cpu[i, 0] > 0.5).sum().item())
                bucket = assign_size_bucket(nf)
                if bucket is None:
                    continue

                region_name = regions.id_to_name(sample.region_id)
                key = (region_name, bucket)
                nm_tp[key] += int(m_nm._all_tp[i].item())
                nm_fp[key] += int(m_nm._all_fp[i].item())
                nm_fn[key] += int(m_nm._all_fn[i].item())
                nf_tp[key] += int(m_nf._all_tp[i].item())
                nf_fp[key] += int(m_nf._all_fp[i].item())
                nf_fn[key] += int(m_nf._all_fn[i].item())
                group_n[key] += 1

    def _pooled_f1(tp_map, fp_map, fn_map):
        data = []
        for key in sorted(group_n):
            region, bucket = key
            tp, fp, fn = tp_map[key], fp_map[key], fn_map[key]
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            data.append({
                "region": region,
                "f1": f1,
                "size_bucket": bucket,
                "n_samples": group_n[key],
            })
        return data

    # ── next_mask figure ──
    nm_data = _pooled_f1(nm_tp, nm_fp, nm_fn)
    print(f"\n  next_mask: {len(nm_data)} groups from "
          f"{sum(group_n.values())} test samples")
    for d in nm_data:
        print(f"    {d['region']:>16} {d['size_bucket']:>6}  "
              f"n={d['n_samples']:>5}  F1={d['f1']:.4f}")
    fig = fig_size_and_region_f1(
        nm_data,
        title=f"Per-pixel F1 by region and fire size ({tag})",
        out_path=str(out_dir / "size_and_region_f1"),
    )
    plt.close(fig)
    print(f"  saved: size_and_region_f1.pdf + .png")

    # ── nf_equiv figure ──
    nf_data = _pooled_f1(nf_tp, nf_fp, nf_fn)
    print(f"\n  nf_equiv: {len(nf_data)} groups")
    for d in nf_data:
        print(f"    {d['region']:>16} {d['size_bucket']:>6}  "
              f"n={d['n_samples']:>5}  F1={d['f1']:.4f}")
    fig = fig_size_and_region_f1(
        nf_data,
        title=f"Per-pixel nf-equiv F1 by region and fire size ({tag})",
        out_path=str(out_dir / "size_and_region_f1_nf"),
    )
    plt.close(fig)
    print(f"  saved: size_and_region_f1_nf.pdf + .png")


@figure("predictions")
def gen_predictions(ds: NextDayDataset, args, out_dir: Path):
    """Prediction probability maps from trained checkpoint(s).

    Requires --checkpoint pointing to one or more best.pt files.
    Produces per checkpoint:
      - predictions_{tag}.pdf — curated case study (LA / Bolivia / Africa)
      - predictions_{tag}_{region}.pdf — per-region supplemental (3 per region)

    All figures use a satellite basemap behind accum_t and score on the
    new-fires-equivalent basis (excluding previously-burned pixels).
    """
    if not args.checkpoint:
        print("  skipped: no --checkpoint provided")
        return

    for ckpt_path in args.checkpoint:
        _gen_predictions_single(ds, ckpt_path, out_dir)


# ── Case study constants ──────────────────────────────────────────────────

_CASE_STUDY_DIR = "data/next_day_v3_case_study"
_LA_FIRE_ID = 442281          # Palisades
_BOLIVIA_FIRE_ID = 6761       # largest 2024 South American fire
_AFRICA_REGION_ID = 3
_N_CASE_CANDIDATES = 60


def _gen_predictions_single(
    ds: NextDayDataset, ckpt_path: str, out_dir: Path,
):
    """Run inference for one checkpoint and save main + supplemental figures.

    Main figure: curated 3-row case study (LA Palisades, Bolivia, Africa).
    Supplemental: per-region figures at P10/P50/P90 of fire size.
    """
    import torch
    from firecomp.core.checkpoint import BestCheckpoint
    from firecomp.core.metrics import Metrics
    from firecomp.core.regions import Regions
    from firecomp.core.torch_utils import get_device
    from firecomp.models.segmentation_models import model_factory

    device = get_device()
    ckpt = BestCheckpoint(Path(ckpt_path).parent).load(
        map_location=device)
    cfg = ckpt["cfg"]
    threshold = ckpt["threshold"]
    tag = cfg.tag or Path(ckpt_path).parent.name
    print(f"\n  checkpoint: {ckpt_path}  (tag={tag})")

    model = model_factory[cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=cfg.encoder_name,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Main figure: curated case study (3 rows) ──
    main_rows = _build_case_study_rows(model, ds, cfg, threshold, device)
    if main_rows:
        fig = fig_prediction_basemap(
            main_rows,
            out_path=str(out_dir / f"predictions_{tag}"),
        )
        plt.close(fig)
        print(f"  saved: predictions_{tag}.pdf + .png")

    # ── Supplemental: 3 per-region figures at P10/P50/P90 of fire size ──
    test_samples = ds.test_samples
    print(f"  Scanning {len(test_samples)} test samples for fire pixel count...")

    fire_counts: list[tuple[int, int]] = []
    for i, sample in enumerate(test_samples):
        target = ds._store.load_field(sample, cfg.target_type)
        n_fire = int((target > 0.5).sum())
        fire_counts.append((i, n_fire))

    regions = Regions()
    by_region: dict[int, list[tuple[int, int]]] = {}
    for idx, cnt in fire_counts:
        rid = test_samples[idx].region_id
        if rid == 0 or cnt == 0:
            continue
        by_region.setdefault(rid, []).append((idx, cnt))

    supp_indices_per_region: list[tuple[str, list[int]]] = []
    for rid in sorted(by_region):
        entries = sorted(by_region[rid], key=lambda t: t[1])
        n = len(entries)
        if n < 3:
            picks = [e[0] for e in entries]
        else:
            p10 = max(0, int(n * 0.10))
            p50 = n // 2
            p90 = min(n - 1, int(n * 0.90))
            picks = [entries[p10][0], entries[p50][0], entries[p90][0]]
        rname = regions.id_to_name(rid)
        supp_indices_per_region.append((rname, picks))
        print(f"  region {rname}: {n} test samples, "
              f"picked {len(picks)} at P10/P50/P90")

    all_supp_indices = set()
    for _, idxs in supp_indices_per_region:
        all_supp_indices.update(idxs)

    padding = cfg.padding
    print(f"  Running inference on {len(all_supp_indices)} supplemental samples "
          f"(padding={padding})...")

    results: dict[int, dict] = {}
    with torch.no_grad():
        for i in all_supp_indices:
            results[i] = _infer_sample(
                model, ds, "test", i, cfg, threshold, device)

    for rname, idxs in supp_indices_per_region:
        region_picked = [results[i] for i in idxs]
        for s in region_picked:
            s["name"] = f"{rname}\nfire {s['fire_id']}"
        region_picked.sort(key=lambda s: s["f1"], reverse=True)
        slug = rname.lower().replace(" ", "_")
        fig = fig_prediction_basemap(
            region_picked,
            title=rname,
            out_path=str(out_dir / f"predictions_{tag}_{slug}"),
        )
        plt.close(fig)
        print(f"  saved: predictions_{tag}_{slug}.pdf")


# ── Shared inference helper ───────────────────────────────────────────────

def _infer_sample(model, ds, split, idx, cfg, threshold, device) -> dict:
    """Run inference on one sample and return a dict for fig_prediction_basemap.

    Loads the sample, runs the model, computes nf_equiv F1 (excluding
    previously-burned pixels from evaluation), and returns a dict with
    all fields needed by ``fig_prediction_basemap``.
    """
    import torch
    from firecomp.core.metrics import Metrics
    from firecomp.next_day.dataset import _union_accum_cur_mask, compute_loss_mask

    samples = getattr(ds, f"{split}_samples")
    sample = samples[idx]
    x_np, _ = ds.load_sample(split, idx)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    pred = torch.sigmoid(model(x)).cpu()[0, 0].numpy()

    raw = ds._store.load(sample)
    _union_accum_cur_mask(raw)
    loss_mask = compute_loss_mask(raw, padding=cfg.padding)
    accum = raw["accum_t"][0]
    target = raw["next_mask"][0].astype(np.float32)

    nf_mask = (loss_mask > 0.5) & (accum < 0)
    f1 = _nf_f1(pred, target, nf_mask, threshold)

    return {
        "accum_t":   accum,
        "target":    target,
        "pred":      pred,
        "loss_mask": loss_mask.astype(np.float32),
        "padding":   cfg.padding,
        "threshold": threshold,
        "fire_id":   sample.fire_id,
        "dt":        sample.dt,
        "lon":       sample.lon,
        "lat":       sample.lat,
        "f1":        f1,
    }


def _nf_f1(pred, target, nf_mask, threshold) -> float:
    """New-fires-equivalent F1: score only the unburned (*nf_mask*) pixels."""
    import torch
    from firecomp.core.metrics import Metrics
    m = Metrics(
        torch.from_numpy(pred)[None, None],
        torch.from_numpy(target)[None, None],
        torch.from_numpy(nf_mask.astype(np.float32))[None, None],
        threshold=threshold,
    )
    return m.f1


# ── Case study main figure ────────────────────────────────────────────────

def _build_case_study_rows(model, ds, cfg, threshold, device) -> list[dict]:
    """Build the curated case study rows for the main prediction figure.

    Row 1: LA 2025 Palisades — from the case-study directory (outside split).
    Row 2: South America 2024 Bolivia — same fire as the inputs grid.
    Row 3: Africa fragmented — erratic fire with many components.

    Returns an empty list if no rows could be built (e.g. data not available).
    """
    rows: list[dict] = []

    # Row 1: LA Palisades (case-study dir)
    la_row = _build_la_row(model, cfg, threshold, device)
    if la_row is not None:
        rows.append(la_row)

    # Row 2: Bolivia (search all splits for the specific fire)
    bol_row = _build_fire_row(
        model, ds, cfg, threshold, device,
        fire_id=_BOLIVIA_FIRE_ID, name="South America\n2024")
    if bol_row is not None:
        rows.append(bol_row)

    # Row 3: Africa fragmented
    afr_row = _build_region_row(
        model, ds, cfg, threshold, device,
        region_id=_AFRICA_REGION_ID, mode="fragmented",
        name="Africa\n(fragmented)")
    if afr_row is not None:
        rows.append(afr_row)

    return rows


def _build_la_row(model, cfg, threshold, device):
    """LA Palisades row from the case-study dataset (outside the split).

    Returns None if the case-study directory does not exist.
    """
    from dataclasses import replace as dc_replace
    from firecomp.next_day.dataset import _compute_is_last_day, _union_accum_cur_mask

    case_dir = Path(_CASE_STUDY_DIR)
    if not case_dir.exists():
        print(f"  LA row: skipped ({case_dir} not found)")
        return None

    import torch
    from firecomp.core.metrics import Metrics

    eval_cfg = dc_replace(cfg, dataset_dir=str(case_dir), batch_size=8,
                          num_workers=0, device=str(device))
    ds_la = NextDayDataset(eval_cfg)
    samples = ds_la._store.load_samples()
    ds_la._test_samples = samples
    ds_la._test_is_last = _compute_is_last_day(samples)
    batch = ds_la._load_all(samples, ds_la._test_is_last)
    batch.to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(batch.x)).cpu()

    best = None  # (nf_f1, i, raw, loss_mask)
    for i, s in enumerate(samples):
        if s.fire_id != _LA_FIRE_ID:
            continue
        raw = ds_la._store.load(s)
        _union_accum_cur_mask(raw)
        from firecomp.next_day.dataset import compute_loss_mask
        lm = compute_loss_mask(raw, padding=cfg.padding)
        nf_mask = (lm > 0.5) & (raw["accum_t"][0] < 0)
        f1 = _nf_f1(pred[i, 0].numpy(),
                     raw["next_mask"][0].astype(np.float32),
                     nf_mask, threshold)
        print(f"  LA Palisades {s.dt}: nf-F1={f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, i, raw, lm, s)

    if best is None:
        print(f"  LA row: skipped (fire {_LA_FIRE_ID} not found)")
        return None

    f1, i, raw, lm, s = best
    print(f"  -> LA pick: {s.dt}  nf-F1={f1:.3f}")
    return {
        "accum_t":   raw["accum_t"][0],
        "target":    raw["next_mask"][0].astype(np.float32),
        "pred":      pred[i, 0].numpy(),
        "loss_mask": lm.astype(np.float32),
        "padding":   cfg.padding,
        "threshold": threshold,
        "fire_id":   s.fire_id,
        "dt":        s.dt,
        "lon":       s.lon,
        "lat":       s.lat,
        "f1":        f1,
        "name":      "LA 2025\nPalisades",
    }


def _build_fire_row(model, ds, cfg, threshold, device, *,
                    fire_id, name) -> dict | None:
    """Find a specific fire across all splits and pick the best nf-F1 day.

    Mirrors ``gen_inputs`` selection for Bolivia: best actively-burning day
    among well-progressed patches (footprint >= 0.5 * max).
    """
    import torch
    from firecomp.next_day.dataset import _union_accum_cur_mask, compute_loss_mask

    cands = []  # (split, idx, footprint_px, cur_px)
    for split in ("train", "val", "test"):
        for i, s in enumerate(getattr(ds, f"{split}_samples")):
            if s.fire_id != fire_id:
                continue
            accum = ds._store.load_field(s, "accum_t")[0]
            cur = ds._store.load_field(s, "cur_mask")[0]
            cands.append((split, i, int((accum >= 0).sum()),
                          int((cur > 0.5).sum())))
    if not cands:
        print(f"  {name.splitlines()[0]} row: skipped (fire {fire_id} not found)")
        return None

    max_foot = max(c[2] for c in cands)
    prog = [c for c in cands if c[2] >= 0.5 * max_foot] or cands
    split, idx, foot, cur_px = max(prog, key=lambda c: c[3])

    x_np, s = ds.load_sample(split, idx)
    with torch.no_grad():
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(x)).cpu()[0, 0].numpy()
    raw = ds._store.load(s)
    _union_accum_cur_mask(raw)
    loss_mask = compute_loss_mask(raw, padding=cfg.padding)
    nf_mask = (loss_mask > 0.5) & (raw["accum_t"][0] < 0)
    f1 = _nf_f1(pred, raw["next_mask"][0].astype(np.float32),
                 nf_mask, threshold)
    print(f"  -> {name.splitlines()[0]} pick: fire {s.fire_id} {s.dt} "
          f"{split}[{idx}]  nf-F1={f1:.3f}  footprint={foot}px active={cur_px}px")
    return {
        "accum_t":   raw["accum_t"][0],
        "target":    raw["next_mask"][0].astype(np.float32),
        "pred":      pred,
        "loss_mask": loss_mask.astype(np.float32),
        "padding":   cfg.padding,
        "threshold": threshold,
        "fire_id":   s.fire_id,
        "dt":        s.dt,
        "lon":       s.lon,
        "lat":       s.lat,
        "f1":        f1,
        "name":      name,
    }


def _build_region_row(model, ds, cfg, threshold, device, *,
                      region_id, mode, name) -> dict | None:
    """Score candidate test samples in a region and pick one to display.

    mode="fragmented": pick the most fragmented fire (max 8-connected
                       component count among patches with >= 40 new-fire pixels).
    mode="largest":    pick the biggest, well-predicted fire (max nf-F1).
    """
    from scipy.ndimage import label as ndlabel
    from firecomp.next_day.dataset import _union_accum_cur_mask, compute_loss_mask

    import torch

    cands = [i for i, s in enumerate(ds.test_samples)
             if s.region_id == region_id]
    if not cands:
        print(f"  {name.splitlines()[0]} row: skipped (no test samples "
              f"for region {region_id})")
        return None

    # Pre-rank: keep the N patches with the most next-day fire pixels.
    by_size = sorted(
        cands,
        key=lambda i: int((ds._store.load_field(
            ds.test_samples[i], "next_mask") > 0.5).sum()),
        reverse=True,
    )
    cands = by_size[:_N_CASE_CANDIDATES]

    scored = []
    for i in cands:
        s = ds.test_samples[i]
        x_np, _ = ds.load_sample("test", i)
        with torch.no_grad():
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            pred = torch.sigmoid(model(x)).cpu()[0, 0].numpy()
        raw = ds._store.load(s)
        _union_accum_cur_mask(raw)
        target = raw["next_mask"][0].astype(np.float32)
        accum = raw["accum_t"][0]
        loss_mask = compute_loss_mask(raw, padding=cfg.padding)
        nf_mask = (loss_mask > 0.5) & (accum < 0)
        nf_f1 = _nf_f1(pred, target, nf_mask, threshold)
        n_new = int(((target > 0.5) & nf_mask).sum())
        _, n_comp = ndlabel(accum >= 0,
                            structure=np.ones((3, 3), dtype=int))
        scored.append({"i": i, "s": s, "raw": raw, "pred": pred,
                       "loss_mask": loss_mask, "f1": nf_f1,
                       "n_new": n_new, "n_comp": int(n_comp)})

    if mode == "fragmented":
        pool = [c for c in scored if c["n_new"] >= 40] or scored
        pick = max(pool, key=lambda c: c["n_comp"])
    else:  # largest
        pool = [c for c in scored if c["n_new"] >= 60] or scored
        pick = max(pool, key=lambda c: c["f1"])

    s = pick["s"]
    print(f"  -> {name.splitlines()[0]} pick: fire {s.fire_id} {s.dt}  "
          f"nf-F1={pick['f1']:.3f}  new-fire={pick['n_new']}px  "
          f"components={pick['n_comp']}")
    return {
        "accum_t":   pick["raw"]["accum_t"][0],
        "target":    pick["raw"]["next_mask"][0].astype(np.float32),
        "pred":      pick["pred"],
        "loss_mask": pick["loss_mask"].astype(np.float32),
        "padding":   cfg.padding,
        "threshold": threshold,
        "fire_id":   s.fire_id,
        "dt":        s.dt,
        "lon":       s.lon,
        "lat":       s.lat,
        "f1":        pick["f1"],
        "name":      name,
    }


def _build_loss_mask(store, sample, padding: int):
    """Build a (1, 1, H, W) torch loss mask for a single sample.

    Uses compute_loss_mask from dataset.py — same masking as training.
    """
    import torch
    from firecomp.next_day.dataset import compute_loss_mask
    raw = store.load(sample)
    mask_np = compute_loss_mask(raw, padding=padding)  # (H, W)
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def _pick_spread(
    candidates: np.ndarray,
    stats,
    n: int,
    min_deg: float,
) -> list[int]:
    """Pick up to *n* fires from *candidates* (sorted by priority).

    Each selected fire must be at least *min_deg* degrees (Euclidean
    in lon/lat) from all previously selected fires.  Falls back to
    just the top-N if spread can't be achieved.
    """
    lons = (stats.min_x + stats.max_x) / 2
    lats = (stats.min_y + stats.max_y) / 2

    selected: list[int] = []
    for i in candidates:
        if len(selected) >= n:
            break
        lon_i, lat_i = float(lons[i]), float(lats[i])
        too_close = False
        for j in selected:
            dlon = lon_i - float(lons[j])
            dlat = lat_i - float(lats[j])
            if (dlon**2 + dlat**2) ** 0.5 < min_deg:
                too_close = True
                break
        if not too_close:
            selected.append(int(i))

    # Fall back: if we couldn't find enough spread fires, fill from top
    if len(selected) < n:
        for i in candidates:
            if len(selected) >= n:
                break
            if int(i) not in selected:
                selected.append(int(i))

    return selected


def _rasterise_fire(proj, fire_id: int) -> np.ndarray:
    """Rasterise a fire's detections into (H, W) detection-day image.

    Time is normalised: t=0 is the first detection, values increase from
    there.  Overlapping pixels keep the latest (max) detection day — shows
    how far the fire spread over its lifetime.
    """
    mask = proj.component == fire_id
    if not mask.any():
        return np.full((64, 64), np.nan, dtype=np.float32)

    x, y, t = proj.x[mask], proj.y[mask], proj.t[mask]
    t = t - t.min()   # normalise: day 0 = first detection
    x0, y0 = int(x.min()), int(y.min())
    h = max(int(y.max()) - y0 + 1, 1)
    w = max(int(x.max()) - x0 + 1, 1)

    raster = np.full((h, w), np.nan, dtype=np.float32)
    for xi, yi, ti in zip(x, y, t):
        ry, rx = int(yi - y0), int(xi - x0)
        if np.isnan(raster[ry, rx]) or ti > raster[ry, rx]:
            raster[ry, rx] = float(ti)
    return raster


# ---------------------------------------------------------------------------
# main + CLI
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    if args.list:
        print("Available figures:")
        for key in _FIGURES:
            print(f"  {key}")
        return

    # Resolve defaults from config when not explicitly provided
    _default_dir = NextDayConfig.dataset_dir
    if args.dataset_dir is None:
        args.dataset_dir = _default_dir
    if args.samples_path is None:
        args.samples_path = str(Path(args.dataset_dir) / "samples.json")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figures that don't need the shared (vegetation-default) dataset load.
    # num_fire_hist loads its own with fire_type=all.
    _NO_DATASET = {"sample_density", "fire_types", "regions",
                    "cluster_lc", "num_fire_hist"}

    targets = args.figures or list(_FIGURES.keys())
    is_full_run = not args.figures  # no explicit selection = all figures

    # Lazy-load dataset only when a figure actually needs it
    ds = None
    for key in targets:
        if key not in _FIGURES:
            print(f"Unknown figure '{key}'. Use --list to see available.")
            continue
        if ds is None and key not in _NO_DATASET:
            cfg = NextDayConfig(
                **({"dataset_dir": args.dataset_dir} if args.dataset_dir else {})
            )
            ds = NextDayDataset(cfg)
        print(f"\n=== {key} ===")
        _FIGURES[key](ds, args, out_dir)

    # Always regenerate the LaTeX manifest (scans existing PDFs on disk).
    from firecomp.next_day.latex import build_figures_tex, write_tex
    sections = build_figures_tex(out_dir)
    tex_path = out_dir / "figures.tex"
    write_tex(tex_path, sections)
    print(f"\nWrote {tex_path} ({len(sections)} figures)")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figure PDFs")
    parser.add_argument(
        "figures", nargs="*",
        help="Figure keys to generate (default: all). Use --list to see options.",
    )
    parser.add_argument("--list", action="store_true", help="List available figures")
    parser.add_argument("--out-dir", default="data/figures",
                        help="Output directory (default: data/figures)")
    parser.add_argument("--dataset-dir", default=None,
                        help="Path to next-day dataset (default: from config)")
    parser.add_argument("--sample-idx", type=int, default=42,
                        help="Train sample index for input channels (default: 42)")
    parser.add_argument("--cluster-year", type=int, default=2020,
                        help="AE embedding year for cluster burnability (default: 2020)")
    parser.add_argument("--samples-path", default=None,
                        help="Path to samples.json (for sample_density figure)")
    parser.add_argument("--checkpoint", nargs="+", default=[],
                        help="Path(s) to best.pt (for predictions figure). "
                             "Multiple checkpoints produce separate PDFs.")
    parser.add_argument("--n-predictions", type=int, default=3,
                        help="Number of test samples in main figure (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
