"""
next_day/implementations/morphological.py — Morphological expansion baseline.

Prediction = today's fire mask dilated by N pixels (binary dilation with a
disk structuring element). Simulates uniform outward spread from the existing
fire perimeter.

- For `next_mask` target: the full dilated mask is the prediction.
- For `new_fires` target: the dilated ring MINUS today's mask is the prediction
  (only the expansion zone, not the already-burning pixels).

Dilation radius is tuned on val (sweeps 1, 2, 3 px), best reported on test.

Usage:
    python -m firecomp.next_day.implementations.morphological [--dataset-dir DIR]
    python -m firecomp.next_day.implementations.morphological --target new_fires
    python -m firecomp.next_day.implementations.morphological --radii 1 2 3 5
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure

from firecomp.core.fire_stats import FireSize
from firecomp.core.metrics import Metrics
from firecomp.core.regions import Regions
from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import NextDayDataset


# ---------------------------------------------------------------------------
# main — CLI
# ---------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="morphological",
        description="Morphological expansion baseline: pred = dilated fire mask.",
    )
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--target", default="next_mask",
                        choices=["next_mask", "new_fires"])
    parser.add_argument("--fire-type", default="vegetation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Dataset split to report on (default: val, "
                             "matches grid DL eval).")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    kw = {"target_type": args.target, "fire_type": args.fire_type,
          "batch_size": args.batch_size, "device": "cpu"}
    if args.dataset_dir is not None:
        kw["dataset_dir"] = args.dataset_dir
    cfg = NextDayConfig(**kw)
    result = eval(cfg, radii=args.radii, run_dir=args.run_dir or None,
                  split=args.split)
    print(f"\nDone. Best radius={result['best_radius']}, F1={result['test_f1']}")


# ---------------------------------------------------------------------------
# eval — single entry point
# ---------------------------------------------------------------------------


def eval(cfg: NextDayConfig, ds: NextDayDataset = None,
         radii: list[int] = None, run_dir: str | None = None,
         split: str = "val") -> dict:
    """Run morphological baseline: tune radius on val, report on chosen split.

    Args:
        cfg:     dataset / task config.
        ds:      optional pre-loaded dataset (avoids re-opening H5).
        radii:   dilation radii to sweep (default [1, 2, 3]).
        run_dir: if set, write result.json here.
        split:   ``"val"`` (default) or ``"test"``.  Result dict keys use
                 the ``test_`` prefix regardless of split (matches dl_2d.py).

    Returns:
        dict with best radius and metrics for the chosen split.
    """
    assert split in ("val", "test"), f"split must be 'val' or 'test', got {split!r}"
    if radii is None:
        radii = [1, 2, 3]
    if ds is None:
        ds = NextDayDataset(cfg)

    is_new_fires = (cfg.target_type == "new_fires")
    split_samples = ds.test_samples if split == "test" else ds.val_samples

    # --- val: sweep radii ---
    print("\n--- morphological baseline: val (radius sweep) ---")
    best_radius, best_val_f1 = _tune_radius(ds, radii, is_new_fires)
    print(f"\nBest radius={best_radius}  val F1={best_val_f1:.4f}")

    # --- report on chosen split ---
    print(f"\n--- morphological baseline: {split} (radius={best_radius}) ---")
    m = _eval_split(ds, split, best_radius, is_new_fires)
    print(m)

    result: dict = {
        "baseline":       "morphological",
        "target":         cfg.target_type,
        "split":          split,
        "best_radius":    best_radius,
        "val_f1":         round(best_val_f1, 4),
        "test_f1":        round(m.f1, 4),
        "test_precision": round(m.precision, 4),
        "test_recall":    round(m.recall, 4),
        "test_iou":       round(m.iou, 4),
        "test_brier":     round(m.brier, 4),
    }

    # ---- nf_equiv (exclude burned pixels) ----
    nf_m = None
    if cfg.target_type == "next_mask":
        print(f"\n--- morphological baseline: nf_equiv {split} (radius={best_radius}) ---")
        nf_m = _eval_split_nf(ds, split, best_radius)
        print(nf_m)
        result["nf_equiv_f1"]        = round(nf_m.f1, 4)
        result["nf_equiv_precision"] = round(nf_m.precision, 4)
        result["nf_equiv_recall"]    = round(nf_m.recall, 4)

    # ---- by region ----
    fire_regions = _build_fire_regions(split_samples)
    if fire_regions is not None:
        by_region = Regions.group(m, split_samples, fire_regions=fire_regions)
        print("\nby region (next_mask):")
        for name, rm in sorted(by_region.items(), key=lambda x: x[1].f1, reverse=True):
            print(
                f"  {name:20s}  F1={rm.f1:.3f}  P={rm.precision:.3f}  "
                f"R={rm.recall:.3f}  n={rm.n_samples}"
            )
        result["by_region"] = {
            name: round(rm.f1, 4) for name, rm in by_region.items()
        }

        # nf_equiv by region
        if nf_m is not None:
            nf_by_region = Regions.group(nf_m, split_samples,
                                         fire_regions=fire_regions)
            print("\nby region (nf_equiv):")
            for name, rm in sorted(nf_by_region.items(),
                                    key=lambda x: x[1].f1, reverse=True):
                print(
                    f"  {name:20s}  F1={rm.f1:.3f}  P={rm.precision:.3f}  "
                    f"R={rm.recall:.3f}  n={rm.n_samples}"
                )
            result["nf_equiv_by_region"] = {
                name: round(rm.f1, 4) for name, rm in nf_by_region.items()
            }
    else:
        print("\nby region: skipped (region raster not available)")

    # ---- by fire size ----
    fire_stats = _load_fire_stats(cfg)
    by_size = FireSize.group(m, split_samples, fire_stats=fire_stats)
    print("\nby fire size:")
    for name, sm in by_size.items():
        print(f"  {name:10s}  F1={sm.f1:.3f}  n={sm.n_samples}")

    # ---- save ----
    if run_dir:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out / 'result.json'}")

    return result


# ---------------------------------------------------------------------------
# radius tuning
# ---------------------------------------------------------------------------


def _tune_radius(ds: NextDayDataset, radii: list[int],
                 is_new_fires: bool) -> tuple[int, float]:
    """Sweep radii on val, return (best_radius, best_f1)."""
    best_radius = radii[0]
    best_f1 = -1.0

    for r in radii:
        m = _eval_split(ds, "val", r, is_new_fires)
        print(f"  radius={r}  val F1={m.f1:.4f}  P={m.precision:.3f}  R={m.recall:.3f}")
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_radius = r

    return best_radius, best_f1


# ---------------------------------------------------------------------------
# split evaluation
# ---------------------------------------------------------------------------


def _eval_split(ds: NextDayDataset, split: str, radius: int,
                is_new_fires: bool) -> Metrics:
    """Run morphological baseline on one split."""
    loader = ds.val() if split == "val" else ds.test()
    struct = _disk_structuring_element(radius)

    all_preds, all_ys, all_masks = [], [], []
    cur_mask_ch: int | None = None

    for batch in loader:
        if cur_mask_ch is None:
            cur_mask_ch = batch.channel_names.index("cur_mask")
        cur_mask = batch.x[:, cur_mask_ch:cur_mask_ch + 1]  # (B, 1, H, W)

        pred = _dilate_batch(cur_mask, struct, is_new_fires)
        all_preds.append(pred.cpu())
        all_ys.append(batch.y.cpu())
        all_masks.append(batch.loss_mask.cpu())

    pred = torch.cat(all_preds)
    target = torch.cat(all_ys)
    mask = torch.cat(all_masks)

    return Metrics(pred, target, mask, threshold=0.5)


def _eval_split_nf(ds: NextDayDataset, split: str, radius: int) -> Metrics:
    """Morphological baseline with nf_equiv masking — exclude burned pixels.

    Same as ``_eval_split`` (next_mask mode) but the loss mask zeros out
    previously-burned pixels (``accum_t_min >= -0.1``), scoring only new spread.
    """
    loader = ds.val() if split == "val" else ds.test()
    struct = _disk_structuring_element(radius)

    all_preds, all_ys, all_masks = [], [], []
    cur_mask_ch: int | None = None
    accum_ch: int | None = None

    for batch in loader:
        if cur_mask_ch is None:
            cur_mask_ch = batch.channel_names.index("cur_mask")
            accum_ch = batch.channel_names.index("accum_t_min")
        cur_mask = batch.x[:, cur_mask_ch:cur_mask_ch + 1]
        pred = _dilate_batch(cur_mask, struct, is_new_fires=False)

        burned = (batch.x[:, accum_ch:accum_ch + 1] >= -0.1).float()
        nf_mask = batch.loss_mask * (1.0 - burned)

        all_preds.append(pred.cpu())
        all_ys.append(batch.y.cpu())
        all_masks.append(nf_mask.cpu())

    pred = torch.cat(all_preds)
    target = torch.cat(all_ys)
    mask = torch.cat(all_masks)

    return Metrics(pred, target, mask, threshold=0.5)


# ---------------------------------------------------------------------------
# dilation
# ---------------------------------------------------------------------------


def _disk_structuring_element(radius: int) -> np.ndarray:
    """Create a disk-shaped structuring element for binary dilation.

    Uses a circle of the given radius — e.g. radius=1 gives a 3x3 cross
    (4-connected), radius=2 gives a ~5x5 disk, etc.
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def _dilate_batch(cur_mask: torch.Tensor, struct: np.ndarray,
                  is_new_fires: bool) -> torch.Tensor:
    """Dilate cur_mask for a whole batch. Returns (B, 1, H, W) float tensor.

    For next_mask target: full dilated mask.
    For new_fires target: dilated ring only (expansion minus original).
    """
    B = cur_mask.shape[0]
    result = torch.zeros_like(cur_mask)

    for i in range(B):
        mask_np = cur_mask[i, 0].cpu().numpy() > 0.5
        dilated = binary_dilation(mask_np, structure=struct).astype(np.float32)
        if is_new_fires:
            # Only the expansion ring — exclude already-burning pixels
            dilated = dilated * (~mask_np).astype(np.float32)
        result[i, 0] = torch.from_numpy(dilated)

    return result


# ---------------------------------------------------------------------------
# helpers (same as dl_2d.py)
# ---------------------------------------------------------------------------


def _build_fire_regions(samples) -> dict[int, int] | None:
    """Build fire_id -> region_id mapping from sample coordinates."""
    try:
        from firecomp.core.regions import RegionRaster
        rr = RegionRaster.load()
        fire_coords = {}
        for s in samples:
            if s.fire_id not in fire_coords:
                fire_coords[s.fire_id] = (s.lon, s.lat)

        fire_ids = list(fire_coords.keys())
        lons = np.array([fire_coords[fid][0] for fid in fire_ids])
        lats = np.array([fire_coords[fid][1] for fid in fire_ids])
        region_ids = rr.lookup_coords(lons, lats)

        return {fid: int(rid) for fid, rid in zip(fire_ids, region_ids)}
    except Exception:
        return None


def _load_fire_stats(cfg) -> dict[int, int]:
    """fire_id -> num_fire_pixels, for size bucketing."""
    meta_path = Path(cfg.dataset_dir) / "fire_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return {int(k): v for k, v in json.load(f).get("fire_stats", {}).items()}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
