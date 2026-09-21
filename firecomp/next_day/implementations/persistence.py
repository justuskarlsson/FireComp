"""
next_day/implementations/persistence.py — Persistence baseline.

Prediction = today's fire mask repeated as tomorrow's prediction.
No training. One eval pass over val + test.

- For `next_mask` target: strong baseline (most fire persists day-to-day).
- For `new_fires` target: F1 ~ 0 by definition (cur_mask predicts pixels
  that are ALREADY burning, but new_fires only scores genuinely new spread).

Usage:
    python -m firecomp.next_day.implementations.persistence [--dataset-dir DIR] [--target next_mask]
    python -m firecomp.next_day.implementations.persistence --target new_fires
"""

import json
from pathlib import Path

import torch

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
        prog="persistence",
        description="Persistence baseline: pred = today's fire mask.",
    )
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--target", default="next_mask",
                        choices=["next_mask", "new_fires"])
    parser.add_argument("--fire-type", default="vegetation")
    parser.add_argument("--batch-size", type=int, default=128)
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
    result = eval(cfg, run_dir=args.run_dir or None, split=args.split)
    print(f"\nDone. F1={result['test_f1']}")


# ---------------------------------------------------------------------------
# eval — single entry point
# ---------------------------------------------------------------------------


def eval(cfg: NextDayConfig, ds: NextDayDataset = None,
         run_dir: str | None = None, split: str = "val") -> dict:
    """Run persistence baseline on the chosen split.

    For each sample, prediction = cur_mask (the current day's fire mask).
    No model, no training, no threshold tuning — just binary copy.

    Args:
        cfg:     dataset / task config.
        ds:      optional pre-loaded dataset (avoids re-opening H5).
        run_dir: if set, write result.json here.
        split:   ``"val"`` (default) or ``"test"``.  Result dict keys use
                 the ``test_`` prefix regardless of split (matches dl_2d.py).

    Returns:
        dict with metrics for the chosen split.
    """
    assert split in ("val", "test"), f"split must be 'val' or 'test', got {split!r}"
    if ds is None:
        ds = NextDayDataset(cfg)

    split_samples = ds.test_samples if split == "test" else ds.val_samples

    print(f"\n--- persistence baseline: {split} ---")
    m = _eval_split(ds, split)
    print(m)

    result: dict = {
        "baseline":       "persistence",
        "target":         cfg.target_type,
        "split":          split,
        "test_f1":        round(m.f1, 4),
        "test_precision": round(m.precision, 4),
        "test_recall":    round(m.recall, 4),
        "test_iou":       round(m.iou, 4),
        "test_brier":     round(m.brier, 4),
    }

    # ---- nf_equiv (exclude burned pixels) ----
    nf_m = None
    if cfg.target_type == "next_mask":
        print(f"\n--- persistence baseline: nf_equiv {split} ---")
        nf_m = _eval_split_nf(ds, split)
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
# split evaluation
# ---------------------------------------------------------------------------


def _eval_split(ds: NextDayDataset, split: str) -> Metrics:
    """Run persistence on one split — pred = cur_mask, threshold = 0.5."""
    loader = ds.val() if split == "val" else ds.test()

    all_preds, all_ys, all_masks = [], [], []
    cur_mask_ch: int | None = None

    for batch in loader:
        if cur_mask_ch is None:
            cur_mask_ch = batch.channel_names.index("cur_mask")
        pred = batch.x[:, cur_mask_ch:cur_mask_ch + 1]  # (B, 1, H, W)
        # Clamp to [0, 1] — cur_mask should already be binary, but be safe
        pred = pred.clamp(0.0, 1.0)
        all_preds.append(pred.cpu())
        all_ys.append(batch.y.cpu())
        all_masks.append(batch.loss_mask.cpu())

    pred = torch.cat(all_preds)
    target = torch.cat(all_ys)
    mask = torch.cat(all_masks)

    return Metrics(pred, target, mask, threshold=0.5)


def _eval_split_nf(ds: NextDayDataset, split: str) -> Metrics:
    """Persistence with nf_equiv masking — exclude burned pixels.

    Same as ``_eval_split`` but the loss mask zeros out previously-burned
    pixels (``accum_t_min >= -0.1``), scoring only genuinely new spread.
    """
    loader = ds.val() if split == "val" else ds.test()

    all_preds, all_ys, all_masks = [], [], []
    cur_mask_ch: int | None = None
    accum_ch: int | None = None

    for batch in loader:
        if cur_mask_ch is None:
            cur_mask_ch = batch.channel_names.index("cur_mask")
            accum_ch = batch.channel_names.index("accum_t_min")
        pred = batch.x[:, cur_mask_ch:cur_mask_ch + 1].clamp(0.0, 1.0)
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
# helpers (same as dl_2d.py)
# ---------------------------------------------------------------------------


def _build_fire_regions(samples) -> dict[int, int] | None:
    """Build fire_id -> region_id mapping from sample coordinates."""
    try:
        from firecomp.core.regions import RegionRaster
        import numpy as np

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
