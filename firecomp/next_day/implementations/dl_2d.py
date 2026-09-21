"""
next_day/implementations/dl_2d.py — Next-day fire spread, 2D segmentation.

UNet (or any smp encoder-decoder) predicting a per-pixel fire mask.
This file IS the implementation. Train and eval are entry points, not
framework calls.

Usage:
    python -m firecomp.next_day.implementations.dl_2d train configs/next_day_unet.json
    python -m firecomp.next_day.implementations.dl_2d eval runs/next_day_unet/best.pt
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from firecomp.core.ablation import Ablation
from firecomp.core.checkpoint import BestCheckpoint
from firecomp.core.cli import Cli
from firecomp.core.fire_stats import FireSize
from firecomp.core.losses import build_loss_fn
from firecomp.core.metrics import Metrics, find_optimal_threshold
from firecomp.core.perf import PerfTimer
from firecomp.core.regions import Regions
from firecomp.core.torch_utils import get_device, setup_precision
from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import NextDayDataset
from firecomp.models.segmentation_models import model_factory


# ---------------------------------------------------------------------------
# TrainResult — returned by train() for programmatic use
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Summary of a training run. Returned by train() for tests / sweeps."""
    losses: list[float] = field(default_factory=list)       # per-epoch avg loss
    val_f1s: list[float] = field(default_factory=list)      # per-epoch val F1
    val_briers: list[float] = field(default_factory=list)   # per-epoch val Brier
    best_f1: float = 0.0
    best_epoch: int = 0
    checkpoint_path: Path | None = None


# ---------------------------------------------------------------------------
# main — CLI
# ---------------------------------------------------------------------------


def main():
    cli = Cli(
        NextDayConfig,
        prog="next_day.dl_2d",
        description="2D UNet next-day fire spread segmentation.",
    )
    cli.command("train", train)
    cli.command("eval", _eval_cli, positional=("checkpoint",))
    cli.run()


def _eval_cli(cfg: NextDayConfig, checkpoint: str):
    eval(checkpoint)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(cfg: NextDayConfig, max_batches: int = 0) -> TrainResult:
    """Train UNet for next-day fire spread; saves best.pt by val F1.

    Args:
        max_batches: if >0, only run this many batches per epoch (smoke test).
                     0 = full epoch (default).  Also settable via cfg.max_batches;
                     the function parameter takes priority when non-zero.

    Returns:
        TrainResult with per-epoch losses, val F1s, and best checkpoint info.
    """
    max_batches = max_batches or cfg.max_batches
    device = get_device(cfg.device)
    setup_precision(device)

    ds = NextDayDataset(cfg)
    model = model_factory[cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=cfg.encoder_name,
    ).to(device)

    loss_fn = build_loss_fn(
        cfg.loss_type,
        pos_weight=cfg.pos_weight,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    # Create loaders ONCE — reuse across epochs so persistent_workers
    # stay alive (no leaked worker processes) and prefetch keeps the GPU fed.
    train_loader = ds.train()
    val_loader = ds.val()
    steps_per_epoch = (
        min(len(train_loader), max_batches) if max_batches > 0 else len(train_loader)
    )
    if cfg.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer, max_lr=cfg.lr,
            steps_per_epoch=steps_per_epoch, epochs=cfg.num_epochs,
        )
    else:
        scheduler = None

    run_dir = Path(cfg.run_dir) if cfg.run_dir else Path(f"data/runs/next_day_{cfg.tag}")
    ckpt = BestCheckpoint(run_dir, metric_name="f1")
    result = TrainResult()
    timer = PerfTimer(device, enabled=cfg.perf)
    best_threshold = 0.5

    for epoch in range(cfg.num_epochs):
        # ---- train (all batches, accumulate loss) ----
        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        n_batches = 0
        timer.reset()

        for i, batch in enumerate(train_loader):
            if max_batches > 0 and i >= max_batches:
                break
            timer.stamp("data")

            logits = model(batch.x)
            loss = loss_fn(logits, batch.y, batch.loss_mask)
            if isinstance(loss, tuple):
                loss = loss[0]  # hybrid returns (total, focal, dice)
            timer.stamp("fwd")

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.detach()
            n_batches += 1
            timer.stamp("bwd")

        avg_loss = epoch_loss.item() / max(n_batches, 1)

        # ---- validate (batched forward, no grad) ----
        model.eval()
        val_preds, val_ys, val_masks = [], [], []
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if max_batches > 0 and j >= max_batches:
                    break
                pred = torch.sigmoid(model(batch.x))
                val_preds.append(pred)
                val_ys.append(batch.y)
                val_masks.append(batch.loss_mask)

        if val_preds:
            val_pred = torch.cat(val_preds)
            val_y = torch.cat(val_ys)
            val_mask = torch.cat(val_masks)
            threshold, m = find_optimal_threshold(val_pred, val_y, val_mask)
        else:
            # Empty val split (e.g. tiny region in transfer experiments).
            # Save every epoch — no early stopping possible.
            threshold = 0.5
            m = Metrics(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))

        result.losses.append(avg_loss)
        result.val_f1s.append(m.f1)
        result.val_briers.append(m.brier)

        perf = timer.summary()
        print(
            f"epoch {epoch:3d}  loss={avg_loss:.4f}  "
            f"val F1={m.f1:.3f}  P={m.precision:.3f}  R={m.recall:.3f}  "
            f"Brier={m.brier:.8f}  t={threshold:.2f}" + (f"  {perf}" if perf else "")
        )

        if ckpt.update(model, score=m.f1, epoch=epoch, threshold=threshold, cfg=cfg):
            best_threshold = threshold
            print(f"  >> new best F1={m.f1:.3f}")
        elif cfg.patience > 0 and (epoch - ckpt.best_epoch) >= cfg.patience:
            print(f"  early stop — no improvement for {cfg.patience} epochs")
            break

    result.best_f1 = ckpt.best_score
    result.best_epoch = ckpt.best_epoch
    result.checkpoint_path = ckpt.path

    print(f"\nbest epoch {ckpt.best_epoch}  F1={ckpt.best_score:.3f}")
    print(f"checkpoint: {ckpt.path}")

    # Write result.json so callers (grid search subprocess) can read it.
    result_dict = {
        "tag":         cfg.tag,
        "model":       cfg.model_type,
        "loss":        cfg.loss_type,
        "pos_weight":  cfg.pos_weight,
        "focal_alpha": cfg.focal_alpha,
        "focal_gamma": cfg.focal_gamma,
        "target":      cfg.target_type,
        "best_epoch":  ckpt.best_epoch,
        "val_f1":      round(ckpt.best_score, 4),
        "val_brier":   round(result.val_briers[ckpt.best_epoch], 4) if result.val_briers else 0.0,
        "threshold":   round(best_threshold, 4),
        "checkpoint":  str(ckpt.path),
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def eval(checkpoint_path: str | Path, ds: NextDayDataset = None,
         dataset_dir: str | None = None, split: str = "test") -> dict:
    """Run best.pt on a dataset split; report F1 by region and by fire size.

    For ``next_mask`` models, also computes "new-fires equivalent" metrics:
    mask out all pixels that have ever burned during the fire event
    (``accum_t``) so only genuinely new spread is scored.  This matches
    the ``new_fires`` target definition and enables fair cross-target
    comparison.

    Args:
        checkpoint_path: path to best.pt.
        ds:              optional pre-loaded dataset (avoids re-loading H5).
        dataset_dir:     override the checkpoint's dataset_dir (useful when
                         the training-time path no longer exists, e.g. after
                         /dev/shm staging expired).
        split:           which dataset split to evaluate on — ``"test"``
                         (default) or ``"val"``.  Result dict keys always
                         use the ``test_`` prefix regardless of split.

    Returns:
        dict with eval metrics.  Keys always include ``test_f1``,
        ``test_precision``, ``test_recall``, ``test_iou``.  For next_mask
        models, ``nf_equiv_*`` keys are also present.
    """
    from dataclasses import replace as _replace

    device = get_device()
    ckpt = BestCheckpoint(Path(checkpoint_path).parent).load(map_location=device)
    cfg = ckpt["cfg"]
    threshold = ckpt["threshold"]

    if ds is None:
        eval_cfg = cfg
        if dataset_dir:
            eval_cfg = _replace(cfg, dataset_dir=dataset_dir)
        ds = NextDayDataset(eval_cfg)

    # Resolve split iterator + sample list
    assert split in ("test", "val"), f"split must be 'test' or 'val', got {split!r}"
    split_iter = ds.test if split == "test" else ds.val
    split_samples = ds.test_samples if split == "test" else ds.val_samples

    model = model_factory[cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=cfg.encoder_name,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Batched forward pass — move to CPU to avoid OOM on large eval sets
    test_preds, test_ys, test_masks = [], [], []
    test_accum: list[torch.Tensor] = []
    collect_accum = (cfg.target_type == "next_mask")
    accum_ch: int | None = None  # resolved on first batch via channel_names

    with torch.no_grad():
        for batch in split_iter():
            pred = torch.sigmoid(model(batch.x))
            test_preds.append(pred.cpu())
            test_ys.append(batch.y.cpu())
            test_masks.append(batch.loss_mask.cpu())
            if collect_accum:
                if accum_ch is None:
                    accum_ch = batch.channel_names.index("accum_t_min")
                test_accum.append(batch.x[:, accum_ch:accum_ch + 1].cpu())

    del model; torch.cuda.empty_cache()
    pred = torch.cat(test_preds); del test_preds
    target = torch.cat(test_ys); del test_ys
    mask = torch.cat(test_masks); del test_masks
    m = Metrics(pred, target, mask, threshold=threshold)

    print(f"\n--- {split} results (threshold={threshold:.2f}) ---")
    print(m)

    result: dict = {
        "tag":            cfg.tag,
        "model":          cfg.model_type,
        "loss":           cfg.loss_type,
        "target":         cfg.target_type,
        "split":          split,
        "val_f1":         round(ckpt.get("f1", 0), 4),
        "threshold":      round(threshold, 4),
        "test_f1":        round(m.f1, 4),
        "test_precision": round(m.precision, 4),
        "test_recall":    round(m.recall, 4),
        "test_iou":       round(m.iou, 4),
        "test_brier":     round(m.brier, 4),
        "checkpoint":     str(checkpoint_path),
    }

    # ---- new-fires equivalent (next_mask only) ----
    # Mask out ALL pixels that have ever burned during this fire event
    # (accum_t >= 0 means burned at some point; -1 = unburned).  This
    # isolates genuinely new spread — matching the new_fires target
    # definition — and enables fair cross-target comparison.
    if collect_accum and test_accum:
        accum = torch.cat(test_accum); del test_accum
        nf_mask = mask * (accum < -0.1).float(); del accum
        nf_threshold, nf_m = find_optimal_threshold(pred, target, nf_mask); del nf_mask
        print(f"\n--- new-fires equivalent (threshold={nf_threshold:.2f}) ---")
        print(nf_m)
        result["nf_equiv_f1"]        = round(nf_m.f1, 4)
        result["nf_equiv_precision"] = round(nf_m.precision, 4)
        result["nf_equiv_recall"]    = round(nf_m.recall, 4)
        result["nf_equiv_iou"]       = round(nf_m.iou, 4)
        result["nf_equiv_brier"]     = round(nf_m.brier, 4)
        result["nf_equiv_threshold"] = round(nf_threshold, 4)

    # ---- by region ----
    fire_regions = _build_fire_regions(split_samples)
    if fire_regions is not None:
        by_region = Regions.group(m, split_samples, fire_regions=fire_regions)
        print("\nby region:")
        for name, rm in sorted(by_region.items(), key=lambda x: x[1].f1, reverse=True):
            print(
                f"  {name:20s}  F1={rm.f1:.3f}  P={rm.precision:.3f}  R={rm.recall:.3f}  n={rm.n_samples}"
            )
        result["by_region"] = {
            name: round(rm.f1, 4) for name, rm in by_region.items()
        }

        # nf equiv by region (next_mask only)
        if "nf_equiv_f1" in result:
            nf_by_region = Regions.group(nf_m, split_samples, fire_regions=fire_regions)
            print("\nby region (nf equiv):")
            for name, rm in sorted(nf_by_region.items(), key=lambda x: x[1].f1, reverse=True):
                print(
                    f"  {name:20s}  F1={rm.f1:.3f}  P={rm.precision:.3f}  R={rm.recall:.3f}  n={rm.n_samples}"
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

    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_fire_regions(samples) -> dict[int, int] | None:
    """Build fire_id -> region_id mapping from sample coordinates.

    Uses RegionRaster to look up each fire's centroid. Returns None if the
    region raster is not available (desktop without GDAL/tif).
    """
    try:
        from firecomp.core.regions import RegionRaster
        import numpy as np

        rr = RegionRaster.load()

        # Deduplicate: one lookup per fire_id (use first sample's coords)
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
