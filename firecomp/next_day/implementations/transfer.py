"""
next_day/implementations/transfer.py — Cross-region transfer experiments.

Paper 1's central contribution: demonstrate that fire spread models don't
generalise across regions.  Trains a model on one (or more) regions and
evaluates on all regions, producing a train-region × test-region F1 matrix.

Three experiment modes:

  matrix   — per-region training + one global training row
  loo      — leave-one-out: train on all-minus-one, eval on held-out region only
  all      — both (default)

Training runs are dispatched as subprocesses across GPUs via GpuQueue (same
mechanism as grid_search.py).  Each subprocess trains AND runs per-region
eval before exiting — no sequential Phase 2.

Positional encoding is disabled by default for all transfer runs so the
model cannot identify the region from coordinates alone.

Usage:
    # Full experiment on cluster dataset (4 GPUs) — both target types
    python -m firecomp.next_day.implementations.transfer --num-gpus 4

    # Only new_fires target
    python -m firecomp.next_day.implementations.transfer --target-types new_fires

    # Only the transfer matrix
    python -m firecomp.next_day.implementations.transfer --mode matrix

    # Leave-one-out only
    python -m firecomp.next_day.implementations.transfer --mode loo

    # Smoke test with limited batches
    python -m firecomp.next_day.implementations.transfer \\
        --dataset-dir data/next_day --max-batches 1 \\
        --out-dir /tmp/transfer

    # Use best config from grid-search results
    python -m firecomp.next_day.implementations.transfer \\
        --config data/runs/grid/best_config.json
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from firecomp.core.checkpoint import BestCheckpoint
from firecomp.core.cli import Cli
from firecomp.core.gpu_queue import GpuQueue
from firecomp.core.metrics import Metrics, find_optimal_threshold
from firecomp.core.regions import Regions
from firecomp.models.segmentation_models import model_factory
from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import NextDayDataset

# GpuQueue launches subprocesses as:
#   python -m <TRAIN_MODULE> train --config <json>
# We point to this module so each subprocess runs _train_and_eval()
# (train + per-region eval in one process, no sequential Phase 2).
TRAIN_MODULE = "firecomp.next_day.implementations.transfer"


# ---------------------------------------------------------------------------
# TransferResult — one row of the transfer matrix
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    """Summary of one transfer training run and its per-region eval scores."""
    train_label: str                    # e.g. "California", "global", "loo_California"
    train_regions: list | None          # None = trained on all regions
    exclude_regions: list | None        # set for leave-one-out runs
    target_type: str                    # "new_fires" or "next_mask"
    by_region: dict[str, float]         # {region_name: F1}
    nf_by_region: dict[str, float] | None  # nf_equiv {region: F1} (next_mask only)
    checkpoint: str                     # path to best.pt
    best_f1: float                      # best val F1 during training


def _split_list(items: list, num_jobs: int, job_index: int) -> list:
    """Split *items* into *num_jobs* contiguous chunks, return chunk *job_index*."""
    n = len(items)
    chunk = n // num_jobs
    remainder = n % num_jobs
    if job_index < remainder:
        start = job_index * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (job_index - remainder) * chunk
        end = start + chunk
    return items[start:end]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    if args.mode == "collect":
        collect_transfer(Path(args.out_dir))
        return

    base = _load_base_config(args)
    num_gpus = GpuQueue.resolve_num_gpus(args.num_gpus)
    regions = Regions()
    target_types = args.target_types

    all_results: list[TransferResult] = []

    if args.mode in ("matrix", "all"):
        print("\n" + "=" * 60)
        print("Transfer Matrix Experiment")
        print(f"Target types: {', '.join(target_types)}")
        print("=" * 60)
        matrix_results = run_transfer_matrix(
            base, regions, args.max_batches, args.out_dir,
            num_gpus=num_gpus, target_types=target_types,
            num_jobs=args.num_jobs, job_index=args.job_index,
        )
        all_results.extend(matrix_results)

        for tt in target_types:
            tt_results = [r for r in matrix_results if r.target_type == tt]
            if not tt_results:
                continue
            matrix = build_transfer_matrix(tt_results, regions)
            save_matrix(matrix, args.out_dir, target_type=tt)
            # Save nf_equiv matrix (next_mask models only)
            nf_matrix = build_transfer_matrix(tt_results, regions, use_nf_equiv=True)
            if any(nf_matrix.values()):
                save_matrix(nf_matrix, args.out_dir, target_type="nf_equiv")
            try:
                plot_transfer_matrix(matrix, args.out_dir, target_type=tt)
            except Exception as e:
                print(f"Warning: heatmap rendering failed for {tt} — {e}")

    if args.mode in ("loo", "all"):
        print("\n" + "=" * 60)
        print("Leave-One-Out Experiment")
        print(f"Target types: {', '.join(target_types)}")
        print("=" * 60)
        loo_results = run_leave_one_out(
            base, regions, args.max_batches, args.out_dir,
            num_gpus=num_gpus, target_types=target_types,
            num_jobs=args.num_jobs, job_index=args.job_index,
        )
        all_results.extend(loo_results)
        for tt in target_types:
            tt_results = [r for r in loo_results if r.target_type == tt]
            if tt_results:
                save_loo_results(tt_results, args.out_dir, target_type=tt)
                # Save nf_equiv LOO results (next_mask models only)
                if any(r.nf_by_region for r in tt_results):
                    save_loo_results(tt_results, args.out_dir,
                                     target_type="nf_equiv",
                                     use_nf_equiv=True)

    save_all_results(all_results, args.out_dir)


# ---------------------------------------------------------------------------
# Adaptive training schedule — small regions get more epochs
# ---------------------------------------------------------------------------

def _count_region_samples(
    base: NextDayConfig, regions: Regions,
) -> tuple[dict[str, int], dict[str, int], int]:
    """Count train + val samples per region (one unfiltered dataset load).

    Returns (train_counts, val_counts, total_train_samples).
    """
    cfg = replace(base, train_regions=None, exclude_regions=None)
    ds = NextDayDataset(cfg)

    fire_regions = Regions.build_fire_regions(
        ds.train_samples + ds.val_samples
    )

    def _count(samples):
        counts: dict[str, int] = {}
        for s in samples:
            rid = fire_regions.get(s.fire_id, 0)
            if rid:
                name = regions.id_to_name(rid)
                counts[name] = counts.get(name, 0) + 1
        return counts

    train_counts = _count(ds.train_samples)
    val_counts = _count(ds.val_samples)
    total = len(ds.train_samples)
    del ds
    return train_counts, val_counts, total


def _scale_training_params(
    n_samples: int, reference_n: int,
    base_epochs: int = 15, base_patience: int = 5,
) -> tuple[int, int]:
    """Scale epochs and patience sub-linearly with sample count.

    Small regions have fast epochs but need many more to converge.
    Fourth-root scaling: 100× fewer samples → ~3× more epochs.

    Examples (reference_n=20000, base_epochs=15, base_patience=5):
        n=20000 → 15 epochs, patience 5   (global — no change)
        n=5000  → 20 epochs, patience 7
        n=1000  → 29 epochs, patience 10
        n=250   → 45 epochs, patience 15
        n=50    → 69 epochs, patience 23
    """
    if n_samples >= reference_n or n_samples == 0:
        return base_epochs, base_patience
    scale = (reference_n / n_samples) ** 0.25
    epochs = min(round(base_epochs * scale), 100)
    patience = min(round(base_patience * scale), 30)
    return max(epochs, base_epochs), max(patience, base_patience)


# ---------------------------------------------------------------------------
# Transfer matrix experiment
# ---------------------------------------------------------------------------

def run_transfer_matrix(
    base: NextDayConfig,
    regions: Regions,
    max_batches: int = 0,
    out_dir: str = "data/runs/transfer",
    num_gpus: int = 1,
    target_types: list[str] | None = None,
    num_jobs: int = 1,
    job_index: int = 0,
) -> list[TransferResult]:
    """Train on each region separately + global; evaluate on all regions.

    Each subprocess trains the model AND runs per-region eval before exiting,
    so by_region scores are already in result.json when GpuQueue collects.

    Epochs and patience are scaled up for small regions so the model has
    time to converge despite fast per-epoch wall time.

    When multiple target_types are given, configs for every (target × region)
    combination are dispatched in one GpuQueue batch.
    """
    if target_types is None:
        target_types = ["new_fires"]

    out = Path(out_dir)
    train_counts, val_counts, total_train = _count_region_samples(base, regions)
    print(f"\nRegion sample counts (total train={total_train}):")

    # --- Phase 1: build configs ---
    configs = []
    tag_to_meta: dict[str, tuple[str, str]] = {}  # tag -> (label, target_type)

    for name in regions.names():
        nt = train_counts.get(name, 0)
        nv = val_counts.get(name, 0)
        ep, pat = _scale_training_params(nt, total_train, base.num_epochs, base.patience)
        skip = " ** SKIP (no val)" if nv == 0 else ""
        print(f"  {name:20s} train={nt:5d}  val={nv:4d} → "
              f"{ep} epochs, patience {pat}{skip}")
        if nv == 0:
            continue
        for tt in target_types:
            ts = _target_short(tt)
            tag = f"transfer_{ts}_{_slug(name)}"
            cfg = replace(
                base,
                train_regions=[name],
                exclude_regions=None,
                include_pos_encoding=False,
                target_type=tt,
                num_epochs=ep,
                patience=pat,
                tag=tag,
                run_dir=str(out / tag),
                max_batches=max_batches,
            )
            configs.append(cfg)
            tag_to_meta[tag] = (name, tt)

    # Global rows (one per target type)
    for tt in target_types:
        ts = _target_short(tt)
        global_tag = f"transfer_{ts}_global"
        cfg_global = replace(
            base,
            train_regions=None,
            exclude_regions=None,
            include_pos_encoding=False,
            target_type=tt,
            tag=global_tag,
            run_dir=str(out / global_tag),
            max_batches=max_batches,
        )
        configs.append(cfg_global)
        tag_to_meta[global_tag] = ("global", tt)

    # --- Job splitting (SLURM arrays) ---
    if num_jobs > 1:
        configs = _split_list(configs, num_jobs, job_index)
        print(f"[job {job_index}/{num_jobs}] {len(configs)} matrix configs")

    # --- Dispatch training + per-region eval (all in one subprocess) ---
    n_targets = len(target_types)
    label = f"Transfer matrix ({n_targets} target{'s' * (n_targets > 1)})"
    rows = GpuQueue.run(
        configs, out_dir=out, train_module=TRAIN_MODULE,
        num_gpus=num_gpus, label=label,
    )

    # --- Collect results (by_region already in result.json) ---
    results = []
    for row in rows:
        tag = row["tag"]
        if row.get("status") == "failed" or not row.get("checkpoint"):
            print(f"  SKIP  {tag} (training failed)")
            continue
        region_label, tt = tag_to_meta[tag]
        results.append(TransferResult(
            train_label=region_label,
            train_regions=row.get("train_regions"),
            exclude_regions=row.get("exclude_regions"),
            target_type=tt,
            by_region=row.get("by_region", {}),
            nf_by_region=row.get("nf_by_region"),
            checkpoint=row["checkpoint"],
            best_f1=round(row.get("val_f1", 0), 4),
        ))

    return results


# ---------------------------------------------------------------------------
# Leave-one-out experiment
# ---------------------------------------------------------------------------

def run_leave_one_out(
    base: NextDayConfig,
    regions: Regions,
    max_batches: int = 0,
    out_dir: str = "data/runs/transfer",
    num_gpus: int = 1,
    target_types: list[str] | None = None,
    num_jobs: int = 1,
    job_index: int = 0,
) -> list[TransferResult]:
    """For each region, train on all-minus-that-region; evaluate on all-minus-that-region.

    The held-out region is excluded from ALL splits (train, val, test).
    This answers the "data pollution" question: does including region X
    in training hurt performance on the remaining regions?

    Each subprocess trains the model AND runs per-region eval before exiting.

    Epochs/patience scaled for the remaining sample count after exclusion.
    When multiple target_types are given, configs for every (target × region)
    combination are dispatched in one GpuQueue batch.
    """
    if target_types is None:
        target_types = ["new_fires"]

    out = Path(out_dir)
    train_counts, _, total_train = _count_region_samples(base, regions)

    # --- Phase 1: build configs ---
    configs = []
    tag_to_meta: dict[str, tuple[str, str]] = {}  # tag -> (label, target_type)

    for region_name in regions.names():
        n_remaining = total_train - train_counts.get(region_name, 0)
        epochs, patience = _scale_training_params(
            n_remaining, total_train, base.num_epochs, base.patience,
        )
        print(f"  {region_name:20s}  remaining={n_remaining:5d} → "
              f"{epochs} epochs, patience {patience}")
        for tt in target_types:
            ts = _target_short(tt)
            tag = f"loo_{ts}_{_slug(region_name)}"
            cfg = replace(
                base,
                train_regions=None,
                exclude_regions=[region_name],
                include_pos_encoding=False,
                target_type=tt,
                num_epochs=epochs,
                patience=patience,
                tag=tag,
                run_dir=str(out / tag),
                max_batches=max_batches,
            )
            configs.append(cfg)
            tag_to_meta[tag] = (f"loo_{region_name}", tt)

    # --- Job splitting (SLURM arrays) ---
    if num_jobs > 1:
        configs = _split_list(configs, num_jobs, job_index)
        print(f"[job {job_index}/{num_jobs}] {len(configs)} LOO configs")

    # --- Dispatch training + per-region eval (all in one subprocess) ---
    n_targets = len(target_types)
    label = f"Leave-one-out ({n_targets} target{'s' * (n_targets > 1)})"
    rows = GpuQueue.run(
        configs, out_dir=out, train_module=TRAIN_MODULE,
        num_gpus=num_gpus, label=label,
    )

    # --- Collect results (by_region already in result.json) ---
    results = []
    for row in rows:
        tag = row["tag"]
        if row.get("status") == "failed" or not row.get("checkpoint"):
            print(f"  SKIP  {tag} (training failed)")
            continue
        loo_label, tt = tag_to_meta[tag]
        results.append(TransferResult(
            train_label=loo_label,
            train_regions=row.get("train_regions"),
            exclude_regions=row.get("exclude_regions"),
            target_type=tt,
            by_region=row.get("by_region", {}),
            nf_by_region=row.get("nf_by_region"),
            checkpoint=row["checkpoint"],
            best_f1=round(row.get("val_f1", 0), 4),
        ))

    return results


# ---------------------------------------------------------------------------
# Per-region evaluation
# ---------------------------------------------------------------------------

def eval_by_region(
    checkpoint_path: "Path | str",
    cfg: NextDayConfig,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Load a checkpoint and score the test split, grouped by region.

    ``train_regions`` is cleared (eval doesn't need a filtered train split).
    ``exclude_regions`` is preserved — for LOO the held-out region is
    excluded from the test set too (data pollution design).

    For ``next_mask`` models, also computes new-fires-equivalent metrics
    (masks out previously burned pixels via accum_t).

    Args:
        checkpoint_path: path to best.pt saved by BestCheckpoint.
        cfg:             config used for training (provides dataset location
                         and model architecture).

    Returns:
        (by_region, nf_by_region) where each is {region_name: F1}.
        nf_by_region is None for non-next_mask models.
        Falls back to {"global": F1} if the region raster is unavailable.
    """
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(cfg.device)

    # Load saved checkpoint (contains model weights, threshold, and training cfg)
    ckpt_data = BestCheckpoint(checkpoint_path.parent).load(map_location=cfg.device)
    threshold = ckpt_data["threshold"]
    ckpt_cfg: NextDayConfig = ckpt_data["cfg"]

    # Clear train_regions (not needed for eval).  Keep exclude_regions —
    # for LOO, the held-out region is excluded from test too.
    # For matrix runs exclude_regions is already None, so no change.
    eval_cfg = replace(ckpt_cfg, train_regions=None)
    ds = NextDayDataset(eval_cfg)

    model = model_factory[ckpt_cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=ckpt_cfg.encoder_name,
    ).to(device)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    # Batched forward pass — move to CPU immediately to avoid OOM on large
    # test sets (24K+ samples × 256×256 ≈ 25 GB per tensor on GPU).
    collect_accum = (ckpt_cfg.target_type == "next_mask")
    accum_ch: int | None = None
    preds, ys, masks = [], [], []
    accum_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in ds.test():
            pred = torch.sigmoid(model(batch.x)).cpu()
            preds.append(pred)
            ys.append(batch.y.cpu())
            masks.append(batch.loss_mask.cpu())
            if collect_accum:
                if accum_ch is None:
                    accum_ch = batch.channel_names.index("accum_t_min")
                accum_list.append(batch.x[:, accum_ch:accum_ch + 1].cpu())

    # Free GPU memory before CPU-side metrics
    del model
    torch.cuda.empty_cache()

    pred = torch.cat(preds); del preds
    target = torch.cat(ys); del ys
    mask = torch.cat(masks); del masks
    m = Metrics(pred, target, mask, threshold=threshold)

    # Global F1 (before region breakdown)
    result: dict[str, float] = {"global": round(m.f1, 4)}

    # Group by region
    fire_regions = Regions.build_fire_regions(ds.test_samples)
    if fire_regions is None:
        return result, None

    by_region = Regions.group(m, ds.test_samples, fire_regions=fire_regions)
    for name, rm in by_region.items():
        result[name] = round(rm.f1, 4)

    # ---- new-fires equivalent (next_mask only) ----
    nf_result: dict[str, float] | None = None
    if collect_accum and accum_list:
        accum = torch.cat(accum_list); del accum_list
        nf_mask = mask * (accum < -0.1).float(); del accum
        nf_threshold, nf_m = find_optimal_threshold(pred, target, nf_mask); del nf_mask
        nf_result = {"global": round(nf_m.f1, 4)}

        nf_by_region = Regions.group(nf_m, ds.test_samples, fire_regions=fire_regions)
        for name, rm in nf_by_region.items():
            nf_result[name] = round(rm.f1, 4)

        print(f"  nf_equiv: global F1={nf_m.f1:.3f} (threshold={nf_threshold:.2f})")

    return result, nf_result


# ---------------------------------------------------------------------------
# Transfer matrix aggregation
# ---------------------------------------------------------------------------

def build_transfer_matrix(
    results: list[TransferResult],
    regions: Regions,
    use_nf_equiv: bool = False,
) -> dict[str, dict[str, float]]:
    """Aggregate per-run TransferResults into a train × test matrix.

    Args:
        results:      one TransferResult per training run.
        regions:      Regions registry (used for ordering only).
        use_nf_equiv: if True, use nf_by_region instead of by_region.

    Returns:
        {train_label: {test_region: F1}}
    """
    if use_nf_equiv:
        return {r.train_label: dict(r.nf_by_region)
                for r in results if r.nf_by_region}
    return {r.train_label: dict(r.by_region) for r in results}


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_matrix(
    matrix: dict[str, dict[str, float]],
    out_dir: str,
    target_type: str | None = None,
) -> Path:
    """Write transfer matrix as JSON + CSV; return path to JSON.

    When target_type is given, filenames include it as a suffix
    (e.g. transfer_matrix_new_fires.json).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = f"_{target_type}" if target_type else ""
    json_path = out / f"transfer_matrix{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(matrix, f, indent=2)

    # CSV: rows = train regions, cols = all test regions (sorted)
    all_test_regions = sorted({k for row in matrix.values() for k in row})
    csv_path = out / f"transfer_matrix{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_region"] + all_test_regions)
        for train_label, row in matrix.items():
            writer.writerow(
                [train_label] + [row.get(r, "") for r in all_test_regions]
            )

    label = f" [{target_type}]" if target_type else ""
    print(f"\nTransfer matrix{label} → {json_path}  ({len(matrix)} train rows)")
    return json_path


def save_loo_results(
    results: list[TransferResult],
    out_dir: str,
    target_type: str | None = None,
    use_nf_equiv: bool = False,
) -> Path:
    """Write leave-one-out results as JSON.

    When *use_nf_equiv* is True, saves ``nf_by_region`` instead of
    ``by_region`` (new-fires-equivalent metrics for next_mask models).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    loo_data = []
    for r in results:
        held_out = r.exclude_regions[0] if r.exclude_regions else None
        by_region = (r.nf_by_region if use_nf_equiv and r.nf_by_region
                     else r.by_region)
        loo_data.append({
            "train_label":    r.train_label,
            "target_type":    r.target_type,
            "exclude_region": held_out,
            "global_f1":      by_region.get("global"),
            "by_region":      by_region,
            "checkpoint":     r.checkpoint,
        })

    suffix = f"_{target_type}" if target_type else ""
    path = out / f"loo_results{suffix}.json"
    with open(path, "w") as f:
        json.dump(loo_data, f, indent=2)
    label = f" [{target_type}]" if target_type else ""
    print(f"LOO results{label} → {path}  ({len(loo_data)} runs)")
    return path


def save_all_results(results: list[TransferResult], out_dir: str) -> Path:
    """Write all run summaries as a single JSON for downstream analysis."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "all_results.json"
    data = [
        {
            "train_label":     r.train_label,
            "target_type":     r.target_type,
            "train_regions":   r.train_regions,
            "exclude_regions": r.exclude_regions,
            "by_region":       r.by_region,
            "checkpoint":      r.checkpoint,
            "best_f1":         r.best_f1,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"All results → {path}  ({len(data)} runs)")
    return path


def collect_transfer(out_dir: Path):
    """Scan all result.json files under *out_dir* and rebuild matrix/LOO JSONs.

    Use after split-job runs (``--num-jobs``) to aggregate results written
    by independent SLURM array tasks into the transfer_matrix_*.json and
    loo_results_*.json files that tables.py expects.
    """
    regions = Regions()

    # Discover all result.json files
    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    )
    if not run_dirs:
        print(f"No result.json found in {out_dir}/*/")
        return

    matrix_results: list[TransferResult] = []
    loo_results: list[TransferResult] = []

    for d in run_dirs:
        with open(d / "result.json") as f:
            row = json.load(f)
        tag = row.get("tag", d.name)

        # Infer target type from tag: transfer_nm_* or transfer_nf_* or loo_nm_* etc.
        if "_nm_" in tag:
            tt = "next_mask"
        elif "_nf_" in tag:
            tt = "new_fires"
        else:
            tt = row.get("target_type", "next_mask")

        is_loo = tag.startswith("loo_")
        if is_loo:
            # loo_nm_africa → exclude_region = "africa" (slug)
            slug = tag.split("_", 2)[2] if tag.count("_") >= 2 else tag
            # Reverse-lookup region name from slug
            region_name = None
            for rn in regions.names():
                if _slug(rn) == slug:
                    region_name = rn
                    break
            tr = TransferResult(
                train_label=f"loo_{region_name or slug}",
                train_regions=row.get("train_regions"),
                exclude_regions=row.get("exclude_regions") or ([region_name] if region_name else None),
                target_type=tt,
                by_region=row.get("by_region", {}),
                nf_by_region=row.get("nf_by_region"),
                checkpoint=row.get("checkpoint", str(d / "best.pt")),
                best_f1=round(row.get("val_f1", 0), 4),
            )
            loo_results.append(tr)
        else:
            # transfer_nm_global, transfer_nm_africa, etc.
            slug = tag.split("_", 2)[2] if tag.count("_") >= 2 else tag
            if slug == "global":
                region_label = "global"
            else:
                region_label = slug
                for rn in regions.names():
                    if _slug(rn) == slug:
                        region_label = rn
                        break
            tr = TransferResult(
                train_label=region_label,
                train_regions=row.get("train_regions"),
                exclude_regions=row.get("exclude_regions"),
                target_type=tt,
                by_region=row.get("by_region", {}),
                nf_by_region=row.get("nf_by_region"),
                checkpoint=row.get("checkpoint", str(d / "best.pt")),
                best_f1=round(row.get("val_f1", 0), 4),
            )
            matrix_results.append(tr)

    print(f"Collected {len(matrix_results)} matrix + {len(loo_results)} LOO "
          f"results from {len(run_dirs)} run dirs")

    # Rebuild matrix JSONs
    for tt in sorted({r.target_type for r in matrix_results}):
        tt_results = [r for r in matrix_results if r.target_type == tt]
        matrix = build_transfer_matrix(tt_results, regions)
        save_matrix(matrix, str(out_dir), target_type=tt)
        nf_matrix = build_transfer_matrix(tt_results, regions, use_nf_equiv=True)
        if any(nf_matrix.values()):
            save_matrix(nf_matrix, str(out_dir), target_type="nf_equiv")

    # Rebuild LOO JSONs
    for tt in sorted({r.target_type for r in loo_results}):
        tt_results = [r for r in loo_results if r.target_type == tt]
        save_loo_results(tt_results, str(out_dir), target_type=tt)
        if any(r.nf_by_region for r in tt_results):
            save_loo_results(tt_results, str(out_dir),
                             target_type="nf_equiv", use_nf_equiv=True)


# ---------------------------------------------------------------------------
# Heatmap plot
# ---------------------------------------------------------------------------

def plot_transfer_matrix(
    matrix: dict[str, dict[str, float]],
    out_dir: str,
    target_type: str | None = None,
) -> Path:
    """Render the transfer matrix as a colour-coded PNG heatmap.

    Args:
        matrix:      {train_label: {test_region: F1}}
        out_dir:     output directory.
        target_type: e.g. "new_fires" — used in title and filename.

    Returns path to the saved PNG.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")       # non-interactive backend — safe in subprocesses
    import matplotlib.pyplot as plt

    # --- Clean up: remove unknown_0 from both axes ---
    matrix = {
        tl: {tr: v for tr, v in row.items() if tr != "unknown_0"}
        for tl, row in matrix.items()
        if tl != "unknown_0"
    }

    # --- Canonical region order (registry id order), "global" last ---
    region_order = Regions().names()  # sorted by id

    # Collect all region names that appear in either axis
    all_train = set(matrix.keys())
    all_test = {k for row in matrix.values() for k in row}

    # Ordered: known regions first (by registry order), then unknowns, then global
    def _sort_key(name: str) -> tuple[int, str]:
        if name == "global":
            return (2, "")
        try:
            idx = region_order.index(name)
            return (0, f"{idx:04d}")
        except ValueError:
            return (1, name)

    train_labels = sorted(all_train, key=_sort_key)
    test_regions = sorted(all_test, key=_sort_key)

    data = np.full((len(train_labels), len(test_regions)), np.nan)
    for i, tl in enumerate(train_labels):
        for j, tr in enumerate(test_regions):
            if tr in matrix[tl]:
                data[i, j] = matrix[tl][tr]

    fig_w = max(8, len(test_regions) * 1.3)
    fig_h = max(4, len(train_labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Color scale between data min/max (not fixed 0–1)
    vmin = float(np.nanmin(data)) if not np.all(np.isnan(data)) else 0.0
    vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0
    tt_display = target_type or "F1"
    im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label=f"F1 ({tt_display})", fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(test_regions)))
    ax.set_xticklabels(test_regions, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(train_labels)))
    ax.set_yticklabels(train_labels, fontsize=8)
    ax.set_xlabel("Eval region", fontsize=10)
    ax.set_ylabel("Train region", fontsize=10)
    ax.set_title(f"Cross-region transfer: F1 ({tt_display})", fontsize=11)

    # Annotate each cell with its value
    for i in range(len(train_labels)):
        for j in range(len(test_regions)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center", fontsize=6, color="black")

    plt.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{target_type}" if target_type else ""
    fig_path = out / f"transfer_matrix{suffix}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap → {fig_path}")
    return fig_path


def plot_transfer_matrix_from_json(
    json_path: str,
    out_dir: str | None = None,
    target_type: str | None = None,
) -> Path:
    """Load a saved transfer matrix JSON and render the heatmap PNG.

    Standalone entry point for re-generating the plot without re-running
    training.  Useful for iterating on visualisation.

    Args:
        json_path:   path to transfer_matrix*.json (as written by save_matrix).
        out_dir:     output directory; defaults to same directory as json_path.
        target_type: label for title/filename; auto-detected from filename if
                     omitted (e.g. "transfer_matrix_new_fires.json" → "new_fires").

    Returns path to the saved PNG.
    """
    jp = Path(json_path)
    with open(jp) as f:
        matrix = json.load(f)

    if out_dir is None:
        out_dir = str(jp.parent)

    if target_type is None:
        # Try to infer from filename: transfer_matrix_<target>.json
        stem = jp.stem  # e.g. "transfer_matrix_new_fires"
        prefix = "transfer_matrix_"
        if stem.startswith(prefix) and len(stem) > len(prefix):
            target_type = stem[len(prefix):]

    return plot_transfer_matrix(matrix, out_dir, target_type=target_type)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Region name → filesystem-safe slug (e.g. 'North America' → 'north_america')."""
    return name.lower().replace(" ", "_").replace("/", "_")


def _target_short(target_type: str) -> str:
    """Short tag suffix for target type (keeps directory names concise)."""
    return {"new_fires": "nf", "next_mask": "nm"}.get(target_type, target_type)


def _load_base_config(args) -> NextDayConfig:
    """Return a base NextDayConfig from --config file or CLI flags."""
    if args.config:
        cfg = NextDayConfig.from_file(args.config)
        # Override dataset location if explicitly specified
        if args.dataset_dir is not None:
            cfg = replace(cfg, dataset_dir=args.dataset_dir)
        return cfg

    kw = {"dataset_version": args.dataset_version,
          "include_pos_encoding": False}
    if args.dataset_dir is not None:
        kw["dataset_dir"] = args.dataset_dir
    return NextDayConfig(**kw)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transfer",
        description="Cross-region transfer learning experiments for Paper 1.",
    )
    p.add_argument(
        "--mode", choices=["matrix", "loo", "all", "collect"],
        default="all",
        help="Which experiment to run (default: all). "
             "'collect' scans result.json files and rebuilds matrix/LOO JSONs.",
    )
    p.add_argument(
        "--target-types", nargs="+",
        default=["new_fires", "next_mask"],
        help="Target types to run (default: both new_fires and next_mask).",
    )
    p.add_argument(
        "--config", default=None,
        help="JSON config file with base model settings (from grid search).",
    )
    p.add_argument(
        "--dataset-dir", default=None,
        help="Dataset directory (default: from config).",
    )
    p.add_argument(
        "--dataset-version", default="latest",
        help="Dataset version sub-directory (default: latest).",
    )
    p.add_argument(
        "--max-batches", type=int, default=0,
        help="Batches per epoch; 0=full epoch, 1=smoke test.",
    )
    p.add_argument(
        "--out-dir", default="data/runs/transfer",
        help="Output directory for results JSON/CSV and heatmap.",
    )
    p.add_argument(
        "--num-gpus", type=int, default=1,
        help="GPUs for queue dispatch (default: 1 = one at a time, "
             "0 = auto-detect all available GPUs).",
    )
    p.add_argument(
        "--num-jobs", "-jn", type=int, default=1,
        help="Split configs across N independent SLURM jobs (use with --job-index).",
    )
    p.add_argument(
        "--job-index", "-ji", type=int, default=0,
        help="This job's index (0-based, use with --num-jobs).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess entry point — train + per-region eval in one process
# ---------------------------------------------------------------------------


def _train_and_eval(cfg: NextDayConfig):
    """Train on configured regions, then evaluate per-region on full test set.

    Called by GpuQueue as a subprocess.  Writes by_region scores into the
    existing result.json so the orchestrator can read them back.
    """
    from firecomp.next_day.implementations.dl_2d import train as dl2d_train

    result = dl2d_train(cfg)

    if result.checkpoint_path and result.checkpoint_path.exists():
        print("\n--- Per-region evaluation ---")
        by_region, nf_by_region = eval_by_region(str(result.checkpoint_path), cfg)

        # Append by_region (+ nf_equiv) to result.json
        result_path = result.checkpoint_path.parent / "result.json"
        with open(result_path) as f:
            data = json.load(f)
        data["by_region"] = by_region
        if nf_by_region is not None:
            data["nf_by_region"] = nf_by_region
        data["train_regions"] = cfg.train_regions
        data["exclude_regions"] = cfg.exclude_regions
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)
        n_nf = f" + {len(nf_by_region)} nf_equiv" if nf_by_region else ""
        print(f"  Updated {result_path} with {len(by_region)} region scores{n_nf}")


def _cli_main():
    """Cli dispatcher for GpuQueue subprocesses (train + eval)."""
    cli = Cli(
        NextDayConfig,
        prog="next_day.transfer",
        description="Transfer learning: train + per-region eval.",
    )
    cli.command("train", _train_and_eval)
    cli.run()


# ---------------------------------------------------------------------------

def _plot_main():
    """CLI entry point for re-generating heatmaps from saved JSON matrices."""
    p = argparse.ArgumentParser(
        prog="transfer plot",
        description="Re-render transfer matrix heatmap from saved JSON.",
    )
    p.add_argument(
        "json_path",
        help="Path to transfer_matrix*.json (from save_matrix).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory for PNG (default: same as JSON file).",
    )
    p.add_argument(
        "--target-type", default=None,
        help="Target type label (auto-detected from filename if omitted).",
    )
    args = p.parse_args(sys.argv[2:])
    path = plot_transfer_matrix_from_json(
        args.json_path, out_dir=args.out_dir, target_type=args.target_type,
    )
    print(f"Done → {path}")


if __name__ == "__main__":
    # GpuQueue calls: python -m ... transfer train --config <json>
    # User calls:     python -m ... transfer [--mode matrix] [--num-gpus 4]
    # Plot only:      python -m ... transfer plot <json_path>
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _cli_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        _plot_main()
    else:
        main()
