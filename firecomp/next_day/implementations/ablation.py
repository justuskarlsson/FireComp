"""
next_day/implementations/ablation.py — Input feature group ablation study.

Trains one baseline model with all features, then one model per excluded
feature group.  Compares test F1 to quantify each group's contribution.

Feature groups (excluded one at a time):
  era5      — ERA5 weather (6ch: VPD, soil moisture, soil ratio, wind mag/sin/cos)
  gfs       — GFS forecast (5ch: temp, RH, wind u/v, precip)
  ae        — Terrain embeddings (5ch: PCA 1-5)
  cur_mask  — Current fire mask (1ch)
  accum     — Accumulated fire state (3ch: accum_t_min, max, count)

Total: 6 runs (1 baseline + 5 exclusions).

Usage:
    # Full experiment, 4 GPUs
    python -m firecomp.next_day.implementations.ablation --num-gpus 4

    # Smoke test
    python -m firecomp.next_day.implementations.ablation --max-batches 1 --out-dir /tmp/ablation

    # Dry run — print configs without training
    python -m firecomp.next_day.implementations.ablation --dry-run

    # Re-evaluate existing checkpoints
    python -m firecomp.next_day.implementations.ablation eval --out-dir data/runs/ablation

    # Use best config from grid search
    python -m firecomp.next_day.implementations.ablation --config data/runs/grid/best_config.json
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

# Must import osgeo-backed modules before segmentation_models — PIL (pulled
# in by smp) loads its own libtiff/libjpeg and breaks a later `from osgeo
# import gdal` with an undefined-symbol error on this conda env.
from firecomp.core.regions import Regions  # noqa: F401

from firecomp.core.checkpoint import BestCheckpoint
from firecomp.core.cli import Cli
from firecomp.core.gpu_queue import GpuQueue
from firecomp.core.metrics import Metrics, find_optimal_threshold
from firecomp.core.torch_utils import get_device
from firecomp.models.segmentation_models import model_factory
from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import NextDayDataset


TRAIN_MODULE = "firecomp.next_day.implementations.ablation"


# ---------------------------------------------------------------------------
# Ablation groups — config overrides to exclude each feature group
# ---------------------------------------------------------------------------

ABLATION_GROUPS: dict[str, dict] = {
    "no_era5":     {"include_weather": False},
    "no_gfs":      {"include_gfs": False},
    "no_ae":       {"include_terrain": False},
    "no_cur_mask": {"include_cur_mask": False},
    "no_accum":    {"include_accum_min": False,
                    "include_accum_max": False,
                    "include_accum_count": False},
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


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


def main():
    args = _parse_args()

    if args.command == "eval":
        eval_all(Path(args.out_dir), dataset_dir=args.dataset_dir,
                 num_jobs=args.num_jobs, job_index=args.job_index)
        return

    if args.command == "eval-collect":
        eval_collect(Path(args.out_dir))
        return

    if args.command == "collect":
        collect_training(Path(args.out_dir))
        return

    base = _load_base_config(args)
    num_gpus = GpuQueue.resolve_num_gpus(args.num_gpus)
    configs = build_configs(base)

    # Job splitting for SLURM arrays
    if args.num_jobs > 1:
        configs = _split_list(configs, args.num_jobs, args.job_index)
        print(f"[job {args.job_index}/{args.num_jobs}] {len(configs)} configs")

    if args.dry_run:
        _print_configs(configs)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_training(configs, out_dir, num_gpus=num_gpus,
                        max_batches=args.max_batches)
    save_summary(rows, out_dir)


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def build_configs(base: NextDayConfig) -> list[NextDayConfig]:
    """Build configs: one baseline (all features) + one per excluded group."""
    configs = [replace(base, tag="ablation_baseline")]

    for group_name, overrides in ABLATION_GROUPS.items():
        tag = f"ablation_{group_name}"
        cfg = replace(base, tag=tag, **overrides)
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------


def run_training(
    configs: list[NextDayConfig],
    out_dir: Path,
    num_gpus: int = 1,
    max_batches: int = 0,
) -> list[dict]:
    """Dispatch training runs via GpuQueue."""
    dispatch = [
        replace(cfg,
                run_dir=str(out_dir / cfg.tag),
                max_batches=max_batches)
        for cfg in configs
    ]
    return GpuQueue.run(
        dispatch, out_dir=out_dir, train_module=TRAIN_MODULE,
        num_gpus=num_gpus, label="Ablation study",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_checkpoint(
    checkpoint_path: str | Path,
    cfg: NextDayConfig,
    dataset_dir: str | None = None,
) -> dict:
    """Evaluate a single checkpoint on the test split.

    For next_mask models, also computes new-fires-equivalent metrics.
    """
    checkpoint_path = Path(checkpoint_path)
    device = get_device(cfg.device)

    ckpt = BestCheckpoint(checkpoint_path.parent).load(map_location=device)
    ckpt_cfg: NextDayConfig = ckpt["cfg"]
    threshold = ckpt["threshold"]

    eval_cfg = ckpt_cfg
    if dataset_dir:
        eval_cfg = replace(ckpt_cfg, dataset_dir=dataset_dir)
    ds = NextDayDataset(eval_cfg)

    model = model_factory[ckpt_cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=ckpt_cfg.encoder_name,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Batched test forward pass — move to CPU to avoid OOM on large test sets
    test_preds, test_ys, test_masks = [], [], []
    test_accum: list[torch.Tensor] = []
    collect_accum = (ckpt_cfg.target_type == "next_mask")
    accum_ch: int | None = None

    with torch.no_grad():
        for batch in ds.test():
            pred = torch.sigmoid(model(batch.x))
            test_preds.append(pred.cpu())
            test_ys.append(batch.y.cpu())
            test_masks.append(batch.loss_mask.cpu())
            if collect_accum:
                if accum_ch is None:
                    try:
                        accum_ch = batch.channel_names.index("accum_t_min")
                    except ValueError:
                        collect_accum = False
                        continue
                test_accum.append(batch.x[:, accum_ch:accum_ch + 1].cpu())

    del model; torch.cuda.empty_cache()
    pred = torch.cat(test_preds); del test_preds
    target = torch.cat(test_ys); del test_ys
    mask = torch.cat(test_masks); del test_masks
    m = Metrics(pred, target, mask, threshold=threshold)

    print(f"\n--- test (threshold={threshold:.2f}) ---")
    print(m)

    result = {
        "test_f1":        round(m.f1, 4),
        "test_precision": round(m.precision, 4),
        "test_recall":    round(m.recall, 4),
        "test_iou":       round(m.iou, 4),
        "test_brier":     round(m.brier, 4),
        "threshold":      round(threshold, 4),
        "n_channels":     ds.num_channels,
    }

    if collect_accum and test_accum:
        accum = torch.cat(test_accum); del test_accum
        nf_mask = mask * (accum < -0.1).float(); del accum
        nf_threshold, nf_m = find_optimal_threshold(pred, target, nf_mask); del nf_mask
        print(f"\nnf_equiv (threshold={nf_threshold:.2f}): {nf_m}")
        result["nf_equiv_f1"]        = round(nf_m.f1, 4)
        result["nf_equiv_precision"] = round(nf_m.precision, 4)
        result["nf_equiv_recall"]    = round(nf_m.recall, 4)
        result["nf_equiv_threshold"] = round(nf_threshold, 4)

    return result


def eval_all(out_dir: Path, dataset_dir: str | None = None,
             num_jobs: int = 1, job_index: int = 0):
    """Re-evaluate all checkpoints under out_dir."""
    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    )
    if not run_dirs:
        print(f"No checkpoints found in {out_dir}")
        return

    if num_jobs > 1:
        run_dirs = _split_list(run_dirs, num_jobs, job_index)
        print(f"[eval job {job_index}/{num_jobs}] {len(run_dirs)} checkpoints")

    print(f"\nEval: {len(run_dirs)} checkpoints in {out_dir}\n")

    all_results = []
    for i, run_dir in enumerate(run_dirs, 1):
        tag = run_dir.name
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(run_dirs)}]  {tag}")
        print(f"{'=' * 60}")

        ckpt = BestCheckpoint(run_dir).load(map_location="cpu")
        cfg = ckpt["cfg"]

        result = eval_checkpoint(
            str(run_dir / "best.pt"), cfg, dataset_dir=dataset_dir)
        result["tag"] = tag
        all_results.append(result)

    if num_jobs > 1:
        path = out_dir / f"ablation_eval_results_{job_index}.json"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nPartial eval results → {path}  ({len(all_results)} runs)")
    else:
        _save_eval_results(all_results, out_dir)


def eval_collect(out_dir: Path):
    """Merge per-job ablation_eval_results_*.json into one file."""
    partials = sorted(out_dir.glob("ablation_eval_results_*.json"))
    if not partials:
        print(f"No ablation_eval_results_*.json found in {out_dir}")
        return

    results = []
    for p in partials:
        with open(p) as f:
            results.extend(json.load(f))

    print(f"Collected {len(results)} eval results from {len(partials)} partial files")
    _save_eval_results(results, out_dir)


def collect_training(out_dir: Path):
    """Scan run subdirectories and collect results into summary JSON.

    Use after split-job training (--num-jobs) to gather results written by
    each independent SLURM job.
    """
    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    )
    if not run_dirs:
        print(f"No result.json found in {out_dir}/*/")
        return

    rows = []
    for d in run_dirs:
        with open(d / "result.json") as f:
            rows.append(json.load(f))

    print(f"Collected {len(rows)} results from {out_dir}/*/result.json")
    save_summary(rows, out_dir)


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------


def save_summary(rows: list[dict], out_dir: Path):
    """Collect per-run results and print comparison table."""
    results = []
    for row in rows:
        tag = row.get("tag", "")
        result = {
            "tag":        tag,
            "val_f1":     round(row.get("val_f1", 0), 4),
            "threshold":  round(row.get("threshold", 0.5), 4),
            "checkpoint": row.get("checkpoint", ""),
        }

        run_dir = Path(row.get("checkpoint", "")).parent if row.get("checkpoint") else None
        if run_dir and (run_dir / "ablation_eval.json").exists():
            with open(run_dir / "ablation_eval.json") as f:
                result.update(json.load(f))

        results.append(result)

    path = out_dir / "ablation_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results -> {path}")

    _print_comparison_table(results)


def _save_eval_results(results: list[dict], out_dir: Path):
    """Save eval-only results."""
    path = out_dir / "ablation_eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation eval results -> {path}")
    _print_comparison_table(results)


def _print_comparison_table(results: list[dict]):
    """Print ablation comparison: baseline F1 vs each excluded group."""
    if not results:
        return

    baseline = next((r for r in results if "baseline" in r.get("tag", "")), None)
    baseline_f1 = baseline.get("test_f1", 0) if baseline else 0
    baseline_nf = baseline.get("nf_equiv_f1", 0) if baseline else 0

    print("\n" + "=" * 80)
    print("Ablation Study — Feature Group Importance")
    print("=" * 80)
    print(f"\n  {'Config':<25s} | {'Ch':>3s} | {'Test F1':>8s} | "
          f"{'Drop':>8s} | {'NF equiv':>8s} | {'Drop':>8s}")
    print("  " + "-" * 76)

    for r in sorted(results, key=lambda x: x.get("test_f1", 0), reverse=True):
        tag = r.get("tag", "?").replace("ablation_", "")
        f1 = r.get("test_f1", 0)
        nf = r.get("nf_equiv_f1", 0)
        ch = r.get("n_channels", "?")
        drop = baseline_f1 - f1 if baseline_f1 else 0
        nf_drop = baseline_nf - nf if baseline_nf else 0

        drop_str = f"{drop:+.4f}" if baseline_f1 else ""
        nf_drop_str = f"{nf_drop:+.4f}" if baseline_nf else ""
        f1_str = f"{f1:.4f}" if f1 else ""
        nf_str = f"{nf:.4f}" if nf else ""

        print(f"  {tag:<25s} | {str(ch):>3s} | {f1_str:>8s} | "
              f"{drop_str:>8s} | {nf_str:>8s} | {nf_drop_str:>8s}")

    print("  " + "-" * 76)
    if baseline_f1:
        print(f"\n  Positive drop = group helps (removing it hurts performance)")


# ---------------------------------------------------------------------------
# Subprocess entry point — train + eval in one process
# ---------------------------------------------------------------------------


def _train_and_eval(cfg: NextDayConfig):
    """Train with specified feature subset, then evaluate on test split.

    Called by GpuQueue as a subprocess.  Writes ablation_eval.json alongside
    the training result.json.
    """
    from firecomp.next_day.implementations.dl_2d import train as dl2d_train

    result = dl2d_train(cfg)

    if result.checkpoint_path and result.checkpoint_path.exists():
        print("\n--- Ablation evaluation ---")
        eval_result = eval_checkpoint(str(result.checkpoint_path), cfg)

        eval_path = result.checkpoint_path.parent / "ablation_eval.json"
        with open(eval_path, "w") as f:
            json.dump(eval_result, f, indent=2)
        print(f"  Eval -> {eval_path}")

        # Append to result.json for GpuQueue collection
        result_path = result.checkpoint_path.parent / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            data.update(eval_result)
            with open(result_path, "w") as f:
                json.dump(data, f, indent=2)


def _cli_main():
    """Cli dispatcher for GpuQueue subprocesses."""
    cli = Cli(
        NextDayConfig,
        prog="next_day.ablation",
        description="Ablation study: train + eval with excluded feature group.",
    )
    cli.command("train", _train_and_eval)
    cli.run()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _load_base_config(args) -> NextDayConfig:
    """Return a base NextDayConfig from --config file or CLI defaults."""
    if args.config:
        cfg = NextDayConfig.from_file(args.config)
        if args.dataset_dir:
            cfg = replace(cfg, dataset_dir=args.dataset_dir)
        return cfg
    return NextDayConfig(
        **({"dataset_dir": args.dataset_dir} if args.dataset_dir else {}),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ablation",
        description="Feature group ablation study for next-day fire spread.",
    )
    p.add_argument(
        "command", nargs="?", default="run",
        choices=["run", "eval", "eval-collect", "collect"],
        help="'run' to train + eval (default), 'eval' to re-evaluate "
             "(supports --num-jobs), 'eval-collect' to merge partial eval "
             "results, 'collect' to gather split-job training results.",
    )
    p.add_argument("--config", default=None,
                   help="JSON config with base model settings.")
    p.add_argument("--dataset-dir", default=None,
                   help="Dataset directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print configs without training.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Batches per epoch; 0=full, 1=smoke test.")
    p.add_argument("--out-dir", default="data/runs/ablation",
                   help="Output directory.")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs for dispatch (0=auto-detect).")
    p.add_argument("--num-jobs", "-jn", type=int, default=1,
                   help="Split across N independent SLURM jobs.")
    p.add_argument("--job-index", "-ji", type=int, default=0,
                   help="This job's index (0-based).")
    return p.parse_args()


def _print_configs(configs: list[NextDayConfig]):
    """Print configs for --dry-run inspection."""
    print(f"\n{len(configs)} configs:\n")
    for i, cfg in enumerate(configs, 1):
        excluded = [g for g, overrides in ABLATION_GROUPS.items()
                    if all(getattr(cfg, k) == v for k, v in overrides.items())]
        label = excluded[0] if excluded else "all features"
        fields = NextDayDataset._select_input_fields(cfg)
        n_ch = sum(f.channels for f in fields)
        print(f"  {i}. {cfg.tag:<30s}  {label:<20s}  channels={n_ch}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _cli_main()
    else:
        main()
