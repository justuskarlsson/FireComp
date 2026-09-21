"""
next_day/implementations/fire_type_study.py — Fire type composition experiments.

Two questions this experiment answers:

  1. **Training composition**: Does including non-vegetation fires (static,
     crop) in training help or hurt vegetation fire prediction?
  2. **Per-type difficulty**: How does prediction difficulty vary across fire
     types?  Static fires (persistent heat) should be trivially predictable;
     crop fires (ephemeral burns) should be hardest.

Training subsets: all, vegetation, crop, static.  All models use
target_type="next_mask" (found best in grid search).  Evaluation computes
both next_mask and new-fires-equivalent metrics.

Each trained model is evaluated on the FULL test set (fire_type="all"), then
metrics are grouped by fire_type (vegetation / static / crop).

Each config is trained ``--num-runs`` times (default 3) for mean ± std.

Usage:
    # Full experiment (4 training subsets × 3 runs), single machine
    python -m firecomp.next_day.implementations.fire_type_study --num-gpus 4

    # SLURM: split across 4 jobs (one fire type per job, 3 runs each)
    python -m firecomp.next_day.implementations.fire_type_study \\
        --num-jobs 4 --job-index $SLURM_ARRAY_TASK_ID
    # After all jobs finish:
    python -m firecomp.next_day.implementations.fire_type_study collect

    # Smoke test
    python -m firecomp.next_day.implementations.fire_type_study --max-batches 1 --out-dir /tmp/ft

    # Dry run — print configs without training
    python -m firecomp.next_day.implementations.fire_type_study --dry-run

    # Re-evaluate existing checkpoints without retraining
    python -m firecomp.next_day.implementations.fire_type_study eval --out-dir data/runs/fire_type

    # Use best config from grid search
    python -m firecomp.next_day.implementations.fire_type_study \\
        --config data/runs/grid/best_config.json
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


# GpuQueue calls back into this module for subprocesses.
TRAIN_MODULE = "firecomp.next_day.implementations.fire_type_study"

FIRE_TYPE_NAMES = {0: "vegetation", 1: "static", 2: "crop"}

# Training configs: which fire_type subsets to train on.
TRAIN_FIRE_TYPES = ["all", "vegetation", "crop", "static"]


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# main — CLI
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    if args.command == "eval":
        eval_all(Path(args.out_dir), dataset_dir=args.dataset_dir)
        return

    if args.command == "collect":
        collect(Path(args.out_dir))
        return

    base = _load_base_config(args)
    num_gpus = GpuQueue.resolve_num_gpus(args.num_gpus)

    configs = build_configs(base, num_runs=args.num_runs)

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


def build_configs(
    base: NextDayConfig,
    train_fire_types: list[str] | None = None,
    num_runs: int = 1,
) -> list[NextDayConfig]:
    """Build configs for the fire_type study.

    All configs use target_type="next_mask" (train once, eval on both
    next_mask and nf_equiv).

    When *num_runs* > 1, each fire-type config is repeated that many
    times with distinct tags (``ft_all_run1``, ``ft_all_run2``, …) so
    that mean ± std-dev can be reported.
    """
    fire_types = train_fire_types or TRAIN_FIRE_TYPES
    configs = []
    for ft in fire_types:
        for run_i in range(1, num_runs + 1):
            tag = f"ft_{ft}" if num_runs == 1 else f"ft_{ft}_run{run_i}"
            cfg = replace(
                base,
                fire_type=ft,
                target_type="next_mask",
                tag=tag,
            )
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
    """Dispatch training runs via GpuQueue. Each subprocess trains + evals."""
    dispatch = [
        replace(cfg,
                run_dir=str(out_dir / cfg.tag),
                max_batches=max_batches)
        for cfg in configs
    ]
    return GpuQueue.run(
        dispatch, out_dir=out_dir, train_module=TRAIN_MODULE,
        num_gpus=num_gpus, label="Fire type study",
    )


# ---------------------------------------------------------------------------
# Evaluation — group test metrics by fire_type
# ---------------------------------------------------------------------------


def eval_by_fire_type(
    checkpoint_path: str | Path,
    cfg: NextDayConfig,
    dataset_dir: str | None = None,
) -> dict:
    """Load a checkpoint and evaluate on ALL fire types, grouped by type.

    Creates a dataset with fire_type="all" regardless of what the model was
    trained on, so the test split contains vegetation + static + crop samples.

    For next_mask models, also computes new-fires-equivalent metrics per type.

    Returns:
        {
            "global": {"f1": ..., "nf_equiv_f1": ...},
            "by_fire_type": {
                "vegetation": {"f1": ..., "n": ..., "nf_equiv_f1": ...},
                "static":     {"f1": ..., "n": ..., "nf_equiv_f1": ...},
                "crop":       {"f1": ..., "n": ..., "nf_equiv_f1": ...},
            }
        }
    """
    checkpoint_path = Path(checkpoint_path)
    device = get_device(cfg.device)

    ckpt = BestCheckpoint(checkpoint_path.parent).load(map_location=device)
    ckpt_cfg: NextDayConfig = ckpt["cfg"]
    threshold = ckpt["threshold"]

    # Load dataset with ALL fire types (override training fire_type)
    eval_cfg = replace(ckpt_cfg, fire_type="all")
    if dataset_dir:
        eval_cfg = replace(eval_cfg, dataset_dir=dataset_dir)
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
    accum_ch: int | None = None

    with torch.no_grad():
        for batch in ds.test():
            pred = torch.sigmoid(model(batch.x))
            test_preds.append(pred.cpu())
            test_ys.append(batch.y.cpu())
            test_masks.append(batch.loss_mask.cpu())
            if accum_ch is None:
                accum_ch = batch.channel_names.index("accum_t_min")
            # .cpu() breaks the view back to batch.x and moves off GPU —
            # otherwise the full 18-channel input tensor is retained per batch.
            test_accum.append(batch.x[:, accum_ch:accum_ch + 1].cpu())

    del model; torch.cuda.empty_cache()
    pred = torch.cat(test_preds); del test_preds
    target = torch.cat(test_ys); del test_ys
    mask = torch.cat(test_masks); del test_masks
    accum = torch.cat(test_accum); del test_accum

    # --- Global metrics ---
    m = Metrics(pred, target, mask, threshold=threshold)
    print(f"\n--- global test (threshold={threshold:.2f}) ---")
    print(m)

    # nf_equiv global
    nf_mask = mask * (accum < -0.1).float(); del accum
    nf_threshold, nf_m = find_optimal_threshold(pred, target, nf_mask); del nf_mask
    print(f"nf_equiv (threshold={nf_threshold:.2f}): {nf_m}")

    result: dict = {
        "global": {
            "f1":         round(m.f1, 4),
            "precision":  round(m.precision, 4),
            "recall":     round(m.recall, 4),
            "iou":        round(m.iou, 4),
            "n":          m.n_samples,
            "nf_equiv_f1":        round(nf_m.f1, 4),
            "nf_equiv_precision": round(nf_m.precision, 4),
            "nf_equiv_recall":    round(nf_m.recall, 4),
        },
    }

    # --- Group by fire_type ---
    samples = ds.test_samples
    per_sample_nm = m.per_sample
    per_sample_nf = nf_m.per_sample
    assert len(per_sample_nm) == len(samples)

    by_ft: dict[str, dict] = {}
    print("\nby fire_type:")
    for ft_val, ft_name in FIRE_TYPE_NAMES.items():
        indices = [i for i, s in enumerate(samples) if s.fire_type == ft_val]
        if not indices:
            continue

        ft_nm = Metrics.from_subset([per_sample_nm[i] for i in indices])
        ft_nf = Metrics.from_subset([per_sample_nf[i] for i in indices])

        print(f"  {ft_name:12s}  F1={ft_nm.f1:.3f}  P={ft_nm.precision:.3f}  "
              f"R={ft_nm.recall:.3f}  n={ft_nm.n_samples}  |  "
              f"nf_equiv F1={ft_nf.f1:.3f}")

        by_ft[ft_name] = {
            "f1":         round(ft_nm.f1, 4),
            "precision":  round(ft_nm.precision, 4),
            "recall":     round(ft_nm.recall, 4),
            "iou":        round(ft_nm.iou, 4),
            "brier":      round(ft_nm.brier, 4),
            "n":          ft_nm.n_samples,
            "nf_equiv_f1":        round(ft_nf.f1, 4),
            "nf_equiv_precision": round(ft_nf.precision, 4),
            "nf_equiv_recall":    round(ft_nf.recall, 4),
        }

    result["by_fire_type"] = by_ft
    return result


# ---------------------------------------------------------------------------
# Re-evaluate existing checkpoints
# ---------------------------------------------------------------------------


def eval_all(out_dir: Path, dataset_dir: str | None = None):
    """Re-evaluate all checkpoints under out_dir, grouped by fire_type.

    Useful when checkpoints already exist from a previous training run and
    you want to (re-)compute fire-type-grouped metrics without retraining.
    """
    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    )
    if not run_dirs:
        print(f"No checkpoints found in {out_dir}")
        return

    print(f"\nEval: {len(run_dirs)} checkpoints in {out_dir}\n")

    all_results = []
    for i, run_dir in enumerate(run_dirs, 1):
        tag = run_dir.name
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(run_dirs)}]  {tag}")
        print(f"{'=' * 60}")

        ckpt = BestCheckpoint(run_dir).load(map_location="cpu")
        cfg = ckpt["cfg"]

        ft_result = eval_by_fire_type(
            str(run_dir / "best.pt"), cfg, dataset_dir=dataset_dir)
        ft_result["tag"] = tag
        ft_result["train_fire_type"] = cfg.fire_type
        all_results.append(ft_result)

    # Save combined results
    path = out_dir / "fire_type_results.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFire type results → {path}")

    _print_summary_table(all_results)


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def save_summary(rows: list[dict], out_dir: Path):
    """Collect per-run results, aggregate multi-run, and print summary."""
    all_results = []
    for row in rows:
        tag = row.get("tag", "")
        run_dir = Path(row.get("checkpoint", "")).parent if row.get("checkpoint") else None

        # Try to load fire_type_eval.json from run dir, fall back to in-memory
        ft_result: dict = {}
        if run_dir and (run_dir / "fire_type_eval.json").exists():
            with open(run_dir / "fire_type_eval.json") as f:
                ft_result = json.load(f)
        if not ft_result:
            ft_result = row.get("by_fire_type_eval", {})
        if not ft_result:
            print(f"  SKIP {tag} — no fire type eval results")
            continue

        ft_result["tag"] = tag
        ft_result["train_fire_type"] = row.get("fire_type", "")
        ft_result["val_f1"] = row.get("val_f1", 0)
        all_results.append(ft_result)

    # Save per-run results
    per_run_path = out_dir / "fire_type_per_run.json"
    with open(per_run_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPer-run results → {per_run_path}")

    # Aggregate multi-run results (group by train_fire_type)
    aggregated = _aggregate_runs(all_results)

    path = out_dir / "fire_type_results.json"
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results → {path}")

    _print_summary_table(aggregated)


def collect(out_dir: Path):
    """Collect results from all run dirs after split-job training.

    Scans ``out_dir`` for sub-directories with ``fire_type_eval.json``,
    loads per-run results, aggregates with mean ± std, and writes the
    final ``fire_type_results.json``.

    Use after ``--num-jobs`` SLURM training to merge results from
    multiple array tasks.
    """
    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "fire_type_eval.json").exists()
    )
    if not run_dirs:
        print(f"No fire_type_eval.json found in {out_dir}")
        return

    all_results = []
    for run_dir in run_dirs:
        tag = run_dir.name
        with open(run_dir / "fire_type_eval.json") as f:
            ft_result = json.load(f)
        # Read config to get train fire_type
        ckpt_path = run_dir / "best.pt"
        if ckpt_path.exists():
            ckpt = BestCheckpoint(run_dir).load(map_location="cpu")
            ft_result["train_fire_type"] = ckpt["cfg"].fire_type
        ft_result["tag"] = tag
        all_results.append(ft_result)

    print(f"Collected {len(all_results)} runs from {out_dir}")

    # Save per-run
    per_run_path = out_dir / "fire_type_per_run.json"
    with open(per_run_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Per-run results → {per_run_path}")

    # Aggregate
    aggregated = _aggregate_runs(all_results)
    path = out_dir / "fire_type_results.json"
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results → {path}")

    _print_summary_table(aggregated)


def _aggregate_runs(results: list[dict]) -> list[dict]:
    """Group per-run results by train_fire_type and compute mean ± std.

    If only one run exists per fire_type, returns results as-is (no std).
    For multi-run, returns one entry per fire_type with mean values and
    ``*_std`` keys for every numeric metric.
    """
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        ft = r.get("train_fire_type", "")
        groups[ft].append(r)

    # If all groups have exactly 1 run, no aggregation needed
    if all(len(runs) == 1 for runs in groups.values()):
        return results

    aggregated = []
    for ft, runs in groups.items():
        n = len(runs)
        if n == 1:
            aggregated.append(runs[0])
            continue

        # Build aggregated entry
        agg: dict = {
            "tag": f"ft_{ft}",
            "train_fire_type": ft,
            "n_runs": n,
        }

        # Aggregate global metrics
        agg["global"] = _mean_std_dict(
            [r.get("global", {}) for r in runs])

        # Aggregate per-fire-type metrics
        all_ft_keys = set()
        for r in runs:
            all_ft_keys |= set(r.get("by_fire_type", {}).keys())
        agg_by_ft: dict[str, dict] = {}
        for ft_key in all_ft_keys:
            ft_dicts = [r.get("by_fire_type", {}).get(ft_key, {})
                        for r in runs]
            agg_by_ft[ft_key] = _mean_std_dict(ft_dicts)
        agg["by_fire_type"] = agg_by_ft

        aggregated.append(agg)

    return aggregated


def _mean_std_dict(dicts: list[dict]) -> dict:
    """Compute mean and std for each numeric key across a list of dicts.

    For key ``"f1"`` with values [0.71, 0.73, 0.72], produces::

        {"f1": 0.72, "f1_std": 0.0082, "n": 100}

    Non-numeric keys (like ``"n"``) use the value from the first dict.
    """
    import numpy as _np

    if not dicts or not any(dicts):
        return {}

    result: dict = {}
    all_keys = set()
    for d in dicts:
        all_keys |= set(d.keys())

    for key in sorted(all_keys):
        vals = [d.get(key) for d in dicts if d.get(key) is not None]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)):
            arr = _np.array(vals, dtype=_np.float64)
            result[key] = round(float(arr.mean()), 4)
            if len(arr) > 1:
                result[f"{key}_std"] = round(float(arr.std()), 4)
        else:
            result[key] = vals[0]  # non-numeric: take first

    return result


def _print_summary_table(results: list[dict]):
    """Print a readable train_fire_type × eval_fire_type table."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("Fire Type Study — Summary")
    print("=" * 80)
    print(f"\n{'Train data':<16s} | {'Metric':<10s} | "
          f"{'global':>8s} | {'vegetation':>10s} | {'static':>8s} | {'crop':>8s}")
    print("-" * 80)

    for r in results:
        tag = r.get("tag", "?")
        g = r.get("global", {})
        by_ft = r.get("by_fire_type", {})

        # next_mask F1
        def _f1(ft_name):
            return by_ft.get(ft_name, {}).get("f1", "")

        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else f"{v:>8s}"

        print(f"{tag:<16s} | {'nm F1':<10s} | "
              f"{_fmt(g.get('f1', '')):>8s} | "
              f"{_fmt(_f1('vegetation')):>10s} | "
              f"{_fmt(_f1('static')):>8s} | "
              f"{_fmt(_f1('crop')):>8s}")

        # nf_equiv F1
        def _nf(ft_name):
            return by_ft.get(ft_name, {}).get("nf_equiv_f1", "")

        print(f"{'':16s} | {'nf F1':<10s} | "
              f"{_fmt(g.get('nf_equiv_f1', '')):>8s} | "
              f"{_fmt(_nf('vegetation')):>10s} | "
              f"{_fmt(_nf('static')):>8s} | "
              f"{_fmt(_nf('crop')):>8s}")
        print("-" * 80)


# ---------------------------------------------------------------------------
# Subprocess entry point — train + fire-type eval in one process
# ---------------------------------------------------------------------------


def _train_and_eval(cfg: NextDayConfig):
    """Train on configured fire_type, then eval grouped by all fire types.

    Called by GpuQueue as a subprocess.  Writes fire_type_eval.json alongside
    the training result.json.
    """
    from firecomp.next_day.implementations.dl_2d import train as dl2d_train

    result = dl2d_train(cfg)

    if result.checkpoint_path and result.checkpoint_path.exists():
        print("\n--- Fire type evaluation ---")
        ft_result = eval_by_fire_type(str(result.checkpoint_path), cfg)

        # Save alongside checkpoint
        ft_path = result.checkpoint_path.parent / "fire_type_eval.json"
        with open(ft_path, "w") as f:
            json.dump(ft_result, f, indent=2)
        print(f"\nFire type eval → {ft_path}")

        # Also append to result.json for GpuQueue collection
        result_path = result.checkpoint_path.parent / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            data["by_fire_type_eval"] = ft_result
            data["fire_type"] = cfg.fire_type
            with open(result_path, "w") as f:
                json.dump(data, f, indent=2)


def _cli_main():
    """Cli dispatcher for GpuQueue subprocesses (train + eval)."""
    cli = Cli(
        NextDayConfig,
        prog="next_day.fire_type_study",
        description="Fire type study: train + fire-type-grouped eval.",
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
        prog="fire_type_study",
        description="Study how fire type composition affects prediction quality.",
    )
    p.add_argument(
        "command", nargs="?", default="run",
        choices=["run", "eval", "collect"],
        help="'run' to train + eval (default), 'eval' to re-evaluate existing "
             "checkpoints, 'collect' to merge split-job results.",
    )
    p.add_argument("--config", default=None,
                   help="JSON config file with base model settings.")
    p.add_argument("--dataset-dir", default=None,
                   help="Dataset directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print configs without training.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Batches per epoch; 0=full, 1=smoke test.")
    p.add_argument("--out-dir", default="data/runs/fire_type",
                   help="Output directory.")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs for dispatch (0=auto-detect).")
    p.add_argument("--num-runs", type=int, default=3,
                   help="Runs per fire-type config for mean±std (default: 3).")
    p.add_argument("--num-jobs", "-jn", type=int, default=1,
                   help="Total SLURM array jobs (splits configs across jobs).")
    p.add_argument("--job-index", "-ji", type=int, default=0,
                   help="This job's index (0-based).")
    return p.parse_args()


def _print_configs(configs: list[NextDayConfig]):
    """Print configs for --dry-run."""
    print(f"\n{len(configs)} configs:\n")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.tag:<20s}  fire_type={cfg.fire_type:<12s}  "
              f"target={cfg.target_type}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _cli_main()
    else:
        main()
