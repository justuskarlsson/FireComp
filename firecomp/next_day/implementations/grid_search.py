"""
next_day/implementations/grid_search.py — Paper 1 grid search.

Runs the 36-configuration Cartesian grid (3 models × 6 loss configs × 2 targets)
for the Paper 1 next-day fire spread benchmark.

Usage:
    # full 36-run grid, single GPU
    python -m firecomp.next_day.implementations.grid_search

    # queue-dispatch across 4 GPUs
    python -m firecomp.next_day.implementations.grid_search --num-gpus 4

    # split into 4 independent SLURM jobs (1 GPU each)
    python -m ... --num-jobs 4 --job-index 0   # job 0: configs 0-8
    python -m ... --num-jobs 4 --job-index 1   # job 1: configs 9-17
    python -m ... --num-jobs 4 --job-index 2   # job 2: configs 18-26
    python -m ... --num-jobs 4 --job-index 3   # job 3: configs 27-35

    # collect results from split jobs into one summary
    python -m firecomp.next_day.implementations.grid_search collect

    # print all 36 configs without training
    python -m firecomp.next_day.implementations.grid_search --dry-run

    # evaluate all checkpoints from a previous grid on the test split
    python -m firecomp.next_day.implementations.grid_search eval --out-dir data/runs/grid

Grid axes
---------
  models   : unet, unet++, vit                                (3)
  loss     : bce x {pw=1,5,10},  focal x {a=0.25,0.50,0.75}  (6)
  targets  : next_mask, new_fires                             (2)
  total    : 36 runs

Each run is launched as a separate subprocess via
``python -m firecomp.next_day.implementations.dl_2d train --config <json>``,
with ``CUDA_VISIBLE_DEVICES`` pinning it to one GPU.  A simple queue
keeps all GPUs busy: when a subprocess finishes, the freed GPU picks up
the next config from the queue.

Stdout/stderr per run goes to ``<out-dir>/<tag>.log``.  After all runs,
results are collected from each run's ``result.json`` into a summary
JSON + CSV under ``--out-dir``.
"""

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

from firecomp.core.gpu_queue import GpuQueue
from firecomp.next_day.config import NextDayConfig

TRAIN_MODULE = "firecomp.next_day.implementations.dl_2d"


# ---------------------------------------------------------------------------
# Loss-parameter combinations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossConfig:
    """One (loss_type, param) pair for the grid."""
    name:        str    # short label used in run tags and filenames
    loss_type:   str    # "bce" | "focal"
    pos_weight:  float  # class weight for BCE (ignored by focal)
    focal_alpha: float  # class weight for focal (ignored by BCE)
    focal_gamma: float = 2.0   # fixed at 2.0 (Lin et al. optimal)


LOSS_CONFIGS: list[LossConfig] = [
    # BCE — vary pos_weight
    LossConfig("bce_pw1",   "bce",   pos_weight=1.0,  focal_alpha=0.25),
    LossConfig("bce_pw5",   "bce",   pos_weight=5.0,  focal_alpha=0.25),
    LossConfig("bce_pw10",  "bce",   pos_weight=10.0, focal_alpha=0.25),
    # Focal — vary alpha, gamma fixed at 2.0
    LossConfig("focal_a25", "focal", pos_weight=1.0,  focal_alpha=0.25),
    LossConfig("focal_a50", "focal", pos_weight=1.0,  focal_alpha=0.50),
    LossConfig("focal_a75", "focal", pos_weight=1.0,  focal_alpha=0.75),
]

MODELS:  list[str] = ["unet", "unet++", "vit"]
TARGETS: list[str] = ["next_mask", "new_fires"]


def _split_list(items: list, num_jobs: int, job_index: int) -> list:
    """Split *items* into *num_jobs* contiguous chunks, return chunk *job_index*.

    First ``len(items) % num_jobs`` jobs get one extra item.
    """
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

    if args.command == "eval":
        eval_grid(Path(args.out_dir), dataset_dir=args.dataset_dir,
                  num_jobs=args.num_jobs, job_index=args.job_index,
                  split=args.split)
        return

    if args.command == "eval-collect":
        eval_collect(Path(args.out_dir))
        return

    if args.command == "collect":
        collect_from_dirs(Path(args.out_dir))
        return

    num_gpus = GpuQueue.resolve_num_gpus(args.num_gpus)
    base     = NextDayConfig()

    # Stage dataset to /dev/shm before building grid so all configs use the
    # fast local path.
    if not args.no_cache:
        local_dir = _stage_dataset(base.dataset_dir, base.dataset_version)
        if local_dir:
            base = replace(base, dataset_dir=local_dir)

    grid = build_grid(base)

    # Job splitting for SLURM arrays: --num-jobs N --job-index I
    # Each job trains a contiguous slice of the grid.
    if args.num_jobs > 1:
        grid = _split_list(grid, args.num_jobs, args.job_index)
        print(f"[job {args.job_index}/{args.num_jobs}] "
              f"{len(grid)} configs")

    if args.dry_run:
        _print_grid(grid)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_grid(grid, out_dir, num_gpus=num_gpus, max_batches=args.max_batches)
    collect_results(rows, out_dir=str(out_dir))


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------

def build_grid(
    base: NextDayConfig,
    models:       list[str]        | None = None,
    loss_configs: list[LossConfig] | None = None,
    targets:      list[str]        | None = None,
) -> list[NextDayConfig]:
    """
    Build the Cartesian grid of configs.

    Pass subset lists to run a smaller grid (e.g. for tests).
    Defaults to the full 36-config paper grid.

    Returns one NextDayConfig per cell; each has a descriptive tag
    and the checkpoint will be saved to data/runs/next_day_<tag>/.
    """
    models       = models       or MODELS
    loss_configs = loss_configs or LOSS_CONFIGS
    targets      = targets      or TARGETS

    grid = []
    for model in models:
        for loss in loss_configs:
            for target in targets:
                tag = _run_tag(model, loss, target)
                cfg = replace(
                    base,
                    model_type   = model,
                    loss_type    = loss.loss_type,
                    pos_weight   = loss.pos_weight,
                    focal_alpha  = loss.focal_alpha,
                    focal_gamma  = loss.focal_gamma,
                    target_type  = target,
                    tag          = tag,
                )
                grid.append(cfg)
    return grid


def _run_tag(model: str, loss: LossConfig, target: str) -> str:
    """Descriptive run tag: grid_unetpp_bce_pw10_newf."""
    model_slug  = model.replace("++", "pp")
    target_slug = "newf" if target == "new_fires" else "next"
    return f"grid_{model_slug}_{loss.name}_{target_slug}"


# ---------------------------------------------------------------------------
# Grid execution — subprocess queue
# ---------------------------------------------------------------------------

def run_grid(
    grid: list[NextDayConfig],
    out_dir: Path,
    num_gpus: int = 1,
    max_batches: int = 0,
) -> list[dict]:
    """
    Run all configs as subprocesses, queue-dispatched across GPUs.

    Thin wrapper around GpuQueue.run() that assigns run_dir and
    max_batches to each config before dispatch.

    Args:
        grid:        list of NextDayConfig to run.
        out_dir:     directory for per-run subdirs, logs, and summary.
        num_gpus:    number of GPUs available.
        max_batches: if >0, cap each epoch (smoke test).
    """
    configs = [
        replace(cfg,
                run_dir=str(out_dir / cfg.tag),
                max_batches=max_batches)
        for cfg in grid
    ]
    return GpuQueue.run(
        configs,
        out_dir=out_dir,
        train_module=TRAIN_MODULE,
        num_gpus=num_gpus,
        label="Grid search",
    )


# ---------------------------------------------------------------------------
# Grid evaluation — re-evaluate checkpoints on the test split
# ---------------------------------------------------------------------------


def eval_grid(out_dir: Path, dataset_dir: str | None = None,
              num_jobs: int = 1, job_index: int = 0,
              split: str = "val"):
    """Re-evaluate all grid checkpoints on a dataset split.

    For each run directory under ``out_dir`` that contains a ``best.pt``,
    loads the checkpoint and evaluates via dl_2d.eval.

    For ``next_mask`` models, also computes "new-fires equivalent" metrics
    (masks out already-burning pixels) to enable fair comparison with
    ``new_fires`` models.

    Uses the **validation** split by default so that grid comparison is
    model-selection (not reporting on the held-out test set).

    Results are saved to ``eval_results.json`` (or ``eval_results_<ji>.json``
    when job-splitting) alongside the training results.

    Args:
        out_dir:     grid search output directory (contains per-run subdirs).
        dataset_dir: override the checkpoint's dataset_dir.  Use when the
                     training-time path (e.g. /dev/shm staging) no longer
                     exists.
        num_jobs:    split eval across N independent SLURM jobs.
        job_index:   this job's index (0-based).
        split:       dataset split — ``"val"`` (default) or ``"test"``.
    """
    from firecomp.next_day.implementations.dl_2d import eval as dl2d_eval

    # Stage dataset to /dev/shm (same as training) — eval reads the full
    # test split so fast I/O matters.
    base = NextDayConfig()
    staged = _stage_dataset(
        dataset_dir or base.dataset_dir,
        base.dataset_version,
    )
    if staged:
        dataset_dir = staged

    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    )
    if not run_dirs:
        print(f"No checkpoints found in {out_dir}")
        return

    # Job splitting: contiguous slice of run_dirs
    if num_jobs > 1:
        run_dirs = _split_list(run_dirs, num_jobs, job_index)
        print(f"[eval job {job_index}/{num_jobs}] {len(run_dirs)} checkpoints")

    print(f"\nEval: {len(run_dirs)} checkpoints in {out_dir}\n")

    results = []
    for i, run_dir in enumerate(run_dirs, 1):
        tag = run_dir.name
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(run_dirs)}]  {tag}")
        print(f"{'='*60}")

        eval_result = dl2d_eval(
            str(run_dir / "best.pt"),
            dataset_dir=dataset_dir,
            split=split,
        )
        results.append(eval_result)

    if num_jobs > 1:
        # Per-job partial results — merged later by eval-collect
        json_path = out_dir / f"eval_results_{job_index}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPartial eval results → {json_path}  ({len(results)} runs)")
    else:
        _save_eval_results(results, out_dir)


def eval_collect(out_dir: Path):
    """Merge per-job eval_results_*.json into eval_results.json."""
    partials = sorted(out_dir.glob("eval_results_*.json"))
    if not partials:
        print(f"No eval_results_*.json found in {out_dir}")
        return

    results = []
    for p in partials:
        with open(p) as f:
            results.extend(json.load(f))

    print(f"Collected {len(results)} eval results from {len(partials)} partial files")
    _save_eval_results(results, out_dir)


def _save_eval_results(results: list[dict], out_dir: Path):
    """Write eval results as JSON + CSV and print ranking."""
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV — derive columns from the data (skip nested dicts like by_region)
    csv_path = out_dir / "eval_results.csv"
    scalar_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k, v in r.items():
            if k not in seen and not isinstance(v, (dict, list)):
                scalar_keys.append(k)
                seen.add(k)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEval results → {json_path}  ({len(results)} runs)")
    print(f"Eval CSV     → {csv_path}")

    # Print ranking by target type
    for target in ("next_mask", "new_fires"):
        subset = [r for r in results if r.get("target") == target]
        if not subset:
            continue
        ranked = sorted(subset, key=lambda r: r.get("test_f1", 0), reverse=True)
        print(f"\n--- {target}: ranked by test F1 ---")
        for i, r in enumerate(ranked, 1):
            line = (f"  {i:2d}. {r['tag']:<45s}  "
                    f"test_F1={r.get('test_f1', 0):.4f}  "
                    f"val_F1={r.get('val_f1', 0):.4f}")
            nf = r.get("nf_equiv_f1")
            if nf is not None:
                line += f"  nf_equiv={nf:.4f}"
            print(line)


# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------

def collect_from_dirs(out_dir: Path) -> Path:
    """Scan run subdirectories and collect all result.json into one summary.

    Use after split-job training (--num-jobs) to gather results written by
    each independent SLURM job.  Also works after a single-job run.

    Returns path to the combined results.json.
    """
    run_dirs = sorted(
        d for d in out_dir.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    )
    if not run_dirs:
        print(f"No result.json found in {out_dir}/*/")
        return out_dir / "results.json"

    rows = []
    for d in run_dirs:
        with open(d / "result.json") as f:
            rows.append(json.load(f))

    print(f"Collected {len(rows)} results from {out_dir}/*/result.json")
    return collect_results(rows, out_dir=str(out_dir))


def collect_results(rows: list[dict], out_dir: str = "data/runs/grid") -> Path:
    """
    Write summary JSON + CSV; print ranking; return the JSON path.

    Files written:
        <out_dir>/results.json   — full results, one object per run
        <out_dir>/results.csv    — same data in tabular form
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "results.json"
    csv_path  = out / "results.csv"

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSummary -> {json_path}  ({len(rows)} runs)")
    _print_ranking(rows)
    return json_path


def _print_ranking(rows: list[dict]):
    """Print all runs ranked by val F1."""
    if not rows:
        return
    ranked = sorted(rows, key=lambda r: r.get("val_f1", 0), reverse=True)
    print("\n--- Configs ranked by val F1 ---")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:2d}. {r['tag']:<50s}  "
              f"val_F1={r.get('val_f1', 0):.4f}  epoch={r.get('best_epoch', -1)}  "
              f"checkpoint={r.get('checkpoint', '')}")


# ---------------------------------------------------------------------------
# Dataset staging to /dev/shm
# ---------------------------------------------------------------------------

SHM_ROOT = Path("/dev/shm")


def _stage_dataset(dataset_dir: str, dataset_version: str) -> str | None:
    """Copy the full dataset directory to /dev/shm for fast I/O.

    Copies the entire ``dataset_dir`` tree (e.g. ``data/next_day_v3/``)
    to ``/dev/shm/<user>/firecomp/<basename>/``.  ``follow_symlinks=True``
    resolves any symlinks into real copies in RAM.

    Returns the new ``dataset_dir``, or None if /dev/shm is not available.
    """
    if not SHM_ROOT.is_dir():
        print("[cache] /dev/shm not available, reading from network drive")
        return None

    src = Path(dataset_dir)
    if not src.is_dir():
        print(f"[cache] source {src} not found, skipping staging")
        return None

    user = os.environ.get("USER", "unknown")
    ds_name = src.name                            # e.g. "10K"
    local_dst = SHM_ROOT / user / "firecomp" / ds_name

    # Already staged if the version subdir exists.
    if (local_dst / dataset_version).is_dir():
        print(f"[cache] already staged: {local_dst}")
        return str(local_dst)

    print(f"[cache] staging {src} to {local_dst} ...")
    t0 = time.monotonic()
    shutil.copytree(src, local_dst, dirs_exist_ok=True,
                    copy_function=shutil.copy2)
    elapsed = time.monotonic() - t0

    total = sum(f.stat().st_size for f in local_dst.rglob("*") if f.is_file())
    speed = total / 1e9 / elapsed if elapsed > 0 else 0
    print(f"[cache] done — {total / 1e9:.1f} GB in {elapsed:.0f}s ({speed:.1f} GB/s)")

    return str(local_dst)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "grid_search",
        description = "36-config grid search for the Paper 1 next-day benchmark.",
    )
    p.add_argument(
        "command", nargs="?", default="run",
        choices=["run", "eval", "eval-collect", "collect"],
        help="'run' to train (default), 'eval' to re-evaluate checkpoints "
             "on the test split (supports --num-jobs/--job-index), "
             "'eval-collect' to merge split eval results, "
             "'collect' to gather split-job training results.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print all configs without running any training.",
    )
    p.add_argument(
        "--out-dir", default="data/runs/grid",
        help="Directory for summary JSON/CSV (default: data/runs/grid).",
    )
    p.add_argument(
        "--dataset-dir", default=None,
        help="Override dataset_dir from checkpoint (eval mode).  Use when "
             "the training-time path no longer exists.",
    )
    p.add_argument(
        "--max-batches", type=int, default=0,
        help="Batches per epoch; 0=full epoch, 1=smoke test.",
    )
    p.add_argument(
        "--num-gpus", type=int, default=1,
        help="GPUs for queue dispatch (default: 1 = one at a time, "
             "0 = auto-detect all available GPUs).",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Skip staging dataset to /dev/shm (read from network drive).",
    )
    p.add_argument(
        "--num-jobs", "-jn", type=int, default=1,
        help="Split grid across N independent jobs (use with --job-index).",
    )
    p.add_argument(
        "--job-index", "-ji", type=int, default=0,
        help="This job's index (0-based, use with --num-jobs).",
    )
    p.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Dataset split for eval — 'val' (default) for model selection, "
             "'test' for final held-out evaluation.",
    )
    return p.parse_args()


def _print_grid(grid: list[NextDayConfig]):
    """Print all configs for --dry-run inspection."""
    print(f"\n{len(grid)} configs:\n")
    for i, cfg in enumerate(grid, 1):
        print(f"  {i:2d}. {cfg.tag:<52s}"
              f"model={cfg.model_type:<8s}"
              f"loss={cfg.loss_type:<6s}"
              f"pw={cfg.pos_weight:<6.1f}"
              f"alpha={cfg.focal_alpha:.2f}  "
              f"target={cfg.target_type}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
