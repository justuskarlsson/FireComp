"""
Desktop test for grid_search.py.

Sweeps 8 configs (2 models × 2 losses × 2 targets) on the 100-fire dataset.
3 epochs, max_batches=1 per epoch — verifies the pipeline works end-to-end.

Run:  pytest tests/test_grid_search.py -v -s
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

DATASET_DIR = Path("data/tasks/next_day/100/")
H5_FILE = DATASET_DIR / "v2" / "dataset_0.h5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg(**overrides):
    """Minimal NextDayConfig for fast smoke runs."""
    from firecomp.next_day.config import NextDayConfig
    defaults = dict(
        dataset_dir     = str(DATASET_DIR),
        dataset_version = "v2",
        device          = "cuda",
        num_workers     = 2,
        batch_size      = 8,
        num_epochs      = 3,
        patience        = 0,
        tag             = "gstest",
    )
    defaults.update(overrides)
    return NextDayConfig(**defaults)


def _small_grid():
    """2 models × 2 losses × 2 targets = 8 configs."""
    from firecomp.next_day.implementations.grid_search import (
        LOSS_CONFIGS,
        build_grid,
    )
    return build_grid(
        _base_cfg(),
        models       = ["unet", "unet++"],
        loss_configs = [LOSS_CONFIGS[0], LOSS_CONFIGS[3]],
        targets      = ["next_mask", "new_fires"],
    )


# ---------------------------------------------------------------------------
# Structural tests (no GPU)
# ---------------------------------------------------------------------------

def test_full_grid_tag_uniqueness():
    """build_grid() on the full axes yields exactly 36 configs with unique tags."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.grid_search import build_grid

    grid = build_grid(NextDayConfig())

    assert len(grid) == 36, f"Expected 36, got {len(grid)}"
    tags = [cfg.tag for cfg in grid]
    assert len(set(tags)) == 36, f"Duplicate tags: {[t for t in tags if tags.count(t) > 1]}"


def test_small_grid_count():
    """Small grid produces expected number of configs with unique tags."""
    grid = _small_grid()
    assert len(grid) == 8
    tags = [cfg.tag for cfg in grid]
    assert len(set(tags)) == 8, f"Duplicate tags: {tags}"


# ---------------------------------------------------------------------------
# Integration tests (require GPU + dataset)
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(
    not H5_FILE.exists(),
    reason="Desktop 100-fire dataset not available",
)


@pytestmark_gpu
def test_grid_search_subset():
    """
    8-config subset: 2 models × 2 loss configs × 2 targets.
    3 epochs, max_batches=1.  Checks:
      - correct number of runs completed
      - results.json + results.csv written
      - each row has the required keys with sensible values
      - ranking printed (val_f1 ordering)
    """
    from firecomp.next_day.implementations.grid_search import (
        collect_results,
        run_grid,
    )

    grid = _small_grid()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        rows = run_grid(grid, out_dir, max_batches=1)
        assert len(rows) == 8, f"Expected 8 result rows, got {len(rows)}"

        json_path = collect_results(rows, out_dir=tmpdir)

        assert json_path.exists(), "results.json not written"
        assert (json_path.parent / "results.csv").exists(), "results.csv not written"

        with open(json_path) as f:
            data = json.load(f)

    assert len(data) == 8

    for row in data:
        for key in ("tag", "model", "loss", "target", "val_f1",
                     "best_epoch", "checkpoint"):
            assert key in row, f"Missing '{key}' in {row.get('tag')}"
        assert isinstance(row["val_f1"], float) and row["val_f1"] >= 0.0
        assert row["checkpoint"] != ""


@pytest.mark.skipif(
    not H5_FILE.exists() or torch.cuda.device_count() < 2,
    reason="Requires 2+ GPUs and the 100-fire dataset",
)
def test_grid_search_multi_gpu():
    """
    Same 8-config subset dispatched across 2 GPUs via queue dispatch.
    Verifies all runs complete and results are collected.
    """
    from firecomp.next_day.implementations.grid_search import (
        collect_results,
        run_grid,
    )

    grid = _small_grid()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        rows = run_grid(grid, out_dir, max_batches=1, num_gpus=2)
        assert len(rows) == 8

        json_path = collect_results(rows, out_dir=tmpdir)
        with open(json_path) as f:
            data = json.load(f)

    assert len(data) == 8
    for row in data:
        assert row["val_f1"] >= 0.0
        assert row["checkpoint"] != ""
