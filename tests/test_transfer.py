"""
Smoke test for the cross-region transfer experiments.

Verifies the full pipeline — region-filtered training, leave-one-out,
matrix aggregation, and heatmap rendering — runs end-to-end without
crashing.  Not a quality test: only 100 fires, 3 epochs, max_batches=1.

Similar in structure to test_train_smoke.py and test_grid_search.py.

Run:  pytest tests/test_transfer.py -v -s
"""

import json
import tempfile
from pathlib import Path

import pytest

DATASET_DIR = Path("data/tasks/next_day/100/")
H5_FILE = DATASET_DIR / "v2" / "dataset_0.h5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg(**overrides):
    """Minimal NextDayConfig for fast smoke runs."""
    from firecomp.next_day.config import NextDayConfig
    defaults = dict(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        device="cuda",
        num_workers=2,
        batch_size=8,
        num_epochs=3,
        patience=0,
        include_pos_encoding=False,
        target_type="new_fires",
        tag="transfer_test",
    )
    defaults.update(overrides)
    return NextDayConfig(**defaults)


# ---------------------------------------------------------------------------
# Unit tests — no GPU, no H5 dataset needed
# ---------------------------------------------------------------------------

def test_config_accepts_region_fields():
    """NextDayConfig stores train_regions and exclude_regions cleanly."""
    from firecomp.next_day.config import NextDayConfig

    cfg = NextDayConfig(train_regions=["California"], exclude_regions=None)
    assert cfg.train_regions == ["California"]
    assert cfg.exclude_regions is None

    cfg2 = NextDayConfig(exclude_regions=["Africa"])
    assert cfg2.exclude_regions == ["Africa"]
    assert cfg2.train_regions is None

    # None defaults
    cfg3 = NextDayConfig()
    assert cfg3.train_regions is None
    assert cfg3.exclude_regions is None


def test_regions_filter_with_prebuilt_mapping():
    """Regions.filter correctly includes/excludes samples given a fire_regions dict."""
    from firecomp.core.regions import Regions
    from firecomp.next_day.dataset import Sample

    regions = Regions()
    all_names = regions.names()
    if len(all_names) < 2:
        pytest.skip("Need at least 2 region names in wildfire_regions.json")

    r1, r2 = all_names[0], all_names[1]
    id1 = regions.name_to_id(r1)
    id2 = regions.name_to_id(r2)

    # Three mock samples: fire 1 → r1, fire 2 → r2, fire 3 → r1
    def _s(fire_id):
        return Sample(fire_id=fire_id, xi=0, yi=0, lon=0.0, lat=0.0,
                      dt="2020-01-01", img_size=256)

    samples = [_s(1), _s(2), _s(3)]
    fire_regions = {1: id1, 2: id2, 3: id1}

    # include r1 → fires 1 and 3
    keep = Regions.filter(samples, include=[r1], fire_regions=fire_regions)
    assert len(keep) == 2
    assert {s.fire_id for s in keep} == {1, 3}

    # exclude r1 → only fire 2
    drop = Regions.filter(samples, exclude=[r1], fire_regions=fire_regions)
    assert len(drop) == 1
    assert drop[0].fire_id == 2

    # neither include nor exclude → identity
    all_s = Regions.filter(samples, fire_regions=fire_regions)
    assert len(all_s) == 3


def test_build_transfer_matrix():
    """build_transfer_matrix aggregates TransferResults into a nested dict."""
    from firecomp.next_day.implementations.transfer import (
        TransferResult,
        build_transfer_matrix,
    )
    from firecomp.core.regions import Regions

    regions = Regions()
    results = [
        TransferResult(
            train_label="RegionA",
            train_regions=["RegionA"],
            exclude_regions=None,
            target_type="new_fires",
            by_region={"RegionA": 0.80, "RegionB": 0.30},
            nf_by_region=None,
            checkpoint="",
            best_f1=0.80,
        ),
        TransferResult(
            train_label="global",
            train_regions=None,
            exclude_regions=None,
            target_type="new_fires",
            by_region={"RegionA": 0.70, "RegionB": 0.60},
            nf_by_region=None,
            checkpoint="",
            best_f1=0.70,
        ),
    ]

    matrix = build_transfer_matrix(results, regions)
    assert set(matrix.keys()) == {"RegionA", "global"}
    assert matrix["RegionA"]["RegionA"] == 0.80
    assert matrix["RegionA"]["RegionB"] == 0.30
    assert matrix["global"]["RegionB"] == 0.60


def test_save_matrix_writes_json_and_csv():
    """save_matrix writes readable JSON and correct-shape CSV."""
    from firecomp.next_day.implementations.transfer import save_matrix

    matrix = {
        "region_a": {"X": 0.50, "Y": 0.30},
        "region_b": {"X": 0.40, "Y": 0.75},
        "global":   {"X": 0.65, "Y": 0.60},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = save_matrix(matrix, tmpdir)

        assert json_path.exists(), "transfer_matrix.json not created"
        csv_path = json_path.parent / "transfer_matrix.csv"
        assert csv_path.exists(), "transfer_matrix.csv not created"

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded == matrix

        lines = csv_path.read_text().splitlines()
        assert len(lines) == 4              # header + 3 rows
        assert "X" in lines[0] and "Y" in lines[0]


def test_plot_transfer_matrix_creates_png():
    """plot_transfer_matrix renders a PNG for a mock matrix."""
    from firecomp.next_day.implementations.transfer import plot_transfer_matrix

    matrix = {
        "RegionA": {"RegionA": 0.80, "RegionB": 0.30, "RegionC": 0.20},
        "RegionB": {"RegionA": 0.40, "RegionB": 0.75, "RegionC": 0.35},
        "global":  {"RegionA": 0.65, "RegionB": 0.60, "RegionC": 0.55},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = plot_transfer_matrix(matrix, tmpdir)
        assert fig_path.exists(), "transfer_matrix.png not created"
        assert fig_path.stat().st_size > 1000, "PNG looks empty"


# ---------------------------------------------------------------------------
# Dataset-level filter test — requires H5 + region raster, no GPU
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not H5_FILE.exists(), reason="100-fire dataset not available")
def test_region_filter_actually_reduces_sample_count():
    """Verify that train_regions actually filters — not a silent no-op.

    Loads the 100-fire dataset twice: once unfiltered, once with the largest
    region as train_regions.  The filtered count MUST be strictly less than
    the unfiltered count.
    """
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.dataset import NextDayDataset
    from firecomp.core.regions import Regions

    base_kw = dict(
        dataset_dir=str(DATASET_DIR), dataset_version="v2",
        include_pos_encoding=False, target_type="new_fires", device="cpu",
    )

    ds_full = NextDayDataset(NextDayConfig(**base_kw))
    total_train = len(ds_full.train_samples)

    fire_regions = Regions.build_fire_regions(ds_full.train_samples)
    assert fire_regions is not None, (
        "Region raster could not be loaded — filtering cannot be verified"
    )

    # Find the largest region
    regions = Regions()
    counts = {}
    for s in ds_full.train_samples:
        rid = fire_regions.get(s.fire_id, 0)
        if rid != 0:
            name = regions.id_to_name(rid)
            counts[name] = counts.get(name, 0) + 1
    assert counts, "No named regions found in training samples"

    target = max(counts, key=counts.get)
    expected = counts[target]

    ds_filtered = NextDayDataset(NextDayConfig(train_regions=[target], **base_kw))
    filtered_train = len(ds_filtered.train_samples)

    print(f"\n  Unfiltered: {total_train}  |  {target}: {filtered_train}  (expected {expected})")

    assert filtered_train < total_train, (
        f"Filter did NOT reduce count: {filtered_train} == {total_train}. "
        f"Region raster may not be loading correctly."
    )
    assert filtered_train == expected, (
        f"Filtered count {filtered_train} != expected {expected} for {target}"
    )


# ---------------------------------------------------------------------------
# Integration tests — require GPU + 100-fire dataset
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(
    not H5_FILE.exists(),
    reason="Desktop 100-fire dataset not available",
)


def _find_region_with_most_train_samples(ds):
    """Return (region_name, count) for the region with the most training samples."""
    from firecomp.core.regions import Regions

    fire_regions = Regions.build_fire_regions(ds.train_samples)
    if fire_regions is None:
        return None, 0

    regions = Regions()
    counts = {}
    for s in ds.train_samples:
        rid = fire_regions.get(s.fire_id, 0)
        if rid != 0:
            name = regions.id_to_name(rid)
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        return None, 0
    best = max(counts, key=counts.get)
    return best, counts[best]


@pytestmark_gpu
def test_region_filtered_training_end_to_end():
    """Train on one region via GpuQueue, eval on all: 1 batch/epoch × 3 epochs."""
    from dataclasses import replace
    from firecomp.core.gpu_queue import GpuQueue
    from firecomp.next_day.dataset import NextDayDataset
    from firecomp.next_day.implementations.transfer import eval_by_region, TRAIN_MODULE

    base = _base_cfg()
    ds = NextDayDataset(base)

    target_region, count = _find_region_with_most_train_samples(ds)
    if target_region is None:
        pytest.skip("Region raster not available or no named regions in dataset")
    del ds

    print(f"\n  Target region: {target_region} ({count} train samples)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tag = f"smoke_region_{target_region.lower().replace(' ', '_')}"
        cfg = _base_cfg(
            train_regions=[target_region],
            tag=tag,
            run_dir=str(Path(tmpdir) / tag),
            max_batches=1,
        )

        rows = GpuQueue.run(
            [cfg], out_dir=Path(tmpdir), train_module=TRAIN_MODULE,
            num_gpus=1, label="Test region-filtered",
        )

        assert len(rows) == 1
        row = rows[0]
        assert row.get("checkpoint"), f"No checkpoint: {row}"

        by_region = eval_by_region(row["checkpoint"], cfg)

    assert isinstance(by_region, dict)
    assert len(by_region) >= 1, "Eval must return at least one region score"


@pytestmark_gpu
def test_leave_one_out_training_end_to_end():
    """Exclude smallest region, train via GpuQueue: 1 batch/epoch × 3 epochs."""
    from dataclasses import replace
    from firecomp.core.gpu_queue import GpuQueue
    from firecomp.next_day.dataset import NextDayDataset
    from firecomp.core.regions import Regions
    from firecomp.next_day.implementations.transfer import eval_by_region, TRAIN_MODULE

    base = _base_cfg()
    ds = NextDayDataset(base)

    fire_regions = Regions.build_fire_regions(ds.train_samples)
    if fire_regions is None:
        pytest.skip("Region raster not available")

    regions = Regions()
    counts = {}
    for s in ds.train_samples:
        rid = fire_regions.get(s.fire_id, 0)
        if rid != 0:
            name = regions.id_to_name(rid)
            counts[name] = counts.get(name, 0) + 1
    del ds

    if not counts:
        pytest.skip("No named regions in training samples")

    # Exclude the smallest region — maximises remaining training data
    exclude = min(counts, key=counts.get)
    print(f"\n  Excluding: {exclude} ({counts[exclude]} samples)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tag = f"smoke_loo_{exclude.lower().replace(' ', '_')}"
        cfg = _base_cfg(
            exclude_regions=[exclude],
            tag=tag,
            run_dir=str(Path(tmpdir) / tag),
            max_batches=1,
        )

        rows = GpuQueue.run(
            [cfg], out_dir=Path(tmpdir), train_module=TRAIN_MODULE,
            num_gpus=1, label="Test LOO",
        )

        assert len(rows) == 1
        row = rows[0]
        assert row.get("checkpoint"), f"No checkpoint: {row}"

        by_region = eval_by_region(row["checkpoint"], cfg)

    assert isinstance(by_region, dict)
    assert len(by_region) >= 1


@pytestmark_gpu
def test_transfer_matrix_collection_and_save():
    """Multiple single-region runs → correct JSON/CSV transfer matrix."""
    from firecomp.next_day.dataset import NextDayDataset
    from firecomp.core.regions import Regions
    from firecomp.next_day.implementations.transfer import (
        TransferResult,
        build_transfer_matrix,
        save_matrix,
    )

    # Build fake results (no training needed for this aggregation test)
    regions = Regions()
    names = regions.names()[:3]

    results = [
        TransferResult(
            train_label=name,
            train_regions=[name],
            exclude_regions=None,
            target_type="new_fires",
            by_region={n: round(0.5 + 0.05 * i, 3) for i, n in enumerate(names)},
            checkpoint="fake.pt",
            best_f1=0.5,
        )
        for name in names
    ] + [
        TransferResult(
            train_label="global",
            train_regions=None,
            exclude_regions=None,
            target_type="new_fires",
            by_region={n: round(0.65 + 0.02 * i, 3) for i, n in enumerate(names)},
            checkpoint="fake.pt",
            best_f1=0.65,
        )
    ]

    matrix = build_transfer_matrix(results, regions)
    assert set(matrix.keys()) == set(names) | {"global"}

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = save_matrix(matrix, tmpdir)
        assert json_path.exists()
        csv_path = json_path.parent / "transfer_matrix.csv"
        assert csv_path.exists()

        with open(json_path) as f:
            loaded = json.load(f)

    assert set(loaded.keys()) == set(matrix.keys())
    for train_label, row in loaded.items():
        assert set(row.keys()) == set(names), (
            f"Row {train_label!r} has wrong test regions: {set(row.keys())}"
        )
