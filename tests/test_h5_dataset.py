"""
Integration tests for NextDayDataset with H5 backend.

Requires the 100-sample desktop dataset at data/tasks/next_day/100/.
Tests are skipped if the dataset is not available.

Run:  pytest tests/test_h5_dataset.py -v -s   (use -s to see prints)
"""

import json
from collections import Counter, defaultdict

import numpy as np
import pytest
import torch
from pathlib import Path


DATASET_DIR = Path("data/tasks/next_day/100")
H5_FILE = DATASET_DIR / "dataset_0.h5"

pytestmark = pytest.mark.skipif(
    not H5_FILE.exists(),
    reason="Desktop dataset not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds():
    """Load dataset once for the whole module."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.dataset import NextDayDataset

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        device="cpu",
        num_workers=0,
        batch_size=4,
    )
    return NextDayDataset(cfg)


@pytest.fixture(scope="module")
def ds_new_fires():
    """Dataset with new_fires target."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.dataset import NextDayDataset

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        device="cpu",
        num_workers=0,
        batch_size=4,
        target_type="new_fires",
    )
    return NextDayDataset(cfg)


# ---------------------------------------------------------------------------
# Tests: dataset construction
# ---------------------------------------------------------------------------

def test_dataset_loads(ds):
    """Dataset initializes and has samples in all splits."""
    assert len(ds.train_samples) > 0
    assert len(ds.val_samples) > 0
    assert len(ds.test_samples) > 0


def test_num_channels_positive(ds):
    assert ds.num_channels > 0


def test_input_groups_cover_all_channels(ds):
    """input_groups channel indices cover [0, num_channels) exactly."""
    all_indices = []
    for indices in ds.input_groups.values():
        all_indices.extend(indices)
    assert sorted(all_indices) == list(range(ds.num_channels))


# ---------------------------------------------------------------------------
# Tests: batch loading
# ---------------------------------------------------------------------------

def test_train_batch_shapes(ds):
    """Train loader produces batches with correct shapes."""
    batch = next(iter(ds.train()))
    B = batch.x.shape[0]
    assert batch.x.shape == (B, ds.num_channels, 256, 256)
    assert batch.y.shape == (B, 1, 256, 256)
    assert batch.loss_mask.shape == (B, 1, 256, 256)
    assert batch.idx.shape == (B,)
    assert len(batch.samples) == B


def test_batch_no_nan_inf(ds):
    """Input tensor has no NaN or Inf values."""
    batch = next(iter(ds.train()))
    assert not torch.isnan(batch.x).any(), "NaN in batch.x"
    assert not torch.isinf(batch.x).any(), "Inf in batch.x"


def test_target_binary(ds):
    """Target tensor is binary (0 or 1)."""
    batch = next(iter(ds.train()))
    unique = batch.y.unique()
    assert all(v in (0.0, 1.0) for v in unique.tolist())


def test_loss_mask_binary(ds):
    """Loss mask is binary (0 or 1)."""
    batch = next(iter(ds.train()))
    unique = batch.loss_mask.unique()
    assert all(v in (0.0, 1.0) for v in unique.tolist())


def test_new_fires_target(ds_new_fires):
    """new_fires target loads with correct shape."""
    batch = next(iter(ds_new_fires.train()))
    assert batch.y.shape == (batch.x.shape[0], 1, 256, 256)
    assert batch.x.shape[1] == ds_new_fires.num_channels


# ---------------------------------------------------------------------------
# Tests: val / test splits
# ---------------------------------------------------------------------------

def test_val_batch(ds):
    batch = next(iter(ds.val()))
    assert batch.x.shape[1] == ds.num_channels


def test_test_batch(ds):
    batch = next(iter(ds.test()))
    assert batch.x.shape[1] == ds.num_channels


# ---------------------------------------------------------------------------
# Tests: model forward pass on real data
# ---------------------------------------------------------------------------

def test_forward_pass_unet(ds):
    """UNet forward pass on real data produces finite output."""
    from firecomp.models.segmentation_models import model_factory

    batch = next(iter(ds.train()))
    model = model_factory["unet"](in_channels=ds.num_channels)
    model.eval()
    with torch.no_grad():
        pred = model(batch.x)
    assert pred.shape == (batch.x.shape[0], 1, 256, 256)
    assert torch.isfinite(pred).all()


def test_loss_computes(ds):
    """Loss computation on real data produces finite scalar."""
    from firecomp.core.losses import build_loss_fn
    from firecomp.models.segmentation_models import model_factory

    batch = next(iter(ds.train()))
    model = model_factory["unet"](in_channels=ds.num_channels)
    model.eval()
    loss_fn = build_loss_fn("bce")
    with torch.no_grad():
        pred = model(batch.x)
        loss = loss_fn(pred, batch.y, batch.loss_mask)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"


# ---------------------------------------------------------------------------
# Tests: dataset versioning (v1)
# ---------------------------------------------------------------------------

V1_DIR = DATASET_DIR / "v1"
V2_DIR = DATASET_DIR / "v2"


@pytest.mark.skipif(
    not (V1_DIR / "dataset_0.h5").exists(),
    reason="v1 directory not set up",
)
class TestVersioningV1:

    def test_v1_loads(self):
        """dataset_version='v1' loads from v1/ subdirectory."""
        from firecomp.next_day.config import NextDayConfig
        from firecomp.next_day.dataset import NextDayDataset

        cfg = NextDayConfig(
            dataset_dir=str(DATASET_DIR),
            dataset_version="v1",
            device="cpu",
            num_workers=0,
            batch_size=4,
        )
        ds = NextDayDataset(cfg)
        print(f"\n  v1 dataset: {len(ds.train_samples)} train, "
              f"{len(ds.val_samples)} val, {len(ds.test_samples)} test")
        print(f"  num_channels: {ds.num_channels}")
        print(f"  input groups: {list(ds.input_groups.keys())}")
        assert len(ds.train_samples) > 0
        assert ds.num_channels > 0

    def test_v1_batch_matches_latest(self):
        """v1 and latest produce batches with the same shape/channels."""
        from firecomp.next_day.config import NextDayConfig
        from firecomp.next_day.dataset import NextDayDataset

        cfg_latest = NextDayConfig(
            dataset_dir=str(DATASET_DIR), dataset_version="latest",
            device="cpu", num_workers=0, batch_size=2,
        )
        cfg_v1 = NextDayConfig(
            dataset_dir=str(DATASET_DIR), dataset_version="v1",
            device="cpu", num_workers=0, batch_size=2,
        )
        ds_latest = NextDayDataset(cfg_latest)
        ds_v1 = NextDayDataset(cfg_v1)

        print(f"\n  latest samples: {len(ds_latest.train_samples)} train")
        print(f"  v1 samples:     {len(ds_v1.train_samples)} train")
        print(f"  latest channels: {ds_latest.num_channels}")
        print(f"  v1 channels:     {ds_v1.num_channels}")

        assert ds_latest.num_channels == ds_v1.num_channels
        assert len(ds_latest.train_samples) == len(ds_v1.train_samples)

    def test_v1_forward_pass(self):
        """Full forward pass works with v1 data."""
        from firecomp.next_day.config import NextDayConfig
        from firecomp.next_day.dataset import NextDayDataset
        from firecomp.models.segmentation_models import model_factory

        cfg = NextDayConfig(
            dataset_dir=str(DATASET_DIR), dataset_version="v1",
            device="cpu", num_workers=0, batch_size=2,
        )
        ds = NextDayDataset(cfg)
        batch = next(iter(ds.train()))
        model = model_factory["unet"](in_channels=ds.num_channels)
        model.eval()
        with torch.no_grad():
            pred = model(batch.x)
        print(f"\n  v1 forward pass: input {batch.x.shape} → output {pred.shape}")
        assert pred.shape == (batch.x.shape[0], 1, 256, 256)
        assert torch.isfinite(pred).all()


@pytest.mark.skipif(
    not (V2_DIR / "dataset_0.h5").exists(),
    reason="v2 directory not set up (run: patch --filter-inconsistent)",
)
class TestVersioningV2:

    def test_v2_loads(self):
        """dataset_version='v2' loads filtered dataset."""
        from firecomp.next_day.config import NextDayConfig
        from firecomp.next_day.dataset import NextDayDataset

        cfg = NextDayConfig(
            dataset_dir=str(DATASET_DIR),
            dataset_version="v2",
            device="cpu",
            num_workers=0,
            batch_size=4,
        )
        ds = NextDayDataset(cfg)
        total = len(ds.train_samples) + len(ds.val_samples) + len(ds.test_samples)
        print(f"\n  v2 dataset: {len(ds.train_samples)} train, "
              f"{len(ds.val_samples)} val, {len(ds.test_samples)} test")
        print(f"  v2 total: {total} (should be < 1577, inconsistent samples removed)")
        print(f"  num_channels: {ds.num_channels}")
        assert len(ds.train_samples) > 0
        assert total < 1577, "v2 should have fewer samples than v1"

    def test_v2_fewer_samples_than_latest(self):
        """v2 has strictly fewer samples than latest (inconsistent ones removed)."""
        from firecomp.next_day.config import NextDayConfig
        from firecomp.next_day.dataset import NextDayDataset

        cfg_latest = NextDayConfig(
            dataset_dir=str(DATASET_DIR), dataset_version="latest",
            device="cpu", num_workers=0, batch_size=2,
        )
        cfg_v2 = NextDayConfig(
            dataset_dir=str(DATASET_DIR), dataset_version="v2",
            device="cpu", num_workers=0, batch_size=2,
        )
        ds_latest = NextDayDataset(cfg_latest)
        ds_v2 = NextDayDataset(cfg_v2)
        n_latest = len(ds_latest.train_samples) + len(ds_latest.val_samples) + len(ds_latest.test_samples)
        n_v2 = len(ds_v2.train_samples) + len(ds_v2.val_samples) + len(ds_v2.test_samples)

        print(f"\n  latest total: {n_latest}")
        print(f"  v2 total:     {n_v2}")
        print(f"  removed:      {n_latest - n_v2}")
        assert n_v2 < n_latest

    def test_v2_no_inconsistent_vnp14(self):
        """v2 should have no samples with vnp14 shape != (1, 256, 256)."""
        import hdf5plugin  # noqa: F401
        import h5py

        with open(V2_DIR / "dataset_0.json") as f:
            samples = json.load(f)

        h5 = h5py.File(V2_DIR / "dataset_0.h5", "r")
        bad = []
        for s in samples:
            # Use idx (original H5 key), not list position
            idx = s["idx"]
            vnp14_shape = h5[str(idx)]["vnp14"].shape
            if vnp14_shape != (1, 256, 256):
                bad.append((idx, vnp14_shape))
        h5.close()

        print(f"\n  v2 samples checked: {len(samples)}")
        print(f"  vnp14 inconsistent: {len(bad)}")
        if bad:
            print(f"  bad samples: {bad[:10]}")
        assert len(bad) == 0, f"{len(bad)} samples still have inconsistent vnp14"

    def test_v2_forward_pass(self):
        """Forward pass works with v2 data."""
        from firecomp.next_day.config import NextDayConfig
        from firecomp.next_day.dataset import NextDayDataset
        from firecomp.models.segmentation_models import model_factory

        cfg = NextDayConfig(
            dataset_dir=str(DATASET_DIR), dataset_version="v2",
            device="cpu", num_workers=0, batch_size=2,
        )
        ds = NextDayDataset(cfg)
        batch = next(iter(ds.train()))
        model = model_factory["unet"](in_channels=ds.num_channels)
        model.eval()
        with torch.no_grad():
            pred = model(batch.x)
        print(f"\n  v2 forward pass: input {batch.x.shape} → output {pred.shape}")
        assert pred.shape == (batch.x.shape[0], 1, 256, 256)
        assert torch.isfinite(pred).all()

    def test_v2_patch_meta(self):
        """patch_meta.json exists and records the filters applied."""
        meta_path = V2_DIR / "patch_meta.json"
        assert meta_path.exists(), "patch_meta.json missing"

        with open(meta_path) as f:
            meta = json.load(f)

        print(f"\n  patch_meta: {json.dumps(meta, indent=2)}")
        assert meta["n_target"] < meta["n_source"]
        assert meta["filters"]["filter_inconsistent"] is True


# ---------------------------------------------------------------------------
# Tests: dataset integrity verification
# ---------------------------------------------------------------------------

class TestDatasetIntegrity:
    """Verify H5 contents: field presence, shapes, dtypes, value ranges."""

    @pytest.fixture(scope="class")
    def h5_data(self):
        """Open H5 and JSON for the class."""
        import hdf5plugin  # noqa: F401
        import h5py

        h5 = h5py.File(H5_FILE, "r")
        with open(DATASET_DIR / "dataset_0.json") as f:
            samples = json.load(f)
        yield h5, samples
        h5.close()

    def test_sample_count_matches(self, h5_data):
        """H5 numeric keys and JSON entries are aligned."""
        h5, samples = h5_data
        numeric_keys = [k for k in h5 if k.isdigit()]
        print(f"\n  JSON samples:    {len(samples)}")
        print(f"  H5 sample keys:  {len(numeric_keys)}")
        print(f"  H5 key range:    0..{max(int(k) for k in numeric_keys)}")
        assert len(samples) == len(numeric_keys)

    def test_all_samples_have_core_fields(self, h5_data):
        """Every sample has the minimum fields needed for training."""
        h5, samples = h5_data
        core_fields = {"accum_t", "cur_mask", "next_mask", "new_fires",
                       "ignition", "vnp14", "weather", "gfs"}

        missing_report: dict[str, list[int]] = defaultdict(list)
        for i in range(len(samples)):
            grp = h5[str(i)]
            actual = set(grp.keys())
            for field in core_fields:
                if field not in actual:
                    missing_report[field].append(i)

        print(f"\n  Checked {len(samples)} samples for {len(core_fields)} core fields")
        if missing_report:
            for field, idxs in sorted(missing_report.items()):
                print(f"  MISSING '{field}' in {len(idxs)} samples: "
                      f"{idxs[:10]}{'...' if len(idxs) > 10 else ''}")
        else:
            print("  All core fields present in every sample")

        assert not missing_report, (
            f"Missing core fields: "
            + ", ".join(f"{f} ({len(v)} samples)" for f, v in missing_report.items())
        )

    def test_canopy_height_coverage(self, h5_data):
        """Report canopy_height availability (expected: only 2021+ samples)."""
        h5, samples = h5_data
        has_canopy = []
        no_canopy = []
        for i, s in enumerate(samples):
            if "canopy_height" in h5[str(i)]:
                has_canopy.append(i)
            else:
                no_canopy.append(i)

        # Year breakdown
        years_with = Counter(s["dt"][:4] for i, s in enumerate(samples) if i in set(has_canopy))
        years_without = Counter(s["dt"][:4] for i, s in enumerate(samples) if i in set(no_canopy))

        print(f"\n  canopy_height present: {len(has_canopy)} / {len(samples)} samples")
        print(f"  canopy_height absent:  {len(no_canopy)} / {len(samples)} samples")
        print(f"  Years WITH canopy:    {dict(sorted(years_with.items()))}")
        print(f"  Years WITHOUT canopy: {dict(sorted(years_without.items()))}")

        # Not an error — canopy is optional. Just report.
        assert len(has_canopy) + len(no_canopy) == len(samples)

    def test_field_shapes_consistent(self, h5_data):
        """Report shape consistency for each field across all samples."""
        h5, samples = h5_data

        field_shapes: dict[str, Counter] = defaultdict(Counter)
        for i in range(len(samples)):
            grp = h5[str(i)]
            for field_name in grp:
                field_shapes[field_name][grp[field_name].shape] += 1

        print(f"\n  Per-field shape distribution:")
        inconsistent = []
        for field, shapes in sorted(field_shapes.items()):
            if len(shapes) == 1:
                shape, count = list(shapes.items())[0]
                print(f"    {field:20s} {str(shape):20s} ({count} samples)")
            else:
                # Multiple shapes — flag it
                parts = ", ".join(f"{s}: {c}" for s, c in shapes.most_common())
                print(f"    {field:20s} *** INCONSISTENT *** {parts}")
                inconsistent.append(field)

        # vnp14 is known to have inconsistent shapes (1 or 5 channels),
        # handled by slicing to [0:1] in _load_sample. Others should be consistent.
        unexpected = [f for f in inconsistent if f != "vnp14"]
        if unexpected:
            print(f"\n  Unexpected inconsistencies: {unexpected}")
        assert not unexpected, f"Unexpected shape inconsistencies: {unexpected}"

    def test_tile_data_present(self, h5_data):
        """Per-tile fields (ae_pca_5, vnp02_terrain) exist for all tiles."""
        h5, samples = h5_data
        tile_groups = ["ae_pca_5", "ae_12345", "vnp02_terrain"]

        # Collect unique tile keys from samples
        tile_keys = set()
        for s in samples:
            tile_keys.add(f"{s['fire_id']}_{s['xi']}_{s['yi']}")

        print(f"\n  Unique tiles: {len(tile_keys)}")
        for group_name in tile_groups:
            if group_name not in h5:
                print(f"  {group_name}: GROUP MISSING from H5!")
                continue
            available = set(h5[group_name].keys())
            missing = tile_keys - available
            extra = available - tile_keys
            sample_shape = next(iter(h5[group_name].values())).shape
            sample_dtype = next(iter(h5[group_name].values())).dtype
            print(f"  {group_name:20s} tiles={len(available):4d}, "
                  f"shape={sample_shape}, dtype={sample_dtype}, "
                  f"missing={len(missing)}, extra={len(extra)}")
            if missing:
                print(f"    Missing tiles: {list(missing)[:5]}...")
            assert not missing, f"{group_name}: {len(missing)} tiles missing"

    def test_statistics_completeness(self, h5_data):
        """Statistics group has mean/std for normalized fields."""
        h5, _ = h5_data
        expected_fields = ["ae_pca_5", "ae_12345", "vnp02_terrain",
                           "weather", "weather_pct", "gfs"]

        assert "statistics" in h5, "No 'statistics' group in H5"
        stats = h5["statistics"]
        print(f"\n  Statistics fields: {list(stats.keys())}")

        for field in expected_fields:
            assert field in stats, f"No stats for '{field}'"
            mean = stats[field]["mean"][:]
            std = stats[field]["std"][:]
            print(f"    {field:20s} mean={np.array2string(mean, precision=3, suppress_small=True):40s} "
                  f"std={np.array2string(std, precision=3, suppress_small=True)}")
            assert mean.shape == std.shape
            assert not np.any(np.isnan(mean)), f"NaN in {field} mean"
            assert not np.any(np.isnan(std)), f"NaN in {field} std"

    def test_sample_metadata_fields(self, h5_data):
        """JSON samples have all required metadata fields."""
        _, samples = h5_data
        required = {"fire_id", "xi", "yi", "lon", "lat", "dt", "img_size"}

        print(f"\n  JSON sample fields (first entry): {list(samples[0].keys())}")

        for i, s in enumerate(samples):
            missing = required - set(s.keys())
            assert not missing, f"Sample {i} missing fields: {missing}"

        # Summary stats
        fire_ids = set(s["fire_id"] for s in samples)
        years = Counter(s["dt"][:4] for s in samples)
        print(f"  Total samples:  {len(samples)}")
        print(f"  Unique fires:   {len(fire_ids)}")
        print(f"  Year distribution: {dict(sorted(years.items()))}")
        print(f"  Lat range:      [{min(s['lat'] for s in samples):.2f}, "
              f"{max(s['lat'] for s in samples):.2f}]")
        print(f"  Lon range:      [{min(s['lon'] for s in samples):.2f}, "
              f"{max(s['lon'] for s in samples):.2f}]")
