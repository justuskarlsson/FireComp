"""
Tests for the distill pipeline — specifically verifying that multi-shard
source datasets are correctly merged into destination files without
data loss or key collisions.

The original bug: samples from different source shards can have the same
local `idx` (e.g., idx=0 in shard 0 and idx=0 in shard 3). When distill
merges them into a single output file, the second copy would fail with
"destination object already exists".

The fix: re-index samples sequentially in the output and update s.idx so
the JSON metadata matches.

Run:  pytest tests/test_distill.py -v -s
"""

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures — synthetic multi-shard dataset
# ---------------------------------------------------------------------------

@dataclass
class FakeSample:
    """Minimal Sample for testing."""
    fire_id: int
    xi: int
    yi: int
    lon: float
    lat: float
    dt: str
    img_size: int = 256
    idx: int = -1

    @property
    def tile_key(self) -> str:
        return f"{self.fire_id}_{self.xi}_{self.yi}"


def _make_source_dataset(tmp_path: Path, n_shards: int = 3,
                         samples_per_shard: int = 5,
                         n_fires: int = 2) -> Path:
    """
    Create a synthetic multi-shard dataset where each shard has
    locally-indexed samples (0..samples_per_shard-1).

    Each sample's H5 group contains a small 'data' array with a unique
    fingerprint so we can verify correct data was copied.
    """
    src = tmp_path / "source"
    src.mkdir()

    all_samples: list[FakeSample] = []

    for shard_idx in range(n_shards):
        h5_path = src / f"dataset_{shard_idx}.h5"
        shard_samples = []

        with h5py.File(h5_path, "w") as h5:
            # Add statistics group to first shard
            if shard_idx == 0:
                stats_grp = h5.create_group("statistics")
                wg = stats_grp.create_group("weather")
                wg.create_dataset("mean", data=np.zeros(3))
                wg.create_dataset("std", data=np.ones(3))

            for local_idx in range(samples_per_shard):
                # Unique fingerprint: encodes shard + local index
                fingerprint = np.array([shard_idx * 1000 + local_idx],
                                       dtype=np.float32)
                grp = h5.create_group(str(local_idx))
                grp.create_dataset("data", data=fingerprint)
                # Also store a 2D field to simulate real data
                grp.create_dataset("cur_mask",
                                   data=np.full((1, 4, 4),
                                                shard_idx * 100 + local_idx,
                                                dtype=np.float32))

                fire_id = (shard_idx * samples_per_shard + local_idx) % n_fires
                s = FakeSample(
                    fire_id=fire_id,
                    xi=0, yi=0,
                    lon=10.0 + shard_idx,
                    lat=60.0 + local_idx,
                    dt=f"2023-01-{local_idx + 1:02d}",
                    idx=local_idx,
                )
                s._h5_file = shard_idx
                shard_samples.append(s)

            # Add a per-tile group
            tile_grp = h5.create_group("ae_pca_5")
            for s in shard_samples:
                tk = s.tile_key
                if tk not in tile_grp:
                    tile_grp.create_dataset(
                        tk, data=np.ones((5, 4, 4), dtype=np.float32))

        # Write JSON for this shard
        raw = []
        for s in shard_samples:
            d = {k: v for k, v in s.__dict__.items() if not k.startswith("_")}
            raw.append(d)
        json_path = src / f"dataset_{shard_idx}.json"
        with open(json_path, "w") as f:
            json.dump(raw, f)

        all_samples.extend(shard_samples)

    return src, all_samples


@pytest.fixture
def source_dataset(tmp_path):
    """Create a 3-shard source dataset with 5 samples each."""
    src_path, samples = _make_source_dataset(tmp_path, n_shards=3,
                                              samples_per_shard=5,
                                              n_fires=2)
    return src_path, samples


# ---------------------------------------------------------------------------
# Tests: _write_distill_output correctness
# ---------------------------------------------------------------------------

class TestDistillOutput:
    """Verify distill produces correct, loadable output."""

    def test_no_collision_same_local_idx(self, tmp_path):
        """
        Core regression test: samples from different shards with the same
        local idx must not collide in the output.
        """
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        # This would previously raise:
        # RuntimeError: Unable to synchronously copy object
        #              (destination object already exists)
        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        # Verify output exists
        assert (dest / "dataset_0.h5").exists()
        assert (dest / "dataset_0.json").exists()

    def test_all_samples_present_in_output(self, tmp_path):
        """Every input sample appears in the output with correct data."""
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        # Record expected fingerprints BEFORE distill (uses original idx)
        expected_fingerprints = []
        src_h5s = [h5py.File(p, "r") for p in src_h5_files]
        for s in samples:
            fp = src_h5s[s._h5_file][str(s.idx)]["data"][:]
            expected_fingerprints.append(fp.item())
        for h5 in src_h5s:
            h5.close()

        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        # Read output and verify all fingerprints present
        with h5py.File(dest / "dataset_0.h5", "r") as dst:
            actual_fingerprints = set()
            # Numeric keys are sample groups
            numeric_keys = sorted(
                [k for k in dst.keys() if k.isdigit()], key=int)
            assert len(numeric_keys) == len(samples)
            for key in numeric_keys:
                fp = dst[key]["data"][:].item()
                actual_fingerprints.add(fp)

        expected_set = set(expected_fingerprints)
        assert actual_fingerprints == expected_set, (
            f"Missing: {expected_set - actual_fingerprints}, "
            f"Extra: {actual_fingerprints - expected_set}")

    def test_json_idx_matches_h5_keys(self, tmp_path):
        """JSON idx fields correctly map to H5 group keys after re-index."""
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        # Record original fingerprints keyed by sample identity
        src_h5s = [h5py.File(p, "r") for p in src_h5_files]
        sample_fingerprints = {}
        for i, s in enumerate(samples):
            fp = src_h5s[s._h5_file][str(s.idx)]["data"][:].item()
            sample_fingerprints[i] = fp
        for h5 in src_h5s:
            h5.close()

        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        # Load JSON and verify each idx points to correct H5 data
        with open(dest / "dataset_0.json") as f:
            json_samples = json.load(f)

        with h5py.File(dest / "dataset_0.h5", "r") as h5:
            for i, js in enumerate(json_samples):
                idx = js["idx"]
                assert str(idx) in h5, f"JSON idx={idx} not in H5"
                actual_fp = h5[str(idx)]["data"][:].item()
                expected_fp = sample_fingerprints[i]
                assert actual_fp == expected_fp, (
                    f"Sample {i}: JSON idx={idx} points to wrong data. "
                    f"Expected fingerprint {expected_fp}, got {actual_fp}")

    def test_sequential_idx_in_output(self, tmp_path):
        """Output samples have sequential idx 0..N-1."""
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        with open(dest / "dataset_0.json") as f:
            json_samples = json.load(f)

        indices = [s["idx"] for s in json_samples]
        assert indices == list(range(len(samples)))

    def test_multi_output_files(self, tmp_path):
        """Distill into multiple output files — each is self-consistent."""
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=4, samples_per_shard=8, n_fires=3)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        n_files = 3
        _write_distill_output(src_h5_files, dest, samples, n_files=n_files)

        total_samples = 0
        for k in range(n_files):
            h5_path = dest / f"dataset_{k}.h5"
            json_path = dest / f"dataset_{k}.json"
            if not h5_path.exists():
                continue

            with open(json_path) as f:
                js = json.load(f)

            with h5py.File(h5_path, "r") as h5:
                numeric_keys = [key for key in h5.keys() if key.isdigit()]
                # JSON count == H5 key count
                assert len(js) == len(numeric_keys), (
                    f"Shard {k}: JSON has {len(js)} samples but "
                    f"H5 has {len(numeric_keys)} keys")
                # Each JSON idx exists in H5
                for s in js:
                    assert str(s["idx"]) in h5, (
                        f"Shard {k}: idx={s['idx']} not in H5")

            total_samples += len(js)

        assert total_samples == len(samples)

    def test_roundtrip_via_sample_store(self, tmp_path):
        """
        Full round-trip: distill → SampleStore.load_samples → load each
        sample → verify data matches source.
        """
        from firecomp.next_day.preprocess import _write_distill_output
        from firecomp.core.dataset_utils import SampleStore

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))

        # Record expected fingerprints in order
        src_h5s = [h5py.File(p, "r") for p in src_h5_files]
        expected = []
        for s in samples:
            fp = src_h5s[s._h5_file][str(s.idx)]["data"][:].item()
            expected.append(fp)
        for h5 in src_h5s:
            h5.close()

        dest = tmp_path / "dest"
        dest.mkdir()
        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        # Now load via SampleStore (as training would)
        store = SampleStore(dest, sample_cls=FakeSample)
        loaded_samples = store.load_samples()

        assert len(loaded_samples) == len(expected)
        for i, s in enumerate(loaded_samples):
            data = store.load(s)
            actual_fp = data["data"].item()
            assert actual_fp == expected[i], (
                f"Sample {i} (fire={s.fire_id}, dt={s.dt}): "
                f"loaded fingerprint {actual_fp} != expected {expected[i]}. "
                f"idx={s.idx}, _h5_file={s._h5_file}")

        store.close()

    def test_statistics_copied(self, tmp_path):
        """Statistics group is copied to first output shard."""
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=2, samples_per_shard=3, n_fires=1)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        with h5py.File(dest / "dataset_0.h5", "r") as h5:
            assert "statistics" in h5
            assert "weather" in h5["statistics"]
            np.testing.assert_array_equal(
                h5["statistics"]["weather"]["mean"][:], np.zeros(3))

    def test_tile_data_copied(self, tmp_path):
        """Per-tile data is merged from all source shards."""
        from firecomp.next_day.preprocess import _write_distill_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "dest"
        dest.mkdir()

        _write_distill_output(src_h5_files, dest, samples, n_files=1)

        expected_tiles = {s.tile_key for s in samples}
        with h5py.File(dest / "dataset_0.h5", "r") as h5:
            assert "ae_pca_5" in h5
            actual_tiles = set(h5["ae_pca_5"].keys())
            assert expected_tiles <= actual_tiles, (
                f"Missing tiles: {expected_tiles - actual_tiles}")


# ---------------------------------------------------------------------------
# Tests: patch workflow (should NOT have the same bug)
# ---------------------------------------------------------------------------

class TestPatchNoCollision:
    """Verify patch preserves shard-local idx correctly."""

    def test_patch_preserves_shard_mapping(self, tmp_path):
        """
        Patch keeps samples grouped by source shard — no cross-shard merging,
        so no idx collision is possible.
        """
        from firecomp.next_day.preprocess import _write_patch_output

        src_path, samples = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)
        src_h5_files = sorted(src_path.glob("dataset_*.h5"))
        dest = tmp_path / "patch_dest"
        dest.mkdir()

        # Filter: keep only fire_id=0
        filtered = [s for s in samples if s.fire_id == 0]

        _write_patch_output(src_path, dest, src_h5_files, filtered,
                            source_version="v1")

        # Verify: each output JSON only references its own shard's indices
        for k in range(3):
            json_path = dest / f"dataset_{k}.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                js = json.load(f)

            # The H5 is a symlink — read through it
            h5_path = dest / f"dataset_{k}.h5"
            if not h5_path.exists():
                continue
            with h5py.File(h5_path, "r") as h5:
                for s in js:
                    idx = s["idx"]
                    assert str(idx) in h5, (
                        f"Shard {k}: idx={idx} not in H5 "
                        f"(shard keys: {[x for x in h5.keys() if x.isdigit()][:10]})")


# ---------------------------------------------------------------------------
# Tests: SampleStore loading correctness with multi-shard
# ---------------------------------------------------------------------------

class TestSampleStoreMultiShard:
    """Verify SampleStore correctly routes to the right shard."""

    def test_load_routes_to_correct_shard(self, tmp_path):
        """
        With a multi-shard dataset, SampleStore.load() uses _h5_file to
        route to the correct shard — even when indices overlap.
        """
        from firecomp.core.dataset_utils import SampleStore

        src_path, _ = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)

        store = SampleStore(src_path, sample_cls=FakeSample)
        loaded = store.load_samples()

        # Each sample should have the correct shard assignment
        shard_counts = defaultdict(int)
        for s in loaded:
            shard_counts[s._h5_file] += 1
            # Verify the fingerprint matches expected
            data = store.load(s)
            expected_fp = s._h5_file * 1000 + s.idx
            actual_fp = data["data"].item()
            assert actual_fp == expected_fp, (
                f"Sample shard={s._h5_file} idx={s.idx}: "
                f"expected fp={expected_fp}, got {actual_fp}")

        # Should have samples from all 3 shards
        assert len(shard_counts) == 3
        store.close()

    def test_same_idx_different_shards_not_confused(self, tmp_path):
        """
        Two samples with idx=0 in different shards must return different data.
        This is the scenario that would fail if _h5_file routing was broken.
        """
        from firecomp.core.dataset_utils import SampleStore

        src_path, _ = _make_source_dataset(
            tmp_path, n_shards=3, samples_per_shard=5, n_fires=2)

        store = SampleStore(src_path, sample_cls=FakeSample)
        loaded = store.load_samples()

        # Find samples with idx=0 from different shards
        idx0_samples = [s for s in loaded if s.idx == 0]
        assert len(idx0_samples) == 3, "Should have idx=0 from each of 3 shards"

        fingerprints = []
        for s in idx0_samples:
            data = store.load(s)
            fingerprints.append(data["data"].item())

        # All fingerprints must be different
        assert len(set(fingerprints)) == 3, (
            f"idx=0 samples returned same data! fps={fingerprints}")

        store.close()
