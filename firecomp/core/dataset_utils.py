"""
core/dataset_utils.py — Task-agnostic data loading utilities.

NOT a base class. Free functions, dataclasses, and a thin storage class that
NextDayDataset calls into.

Design:
  - Field      — plain dataclass describing one input/target/aux field
  - load_field — applies dtype_transform → value transform → normalize → upsample
  - SampleStore — H5 storage backend (multi-shard)
  - NormStats   — per-channel mean/std normalization
  - load_samples_json — reads sample metadata
  - make_data_loader — wraps torch.DataLoader with sensible defaults

Both task datasets share these primitives without inheriting from a common
base class. Each task owns its own __getitem__ / collate / split logic.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import h5py
import hdf5plugin  # noqa: F401 — registers zstd filter
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud


# ---------------------------------------------------------------------------
# Field — declarative spec for one data field
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """
    Describes one data field: where to find it, native size, channel count,
    whether to normalize, and optional transforms.

    Replaces the DataField/DataInput/DataAux descriptor machinery + SetGet +
    FireSetGet. Plain dataclass — loading logic lives in load_field, not on
    the field itself.

    Args:
        name:            field identifier (matches filename / H5 key)
        channels:        number of channels (C dim)
        native_size:     spatial size on disk. 256 for full-res, 32 for
                         under-sampled fields like weather/gfs
        normalize:       apply (x - mean) / std using NormStats
        transform:       optional value transform applied AFTER dtype_transform
                         (e.g. rel_t for accum_t)
        dtype_transform: optional dtype/scale transform (e.g. uint8 → float)
        storage:         "per_sample" (one file per sample dir) |
                         "per_tile"   (shared across timesteps of same tile)
    """
    name: str
    channels: int
    native_size: int = 256
    normalize: bool = False
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    dtype_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    storage: str = "per_sample"


# ---------------------------------------------------------------------------
# load_field — single source of truth for the per-field loading pipeline
# ---------------------------------------------------------------------------

def load_field(field: Field, data: np.ndarray, stats: "NormStats",
               target_size: int = 256) -> np.ndarray:
    """
    Apply the standard per-field loading pipeline to a raw numpy array.

    Pipeline:
        1. dtype_transform (e.g. uint8 → float)
        2. transform       (e.g. rel_t for accum_t)
        3. normalize       (per-channel mean/std)
        4. upsample        (if native_size != target_size)

    Returns a (C, target_size, target_size) float32 array.
    """
    if field.dtype_transform is not None:
        data = field.dtype_transform(data)
    else:
        data = data.astype(np.float32)

    if field.transform is not None:
        data = field.transform(data)

    if field.normalize:
        data = stats.normalize(field.name, data)

    if field.native_size != target_size:
        data = upsample_np(data, target_size)

    return data


def upsample_np(data: np.ndarray, target_size: int) -> np.ndarray:
    """(C, h, w) → (C, target_size, target_size) via bilinear interpolation."""
    t = torch.from_numpy(data).unsqueeze(0).float()
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Channel-range bookkeeping
# ---------------------------------------------------------------------------

def channel_ranges(fields: list[Field]) -> dict[str, tuple[int, int]]:
    """
    Compute (start, end) channel index ranges for a list of input fields.

    Used for ablation: zero channels[start:end] = drop that input group.

    >>> channel_ranges([Field("ae_pca_5", 5), Field("weather", 6)])
    {"ae_pca_5": (0, 5), "weather": (5, 11)}
    """
    ranges = {}
    offset = 0
    for f in fields:
        ranges[f.name] = (offset, offset + f.channels)
        offset += f.channels
    return ranges


# ---------------------------------------------------------------------------
# SampleStore — storage backend
# ---------------------------------------------------------------------------

class SampleStore:
    """
    H5 dataset store — opens shards, loads samples, provides stats.

    Owns everything about reading a dataset directory:
      - ``dataset_*.h5``  — data shards
      - ``dataset_*.json`` — per-shard sample metadata
      - ``statistics/``    — normalization stats (in first shard)

    Samples returned by ``load_samples()`` carry a ``_h5_file`` attribute
    so ``load()`` routes to the correct shard.
    """

    def __init__(self, dataset_dir: Path, sample_cls=None):
        self.root = Path(dataset_dir)

        self._h5_paths = sorted(
            p for p in self.root.glob("dataset_*.h5")
        )
        if not self._h5_paths:
            raise FileNotFoundError(
                f"No dataset_*.h5 files in {self.root}")
        self._h5s = [h5py.File(p, "r") for p in self._h5_paths]

        # Sample metadata (one JSON per shard)
        self._sample_cls = sample_cls
        self._json_paths = sorted(
            p for p in self.root.glob("dataset_*.json")
        )

    # -- samples --

    @property
    def stats(self) -> "NormStats":
        """Normalization stats — from ``stats.json`` (v3) or H5 ``statistics/`` (v2)."""
        stats_json = self.root / "stats.json"
        if stats_json.exists():
            return NormStats(stats_json)
        # v2 fallback: stats embedded in first H5 shard
        if "statistics" in self._h5s[0]:
            return NormStats(self._h5s[0]["statistics"])
        raise FileNotFoundError(
            f"No stats.json in {self.root} and no 'statistics' group in H5.\n"
            f"Run:  python -m firecomp.next_day.preprocess compute-stats "
            f"--dataset-dir {self.root}"
        )

    def load_samples(self) -> list:
        """Read all ``dataset_*.json`` files, tag each with ``_h5_file``."""
        all_samples: list = []
        for file_idx, json_path in enumerate(self._json_paths):
            file_samples = load_samples_json(json_path, self._sample_cls)
            for i, s in enumerate(file_samples):
                if hasattr(s, 'idx') and s.idx == -1:
                    s.idx = i
                s._h5_file = file_idx
            all_samples.extend(file_samples)
        return all_samples

    # -- per-sample data --

    def load(self, sample) -> dict[str, np.ndarray]:
        """Load all fields for one sample → dict of numpy arrays."""
        h5 = self._h5s[getattr(sample, '_h5_file', 0)]
        grp = h5[str(sample.idx)]
        return {k: grp[k][:] for k in grp}

    def load_tile(self, tile_key: str, field_name: str) -> np.ndarray:
        """Load a per-tile field (shared across timesteps of same tile)."""
        for h5 in self._h5s:
            if field_name in h5 and tile_key in h5[field_name]:
                return h5[field_name][tile_key][:]
        raise KeyError(f"{field_name}/{tile_key} not found in any shard")

    def load_field(self, sample, field_name: str) -> np.ndarray:
        """Load a single field for one sample."""
        h5 = self._h5s[getattr(sample, '_h5_file', 0)]
        return h5[str(sample.idx)][field_name][:]

    # -- lifecycle --

    def reopen(self):
        """Re-open H5 files. Call from DataLoader worker_init_fn."""
        self._h5s = [h5py.File(p, "r") for p in self._h5_paths]

    def close(self):
        for h5 in self._h5s:
            h5.close()
        self._h5s = []


# ---------------------------------------------------------------------------
# Sample metadata loading
# ---------------------------------------------------------------------------

def load_samples_json(path: Path, sample_cls=None) -> list:
    """
    Load samples.json → list of sample metadata.

    If sample_cls is provided, each dict is unpacked into that dataclass.
    Extra keys not in the dataclass are silently dropped (Phase 1 JSONs
    include metadata like ``num_fire``, ``day_of_fire`` that don't map to
    the Sample dataclass).
    """
    with open(path) as f:
        raw = json.load(f)
    if sample_cls is not None:
        from dataclasses import fields as dc_fields
        valid_keys = {f.name for f in dc_fields(sample_cls)}
        return [sample_cls(**{k: v for k, v in d.items() if k in valid_keys})
                for d in raw]
    return raw


# ---------------------------------------------------------------------------
# DataLoader wiring
# ---------------------------------------------------------------------------

def make_data_loader(dataset: tud.Dataset, batch_size: int, shuffle: bool,
                     num_workers: int = 4, collate_fn=None,
                     pin_memory: bool = True,
                     worker_init_fn=None) -> tud.DataLoader:
    """
    Thin wrapper around torch.DataLoader with sensible defaults.

    worker_init_fn: called in each worker after fork. Used by the H5 backend
    to re-open file handles (h5py handles are not fork-safe).
    """
    use_workers = num_workers > 0
    return tud.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
    )


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

class NormStats:
    """
    Per-channel mean/std for normalized fields.

    Supports two sources:
    - JSON file (``stats.json``): ``{"field": {"mean": [...], "std": [...]}}``
    - H5 group (``statistics/``): ``statistics/field/mean``, ``statistics/field/std``
    """

    def __init__(self, source):
        """
        Args:
            source: Path to stats.json, OR an h5py Group containing
                    per-field mean/std datasets.
        """
        if isinstance(source, Path) or isinstance(source, str):
            source = Path(source)
            with open(source) as f:
                raw = json.load(f)
            self._stats = {
                name: (np.array(s["mean"], dtype=np.float32),
                       np.array(s["std"], dtype=np.float32))
                for name, s in raw.items()
            }
        else:
            # h5py Group: statistics/field_name/{mean, std}
            self._stats = {
                name: (source[name]["mean"][:].astype(np.float32),
                       source[name]["std"][:].astype(np.float32))
                for name in source
            }

    def normalize(self, name: str, data: np.ndarray) -> np.ndarray:
        """(C, H, W) → normalized. No-op if field has no stats."""
        if name not in self._stats:
            return data
        mean, std = self._stats[name]
        return (data - mean[:, None, None]) / std[:, None, None]

    def has(self, name: str) -> bool:
        return name in self._stats


# ---------------------------------------------------------------------------
# Common dtype transforms (shared by both tasks)
# ---------------------------------------------------------------------------

def uint8_to_float(data: np.ndarray) -> np.ndarray:
    """AE embeddings: uint8 → float [-0.6, 0.6], float → float32 with nodata masking.

    uint8 raw embeddings (ae_12345): 0-254 are valid, 255 = nodata.
    PCA embeddings (ae_pca_5): already float, -32768 = nodata sentinel.
    Both nodata values are replaced with 0.0 (≈ channel mean).
    """
    if data.dtype == np.uint8:
        # Inverse of ae_float_to_uint8: 0-254 → [-0.6, 0.6]
        out = -0.6 + (data.astype(np.float32) / 254.0) * 1.2
        out[data == 255] = 0.0
        return out
    # float PCA: -32768 is nodata
    out = data.astype(np.float32)
    out[out < -1e4] = 0.0
    return out


def float16_to_float32(data: np.ndarray) -> np.ndarray:
    """Some fields stored as float16 to save space; cast for compute."""
    if data.dtype == np.float16:
        return data.astype(np.float32)
    return data
