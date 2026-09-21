"""
next_day/dataset.py — NextDayDataset for per-pixel fire spread.

Shared loading mechanics live in core/dataset_utils.py — this module just
declares NextDay's fields, computes its loss mask, and owns its split logic.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import torch

from firecomp.core.dataset_utils import (
    Field,
    SampleStore,
    channel_ranges,
    load_field,
    make_data_loader,
    uint8_to_float,
)
from firecomp.core.torch_utils import get_device
from firecomp.next_day.config import NextDayConfig


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One tile at one timestep of one fire."""
    fire_id: int
    xi: int                          # tile x index within fire bounding box
    yi: int                          # tile y index within fire bounding box
    lon: float
    lat: float
    dt: str                          # ISO date
    img_size: int                    # always 256
    idx: int = -1                    # filesystem index (set after JSON load)
    fire_type: int = -1              # FireType: 0=vegetation, 1=static, 2=crop
    region_id: int = 0               # wildfire_regions.tif region
    num_fire: int = 0                # lifetime detection count of the fire

    @property
    def tile_key(self) -> str:
        """Same fire + same tile = same key across timesteps."""
        return f"{self.fire_id}_{self.xi}_{self.yi}"


# Backward compat: old name for NextDayConfig (used by external callers).
DataConfig = NextDayConfig


# ---------------------------------------------------------------------------
# Field declarations — single source of truth for channels
# ---------------------------------------------------------------------------

ALL_FIELDS: dict[str, Field] = {
    # -- inputs --
    "accum_t":       Field("accum_t",       channels=1, transform=lambda d: _rel_t(d)),
    "accum_t_min":   Field("accum_t_min",   channels=1, transform=lambda d: _rel_t(d)),
    "accum_t_max":   Field("accum_t_max",   channels=1, transform=lambda d: _rel_t(d)),
    "accum_t_count": Field("accum_t_count", channels=1, normalize=True),
    "cur_mask":      Field("cur_mask",      channels=1),
    "ae_pca_5":      Field("ae_pca_5",      channels=5, normalize=True,
                           dtype_transform=uint8_to_float),
    "ae_12345":      Field("ae_12345",      channels=5, normalize=True,
                           dtype_transform=uint8_to_float),
    "vnp02_terrain": Field("vnp02_terrain", channels=5, normalize=True, storage="per_tile"),
    "weather":       Field("weather",       channels=6, native_size=32, normalize=True),
    "weather_pct":   Field("weather_pct",   channels=6, native_size=32, normalize=True),
    "gfs":           Field("gfs",           channels=5, native_size=32, normalize=True),
    "canopy_height": Field("canopy_height", channels=1, normalize=True),
    "pos_encoding":  Field("pos_encoding",  channels=4),

    # -- targets --
    "next_mask":     Field("next_mask",     channels=1),
    "new_fires":     Field("new_fires",     channels=1),

    # -- aux (for loss mask, v3) --
    "vnp14_t":           Field("vnp14_t",           channels=1),
    "vnp14_t1":          Field("vnp14_t1",          channels=1),
    "fire_id_mask_t1":   Field("fire_id_mask_t1",   channels=1),
}


# Display names per field, in channel order.  Used by channel_names property
# and figure generation.  Must stay in sync with ALL_FIELDS channel counts.
FIELD_DISPLAY_NAMES: dict[str, list[str]] = {
    "accum_t":       ["accum_t"],
    "accum_t_min":   ["accum_t_min"],
    "accum_t_max":   ["accum_t_max"],
    "accum_t_count": ["accum_t_count"],
    "cur_mask":      ["cur_mask"],
    "ae_pca_5":      ["PCA 1", "PCA 2", "PCA 3", "PCA 4", "PCA 5"],
    "ae_12345":      ["AE 1", "AE 2", "AE 3", "AE 4", "AE 5"],
    "vnp02_terrain": ["I1", "I2", "I3", "I4", "I5"],
    "weather":       [
        "Current · VPD", "Current · Soil Moist.", "Current · Soil Ratio",
        "Current · Wind Mag.", "Current · Wind sin", "Current · Wind cos",
    ],
    "weather_pct":   [
        "Current · VPD pct", "Current · Soil Moist. pct", "Current · Soil pct",
        "Current · Wind Mag. pct", "Current · Wind sin pct", "Current · Wind cos pct",
    ],
    "gfs":           [
        "Forecast · Temp Max", "Forecast · RH Min",
        "Forecast · U Wind Max", "Forecast · V Wind Max",
        "Forecast · Precip Sum",
    ],
    "canopy_height": ["Canopy Height"],
    "pos_encoding":  ["Pos sin(lat)", "Pos cos(lat)", "Pos sin(lon)", "Pos cos(lon)"],
}


_ACCUM_T_SCALE = np.float32(30.0)  # days — fire pixels older than this are clamped to 1


def _rel_t(accum_t: np.ndarray) -> np.ndarray:
    """accum_t: absolute day number (-1 = unobserved) -> days_since / 30.

    Raw accum_t stores the absolute day number (e.g. 4563) of first fire
    observation per pixel.  We convert to "days since first observation"
    relative to the most recent observation in this sample:

        0   = appeared on the most recent day (active front)
        0.03 = appeared 1 day ago
        0.5  = appeared 15 days ago
        1.0  = appeared 30+ days ago (clamped)
       -1    = unobserved (never burned — clear sentinel, distinct from 0)

    Fixed 30-day divisor gives absolute temporal meaning regardless of
    fire duration: 0.5 always means "15 days ago", whether the fire is
    20 days or 60 days old.  No dataset-level normalization needed.
    """
    accum_t = accum_t.astype(np.float32)
    mask = accum_t < 0                                  # -1 = unobserved
    observed = accum_t[~mask]
    if len(observed) == 0:
        return np.full_like(accum_t, -1.0)
    max_t = observed.max()                              # most recent day
    days_since = max_t - accum_t                        # 0 = newest
    result = np.clip(days_since / _ACCUM_T_SCALE, 0, 1.0)
    result[mask] = -1.0                                 # unobserved sentinel
    return result


def _union_accum_cur_mask(raw: dict) -> None:
    """Union cur_mask into accum_t fields (in-place on *raw*).

    cur_mask is derived from VNP14 geocoordinates and may include fire
    pixels that the accumulation (built from a separate pass) misses.
    This ensures consistency before ``_rel_t`` normalisation:

    - **accum_t / accum_t_min** (first detection): set unburned (−1)
      pixels that are in cur_mask to ``max_t`` (the sample's most
      recent observed day).  Already-burned pixels keep their original
      first-detection time.
    - **accum_t_max** (most recent detection): set ALL cur_mask pixels
      to ``max_t`` — they are burning *right now*.
    - **accum_t_count** (detection count): set to 1 where currently 0
      and cur_mask is active; leave the rest unchanged.
    """
    cur = raw.get("cur_mask")
    if cur is None:
        return
    burning = cur > 0.5                                    # (1, H, W) bool

    # --- min / accum_t (first detection) ---
    for key in ("accum_t", "accum_t_min"):
        if key not in raw:
            continue
        arr = raw[key].copy()
        observed = arr[arr >= 0]
        max_t = int(observed.max()) if len(observed) > 0 else 0
        arr[burning & (arr < 0)] = max_t                   # unburned → today
        raw[key] = arr

    # --- max (most recent detection) ---
    if "accum_t_max" in raw:
        arr = raw["accum_t_max"].copy()
        observed = arr[arr >= 0]
        max_t = int(observed.max()) if len(observed) > 0 else 0
        arr[burning] = max_t                               # all cur_mask → today
        raw["accum_t_max"] = arr

    # --- count ---
    if "accum_t_count" in raw:
        arr = raw["accum_t_count"].copy()
        arr[burning & (arr == 0)] = 1                      # 0 → 1
        raw["accum_t_count"] = arr


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    x: torch.Tensor              # (B, C, 256, 256) all inputs upsampled+normalized
    y: torch.Tensor              # (B, 1, 256, 256) target
    loss_mask: torch.Tensor      # (B, 1, 256, 256) where to compute loss
    idx: torch.Tensor            # (B,) sample indices into this split
    samples: list[Sample]        # (B,) sample metadata
    channel_names: list[str] | None = None   # (C,) display name per input channel

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        """Transfer tensors to device. Returns self for chaining."""
        self.x = self.x.to(device, non_blocking=non_blocking)
        self.y = self.y.to(device, non_blocking=non_blocking)
        self.loss_mask = self.loss_mask.to(device, non_blocking=non_blocking)
        self.idx = self.idx.to(device, non_blocking=non_blocking)
        return self


# ---------------------------------------------------------------------------
# NextDayDataset
# ---------------------------------------------------------------------------

class NextDayDataset:
    """
    Loads preprocessed next-day data and yields ready-to-train batches.

    All complexity is internal:
      - field selection from config (terrain variant, weather flags)
      - per-sample / per-tile storage routing via SampleStore
      - upsample weather 32→256, normalize, concat into x
      - compute spatial loss mask (VNP14 observation + last-day + padding)
      - split by fire_id (no fire spans two splits)
      - device transfer

    The training script just does:
        for batch in ds.train():
            pred = model(batch.x)
            loss = loss_fn(pred, batch.y, batch.loss_mask)
    """

    def __init__(self, cfg: NextDayConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device)

        dataset_path = Path(cfg.dataset_dir)
        if cfg.dataset_version != "latest":
            dataset_path = dataset_path / cfg.dataset_version

        self._store = SampleStore(dataset_path, sample_cls=Sample)
        self._stats = self._store.stats
        all_samples = self._store.load_samples()

        # Fire type filtering (v3: per-sample fire_type from LC classifier).
        if cfg.fire_type != "all":
            from firecomp.core.fire_filter import FireType
            type_map = {
                "vegetation": FireType.VEGETATION,
                "static": FireType.STATIC,
                "crop": FireType.CROP,
            }
            ft_val = type_map.get(cfg.fire_type)
            if ft_val is not None:
                n_before = len(all_samples)
                all_samples = [s for s in all_samples if s.fire_type == ft_val]
                print(f"[fire_type={cfg.fire_type}] {n_before} → "
                      f"{len(all_samples)} samples")

        # Subsample (replaces distill): keep N random samples, whole-fire.
        if cfg.max_samples > 0 and cfg.max_samples < len(all_samples):
            all_samples = _subsample_by_fire(
                all_samples, cfg.max_samples, seed=cfg.subsample_seed)
            print(f"Subsampled to {len(all_samples)} samples "
                  f"(max_samples={cfg.max_samples}, seed={cfg.subsample_seed})")

        self._input_fields = self._select_input_fields(cfg)
        self._target_field = ALL_FIELDS[cfg.target_type]
        # v3 loss mask uses vnp14_t/vnp14_t1 + fire_id_mask_t1

        self._channel_ranges = channel_ranges(self._input_fields)
        self._total_channels = sum(f.channels for f in self._input_fields)

        # Stratified temporal split — per-region 60/15/25 by date
        train_idx, val_idx, test_idx = self._split_by_date(all_samples)
        self._train_samples = [all_samples[i] for i in train_idx]
        self._val_samples = [all_samples[i] for i in val_idx]
        self._test_samples = [all_samples[i] for i in test_idx]

        # Region filtering for transfer experiments.
        # train_regions: include-filter on train + val (matrix per-region runs).
        # exclude_regions: exclude from ALL splits — train, val, AND test
        #   (LOO "data pollution" design: the excluded region is gone entirely).
        if cfg.train_regions is not None or cfg.exclude_regions is not None:
            from firecomp.core.regions import Regions
            fire_regions = Regions.build_fire_regions(all_samples)
            self._train_samples = Regions.filter(
                self._train_samples,
                include=cfg.train_regions,
                exclude=cfg.exclude_regions,
                fire_regions=fire_regions,
            )
            self._val_samples = Regions.filter(
                self._val_samples,
                include=cfg.train_regions,
                exclude=cfg.exclude_regions,
                fire_regions=fire_regions,
            )
            if cfg.exclude_regions is not None:
                self._test_samples = Regions.filter(
                    self._test_samples,
                    exclude=cfg.exclude_regions,
                    fire_regions=fire_regions,
                )
            print(f"[transfer] region filter (include={cfg.train_regions}, "
                  f"exclude={cfg.exclude_regions}): "
                  f"train={len(self._train_samples)}, "
                  f"val={len(self._val_samples)}, "
                  f"test={len(self._test_samples)}")

        # is_last_day per split — precomputed once
        self._train_is_last = _compute_is_last_day(self._train_samples)
        self._val_is_last = _compute_is_last_day(self._val_samples)
        self._test_is_last = _compute_is_last_day(self._test_samples)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(self, batch_size=None, shuffle=True):
        return self._make_loader(self._train_samples, self._train_is_last,
                                 batch_size or self.cfg.batch_size, shuffle)

    def val(self, batch_size=None):
        bs = batch_size or self.cfg.batch_size
        if bs == -1:
            return self._load_all(self._val_samples, self._val_is_last)
        return self._make_loader(self._val_samples, self._val_is_last, bs, False)

    def test(self, batch_size=None):
        bs = batch_size or self.cfg.batch_size
        if bs == -1:
            return self._load_all(self._test_samples, self._test_is_last)
        return self._make_loader(self._test_samples, self._test_is_last, bs, False)

    @property
    def train_samples(self) -> list[Sample]: return self._train_samples
    @property
    def val_samples(self) -> list[Sample]: return self._val_samples
    @property
    def test_samples(self) -> list[Sample]: return self._test_samples

    @property
    def num_channels(self) -> int:
        return self._total_channels

    @property
    def channel_names(self) -> list[str]:
        """Ordered display names for every input channel (respects config)."""
        names: list[str] = []
        for field in self._input_fields:
            names.extend(FIELD_DISPLAY_NAMES[field.name])
        return names

    @property
    def input_groups(self) -> dict[str, list[int]]:
        """Channel indices per input field — for ablation."""
        return {name: list(range(s, e))
                for name, (s, e) in self._channel_ranges.items()}

    def load_sample(self, split: str, idx: int) -> tuple[np.ndarray, Sample]:
        """Load a single sample's input tensor by split name and index.

        Args:
            split: "train", "val", or "test".
            idx:   Index into that split's sample list.

        Returns:
            (x, sample) where x is (C, H, W) float32 numpy array.
        """
        samples = getattr(self, f"_{split}_samples")
        sample = samples[idx]
        data = self._load_sample(sample, idx)
        return data["x"], sample

    # -----------------------------------------------------------------------
    # Field selection
    # -----------------------------------------------------------------------

    @staticmethod
    def _select_input_fields(cfg: NextDayConfig) -> list[Field]:
        fields = []
        if cfg.include_accum_min:    fields.append(ALL_FIELDS["accum_t_min"])
        if cfg.include_cur_mask:     fields.append(ALL_FIELDS["cur_mask"])
        if cfg.include_accum_max:    fields.append(ALL_FIELDS["accum_t_max"])
        if cfg.include_accum_count:  fields.append(ALL_FIELDS["accum_t_count"])
        if cfg.include_terrain:      fields.append(ALL_FIELDS[cfg.terrain_type])
        if cfg.include_weather:      fields.append(ALL_FIELDS["weather"])
        if cfg.include_weather_pct:  fields.append(ALL_FIELDS["weather_pct"])
        if cfg.include_gfs:          fields.append(ALL_FIELDS["gfs"])
        if cfg.include_canopy:       fields.append(ALL_FIELDS["canopy_height"])
        if cfg.include_pos_encoding: fields.append(ALL_FIELDS["pos_encoding"])
        return fields

    # -----------------------------------------------------------------------
    # Splitting — pure temporal: three non-overlapping date ranges
    # -----------------------------------------------------------------------

    @staticmethod
    def _split_by_date(samples: list[Sample],
                       train_frac: float = 0.60, val_frac: float = 0.15,
                       ) -> tuple[list[int], list[int], list[int]]:
        """Stratified temporal split — per-region date ordering.

        Within each region, samples are sorted by date and split 60/15/25
        into train, val, and test.  This ensures every region has balanced
        representation across splits, avoiding spikes (e.g. N. NA 2023)
        from landing entirely in one split.

        Different regions may have different date boundaries — this is
        acceptable because the model sees local pixels + weather, not
        calendar dates.  Cross-region temporal leakage over 2018–2025 is
        negligible.
        """
        by_region: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            by_region[s.region_id].append(i)

        train, val, test = [], [], []
        for _rid, idxs in sorted(by_region.items()):
            idxs.sort(key=lambda i: samples[i].dt)
            n = len(idxs)
            t1 = int(n * train_frac)
            t2 = int(n * (train_frac + val_frac))
            train.extend(idxs[:t1])
            val.extend(idxs[t1:t2])
            test.extend(idxs[t2:])

        return train, val, test

    # -----------------------------------------------------------------------
    # Single-sample loading
    # -----------------------------------------------------------------------

    def _load_sample(self, sample: Sample, local_idx: int) -> dict:
        """Load one sample's fields + run per-field pipeline. Returns numpy dict."""
        raw = self._store.load(sample)

        # Fix accum_t / cur_mask inconsistency: cur_mask is derived from
        # VNP14 geocoordinates and may have fire pixels that the accumulation
        # (built from a separate detection pass) doesn't include.  Union
        # cur_mask into accum fields so they're consistent before _rel_t.
        _union_accum_cur_mask(raw)

        x_parts = []
        for field in self._input_fields:
            if field.storage == "per_tile":
                data = self._store.load_tile(sample.tile_key, field.name)
            elif field.name in raw:
                data = raw[field.name]
            else:
                data = np.zeros((field.channels, field.native_size,
                                 field.native_size), dtype=np.float32)
            x_parts.append(load_field(field, data, self._stats, target_size=256))
        x = np.concatenate(x_parts, axis=0)                  # (C, 256, 256)

        y = raw[self._target_field.name].astype(np.float32)  # (1, 256, 256)

        vnp14_t = raw["vnp14_t"]    # (1, H, W) uint8
        vnp14_t1 = raw["vnp14_t1"]  # (1, H, W) uint8
        ign_t1 = raw.get("ignition_t1")  # (1, H, W) uint8 or None
        fid_mask = raw.get("fire_id_mask_t1")  # (1, H, W) uint8 or None
        return {"x": x, "y": y, "vnp14_t": vnp14_t, "vnp14_t1": vnp14_t1,
                "ignition_t1": ign_t1, "fire_id_mask_t1": fid_mask,
                "local_idx": local_idx}

    # -----------------------------------------------------------------------
    # Collate + loss mask + device transfer
    # -----------------------------------------------------------------------

    def _collate(self, items: list[dict], samples: list[Sample],
                 is_last_day: torch.Tensor) -> Batch:
        """Collate into a Batch on CPU. Device transfer happens in the
        DataLoader wrapper (_DeviceLoader) so it works with num_workers>0
        (CUDA can't be initialized in forked subprocesses)."""
        x = torch.from_numpy(np.stack([it["x"] for it in items]))
        y = torch.from_numpy(np.stack([it["y"] for it in items]))
        idx = torch.tensor([it["local_idx"] for it in items])

        vnp14_t = torch.from_numpy(
            np.stack([it["vnp14_t"] for it in items]))
        vnp14_t1 = torch.from_numpy(
            np.stack([it["vnp14_t1"] for it in items]))
        # ignition_t1 may be None for older datasets
        ign_arrs = [it["ignition_t1"] for it in items]
        ignition_t1 = (torch.from_numpy(np.stack(ign_arrs))
                       if ign_arrs[0] is not None else None)
        # fire_id_mask_t1 may be None for older datasets
        fid_arrs = [it["fire_id_mask_t1"] for it in items]
        fire_id_mask_t1 = (torch.from_numpy(np.stack(fid_arrs))
                           if fid_arrs[0] is not None else None)
        loss_mask = _compute_loss_mask_v3(
            vnp14_t=vnp14_t, vnp14_t1=vnp14_t1,
            ignition_t1=ignition_t1,
            fire_id_mask_t1=fire_id_mask_t1,
            is_last_day=is_last_day[idx], padding=self.cfg.padding,
        )

        return Batch(
            x=x, y=y, loss_mask=loss_mask, idx=idx,
            samples=[samples[i] for i in idx.tolist()],
            channel_names=self.channel_names,
        )

    def _make_loader(self, samples, is_last_day, batch_size, shuffle):
        torch_ds = _TorchDataset(self, samples)
        store = self._store

        def _worker_init(worker_id):
            store.reopen()

        loader = make_data_loader(
            torch_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda items: self._collate(items, samples, is_last_day),
            worker_init_fn=_worker_init,
        )
        return _DeviceLoader(loader, self.device)

    def _load_all(self, samples, is_last_day) -> Batch:
        items = [self._load_sample(s, i) for i, s in enumerate(samples)]
        return self._collate(items, samples, is_last_day).to(self.device)


# ---------------------------------------------------------------------------
# compute_loss_mask — public API for standalone use (e.g. solidity viz)
# ---------------------------------------------------------------------------

def compute_loss_mask(raw: dict[str, np.ndarray], *, padding: int = 16,
                      is_last_day: bool = False) -> np.ndarray:
    """Compute the v3 loss mask from raw H5 fields.

    Args:
        raw: dict of numpy arrays as returned by SampleStore.load().
             Must contain vnp14_t and vnp14_t1.
        padding: border pixels to exclude.
        is_last_day: if True, trust all observations.

    Returns:
        (H, W) float32 mask, 1 = compute loss, 0 = ignore.
    """
    ild = torch.tensor([1.0 if is_last_day else 0.0])

    vnp14_t = torch.from_numpy(raw["vnp14_t"]).unsqueeze(0)    # (1,1,H,W)
    vnp14_t1 = torch.from_numpy(raw["vnp14_t1"]).unsqueeze(0)
    ign_t1 = None
    if "ignition_t1" in raw and raw["ignition_t1"] is not None:
        ign_t1 = torch.from_numpy(raw["ignition_t1"]).unsqueeze(0)
    fid_mask = None
    if "fire_id_mask_t1" in raw and raw["fire_id_mask_t1"] is not None:
        fid_mask = torch.from_numpy(raw["fire_id_mask_t1"]).unsqueeze(0)
    mask = _compute_loss_mask_v3(vnp14_t, vnp14_t1, ild, padding,
                                 ignition_t1=ign_t1,
                                 fire_id_mask_t1=fid_mask)

    return mask.squeeze().numpy()  # (H, W)


# ---------------------------------------------------------------------------
# _compute_loss_mask_v3 — pure function
# ---------------------------------------------------------------------------

def _compute_loss_mask_v3(vnp14_t: torch.Tensor, vnp14_t1: torch.Tensor,
                          is_last_day: torch.Tensor,
                          padding: int,
                          ignition_t1: Optional[torch.Tensor] = None,
                          fire_id_mask_t1: Optional[torch.Tensor] = None,
                          ) -> torch.Tensor:
    """V3 loss mask using VNP14 observation masks for both days.

    Per-pixel loss mask. 1 = compute loss, 0 = ignore.

    VNP14 fire_mask values (full, no collapse):
        0-2: not-processed / bowtie / missing (no reliable observation)
        3:   water (clear observation)
        4:   cloud (can't see ground — not usable for fire/no-fire)
        5:   clear land
        6:   unclassified (ambiguous)
        7-9: fire (low / nominal / high confidence)
        255: no coverage (no source pixel)

    Observed = clear classification: water (3), land (5), or fire (7-9).
    Cloud and unclassified participate in projection (prevent fire
    bleeding) but are NOT treated as clear observations for loss.

    Loss is valid only where BOTH day_T and day_T+1 are observed,
    AND the pixel is not an unpredictable new ignition on day_T+1.

    Per-fire-ID masking (fire_id_mask_t1): when provided, fire pixels
    on day_T+1 that do NOT belong to the target fire are excluded from
    loss.  These "other fire" pixels remain in the input (cur_mask /
    accum_t) as context but are zeroed in the loss mask so the model
    is only supervised on the target fire's spread + non-fire pixels.

    Last-day override: on the fire's final sampled day, trust all pixels
    (same as v2 — if satellite didn't see fire, it's genuinely gone).
    """
    # observed = clear classification on each day (water=3, land=5, fire=7-9)
    def _clear(v):
        return ((v == 3) | (v == 5) | ((v >= 7) & (v <= 9))).float()
    obs_t = _clear(vnp14_t)     # (B, 1, H, W)
    obs_t1 = _clear(vnp14_t1)
    mask = obs_t * obs_t1

    # Exclude unpredictable new ignitions on day_T+1
    if ignition_t1 is not None:
        mask = mask * (ignition_t1 == 0).float()

    # Per-fire-ID masking: exclude other fires' pixels from loss.
    # A pixel is "other fire" if it's a fire pixel on T+1 (vnp14_t1 >= 7)
    # but doesn't belong to the target fire (fire_id_mask_t1 == 0).
    # Non-fire pixels (vnp14_t1 < 7) are always kept — the model must
    # learn to predict "no fire" on those.
    if fire_id_mask_t1 is not None:
        other_fire = ((vnp14_t1 >= 7) & (fire_id_mask_t1 == 0)).float()
        mask = mask * (1.0 - other_fire)

    # last-day override: trust everything
    last = is_last_day.view(-1, 1, 1, 1).float()
    mask = torch.where(last > 0, torch.ones_like(mask), mask)

    if padding > 0:
        mask[:, :, :padding, :] = 0
        mask[:, :, -padding:, :] = 0
        mask[:, :, :, :padding] = 0
        mask[:, :, :, -padding:] = 0
    return mask


# ---------------------------------------------------------------------------
# is_last_day
# ---------------------------------------------------------------------------

def _compute_is_last_day(samples: list[Sample]) -> torch.Tensor:
    """For each sample, True iff it's the chronologically last sample of its fire."""
    fire_samples: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i, s in enumerate(samples):
        fire_samples[s.fire_id].append((i, s.dt))

    is_last = torch.zeros(len(samples), dtype=torch.bool)
    for entries in fire_samples.values():
        entries.sort(key=lambda x: x[1])
        is_last[entries[-1][0]] = True
    return is_last


# ---------------------------------------------------------------------------
# Convenience loader — delegates to SampleStore
# ---------------------------------------------------------------------------

def load_all_samples(dataset_path: Path) -> list[Sample]:
    """Load all samples from a dataset directory.

    Thin wrapper around ``SampleStore.load_samples()`` for callers that
    need the sample list without constructing a full Dataset (e.g. patch,
    distill).
    """
    store = SampleStore(dataset_path, sample_cls=Sample)
    samples = store.load_samples()
    store.close()
    return samples


# ---------------------------------------------------------------------------
# Subsampling — keep N samples by randomly selecting whole fires
# ---------------------------------------------------------------------------

def _subsample_by_fire(samples: list[Sample], max_samples: int,
                       seed: int = 42) -> list[Sample]:
    """
    Select a random subset of samples up to `max_samples`, keeping whole
    fires together (all timesteps of a fire stay or go as a unit).

    Replaces the old distill / distill_top pipeline — no separate datasets
    on disk, just deterministic subsampling at load time.
    """
    fire_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        fire_to_idx[s.fire_id].append(i)

    fire_ids = list(fire_to_idx)
    rng = random.Random(seed)
    rng.shuffle(fire_ids)

    selected: list[int] = []
    for fid in fire_ids:
        if len(selected) >= max_samples:
            break
        selected.extend(fire_to_idx[fid])

    selected.sort()
    return [samples[i] for i in selected]


# ---------------------------------------------------------------------------
# torch Dataset adapter
# ---------------------------------------------------------------------------

class _TorchDataset(torch.utils.data.Dataset):
    def __init__(self, parent: NextDayDataset, samples: list[Sample]):
        self._parent = parent
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._parent._load_sample(self._samples[idx], idx)


class _DeviceLoader:
    """Wraps a DataLoader to transfer batches to GPU.

    Collate runs in worker subprocesses (CPU tensors) — we do the .to(device)
    here because CUDA can't be initialised in forked workers.
    """

    def __init__(self, loader, device):
        self._loader = loader
        self._device = device

    def __iter__(self):
        for batch in self._loader:
            yield batch.to(self._device)

    def __len__(self):
        return len(self._loader)
