"""
core/fire_stats.py — Fire size grouping, area utilities, wild/tame thresholds,
and fire classification.

FireSize.group() is the same pattern as Regions.group() — post-hoc grouping
of an existing Metrics object. No model runs, no accumulation.
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Wild / tame classification thresholds
# ---------------------------------------------------------------------------
# Used to exclude tame fires (crop burns, flares) from training.

WILD_XY_THRESH = 17.0           # avg_xy_neighbors ≥ this → wild
WILD_IGNITION_THRESH = 1.0      # ignition_ratio   ≤ this → wild
WILD_T_RATIO_THRESH = 0.75      # t_ratio          < this → wild (not oil-like)
TAME_XY_THRESH = 12.0           # avg_xy_neighbors ≤ this (crop combo)
TAME_IGNITION_THRESH = 2.0      # ignition_ratio   ≥ this (crop combo)
TAME_T_RATIO_THRESH = 1.5       # t_ratio          > this → oil
MIN_FIRE_COUNT = 100            # fires with fewer pixels are TOO_SMALL


# ---------------------------------------------------------------------------
# Fire size bucketing
# ---------------------------------------------------------------------------

SIZE_BUCKETS = {
    "tiny":   (0, 100),
    "small":  (100, 1_000),
    "medium": (1_000, 10_000),
    "large":  (10_000, float("inf")),
}


def get_bucket(num_fire_pixels: int) -> str:
    """Classify a fire by its pixel count into a size bucket."""
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= num_fire_pixels < hi:
            return name
    return "large"


class FireSize:
    """
    Group metrics by fire size bucket.

    Usage:
        m = Metrics(pred, target, mask)
        by_size = FireSize.group(m, ds.test_samples, fire_stats=fire_stats)
        print(by_size["large"].f1)
    """

    @staticmethod
    def group(metrics, samples, fire_stats: dict[int, int]) -> dict[str, "Metrics"]:
        """
        Args:
            metrics:    a Metrics object
            samples:    list[Sample] with .fire_id attribute
            fire_stats: dict mapping fire_id -> num_fire_pixels

        Returns:
            {bucket_name: Metrics}
        """
        from firecomp.core.metrics import Metrics as M

        per_sample = metrics.per_sample
        assert len(per_sample) == len(samples)

        buckets = {}    # bucket_name -> list[SampleMetrics]
        for sm, sample in zip(per_sample, samples):
            fid = getattr(sample, "fire_id", None)
            n_pixels = fire_stats.get(fid, 0) if fid is not None else 0
            bucket = get_bucket(n_pixels)
            buckets.setdefault(bucket, []).append(sm)

        return {name: M.from_subset(sms) for name, sms in buckets.items()}


# ---------------------------------------------------------------------------
# Fire area utilities
# ---------------------------------------------------------------------------

def pixels_to_km2(num_pixels: int, pixel_size_m: float = 375.0) -> float:
    """Convert fire pixel count to area in km²."""
    return num_pixels * (pixel_size_m ** 2) / 1e6


# ---------------------------------------------------------------------------
# Fire classification — wild vs tame
# ---------------------------------------------------------------------------

class FireClassification:
    """Compute fire areas, classify tame/wild, and build size-weighted lookups.

    All methods are static — this is a namespace, not an instantiable class.
    Uses ``Fires.Stats`` and ``Fires.Projection`` from ``firecomp.dsrc.vnp14``.
    """

    @staticmethod
    def compute_spatial_areas(
        proj, stats, min_pixels: int = 5,
    ) -> dict[int, int]:
        """Count unique spatial (x, y) pixels per fire component.

        Packs (component, x, y) into int64 for fast np.unique.

        Args:
            proj: Fires.Projection with .component, .x, .y arrays.
            stats: Fires.Stats (unused internally, kept for interface compat).
            min_pixels: minimum unique-pixel threshold per component.

        Returns:
            {component_id: unique_pixel_count} for components ≥ min_pixels.
        """
        c = proj.component

        if min_pixels > 1:
            raw_counts = np.bincount(c, minlength=int(c.max()) + 1)
            keep = raw_counts[c] >= min_pixels
            x_f, y_f, c_f = proj.x[keep], proj.y[keep], c[keep]
        else:
            x_f, y_f, c_f = proj.x, proj.y, c

        keys = (c_f.astype(np.int64) << 33
                | x_f.astype(np.int64) << 16
                | y_f.astype(np.int64))
        unique_keys = np.unique(keys)

        unique_comps = (unique_keys >> 33).astype(np.int32)
        components, counts = np.unique(unique_comps, return_counts=True)

        mask = counts >= min_pixels
        return dict(zip(components[mask].tolist(), counts[mask].tolist()))

    @staticmethod
    def areas_to_km2(
        fire_areas: "dict[int, int] | np.ndarray", stats,
    ) -> np.ndarray:
        """Convert pixel counts to km² accounting for latitude.

        Args:
            fire_areas: dict {component_id: pixel_count} or dense array.
            stats: Fires.Stats with .id, .min_y, .max_y.

        Returns:
            (N,) float64 array of km² values, one per fire in stats.
        """
        from firecomp.dsrc.vnp14 import DEG_CELL_SIZE

        KM_PER_DEG = 111.0
        lats = (stats.min_y + stats.max_y) / 2
        if isinstance(fire_areas, np.ndarray):
            ids = stats.id.astype(np.int32)
            areas_px = np.where(
                ids < len(fire_areas),
                fire_areas[np.clip(ids, 0, len(fire_areas) - 1)], 0,
            ).astype(np.int32)
        else:
            areas_px = np.array(
                [fire_areas.get(int(cid), 0) for cid in stats.id], dtype=np.int32)

        pixel_area = (DEG_CELL_SIZE * KM_PER_DEG) * (
            DEG_CELL_SIZE * KM_PER_DEG * np.cos(np.radians(lats)))
        return areas_px * pixel_area

    @staticmethod
    def identify_tame(stats) -> np.ndarray:
        """Identify tame fires (crop burns, oil flares).

        Oil: t_ratio > 1.5 (temporal signature of oil flares).
        Crop: xy ≤ 12 AND ignition_ratio ≥ 2% (spatially compact,
              many separate ignition points).

        NOTE: The deprecated tame_fires.py compared ignition_ratio (a 0-1
        fraction) against TAME_IGNITION_THRESH=2.0 WITHOUT ``*100``, so the
        crop criterion was effectively dead code (never true).  The ``*100``
        fix here makes crop burns actually classifiable as tame.

        Returns:
            (N,) bool array — True for tame fires.
        """
        is_tame = np.zeros(len(stats.id), dtype=bool)
        has_enough = stats.num_fire >= MIN_FIRE_COUNT
        is_oil = stats.t_ratio > TAME_T_RATIO_THRESH
        is_crop = ((stats.avg_xy_neighbors <= TAME_XY_THRESH)
                   & (stats.ignition_ratio * 100 >= TAME_IGNITION_THRESH))
        is_tame[has_enough] = (is_oil | is_crop)[has_enough]
        return is_tame

    @staticmethod
    def identify_wild(
        stats, is_tame: np.ndarray, areas_km2: np.ndarray,
        min_area_km2: float = 1.0,
    ) -> np.ndarray:
        """Identify confident wild fires (above margin thresholds).

        Wild = ≥100 detections AND not tame AND xy≥17 AND t_ratio < 0.75
        AND area ≥ min.  Ambiguous fires fall into NOT_SURE margin and are
        excluded from loss.

        NOTE: The deprecated tame_fires.py also had an ignition_ratio filter
        but compared the raw fraction (0-1) against 1.0, making it always
        true — ignition was never used to filter wild fires.  We match that
        behaviour here for consistency with the existing dataset.

        Returns:
            (N,) bool array — True for confident wild fires.
        """
        has_enough = stats.num_fire >= MIN_FIRE_COUNT
        is_confident_wild = (
            (stats.avg_xy_neighbors >= WILD_XY_THRESH)
            & (stats.t_ratio < WILD_T_RATIO_THRESH)
        )
        return has_enough & ~is_tame & is_confident_wild & (areas_km2 >= min_area_km2)
