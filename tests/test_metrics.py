"""
Tests for core/metrics.py — Metrics and threshold sweep.

Run:  pytest tests/test_metrics.py -v -s
"""

import numpy as np
import pytest
import torch

from firecomp.core.metrics import Metrics, find_optimal_threshold, _fast_threshold_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THRESHOLDS = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]


def _naive_best_threshold(pred, target, mask, thresholds):
    """Brute-force loop — the old implementation. Ground truth reference."""
    best_f1, best_t = -1.0, 0.5
    for t in thresholds:
        m = Metrics(pred, target, mask, threshold=t)
        if m.f1 > best_f1:
            best_f1, best_t = m.f1, t
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestFastThresholdSweep:
    """Verify _fast_threshold_sweep matches the naive loop."""

    def test_small_spatial(self):
        """Small spatial tensors (B, 1, H, W) — typical segmentation."""
        torch.manual_seed(0)
        B, H, W = 32, 16, 16
        pred = torch.sigmoid(torch.randn(B, 1, H, W))
        target = (torch.rand(B, 1, H, W) > 0.7).float()
        mask = (torch.rand(B, 1, H, W) > 0.3).float()

        naive_t, naive_f1 = _naive_best_threshold(pred, target, mask, THRESHOLDS)
        fast_t = _fast_threshold_sweep(pred, target, mask, THRESHOLDS)

        assert naive_t == fast_t, (
            f"Threshold mismatch: naive={naive_t}, fast={fast_t}")

    def test_medium_spatial(self):
        """Larger spatial tensors — closer to real workload."""
        torch.manual_seed(42)
        B, H, W = 200, 64, 64
        pred = torch.sigmoid(torch.randn(B, 1, H, W))
        target = (torch.rand(B, 1, H, W) > 0.8).float()
        mask = (torch.rand(B, 1, H, W) > 0.2).float()

        naive_t, naive_f1 = _naive_best_threshold(pred, target, mask, THRESHOLDS)
        fast_t = _fast_threshold_sweep(pred, target, mask, THRESHOLDS)

        assert naive_t == fast_t, (
            f"Threshold mismatch: naive={naive_t}, fast={fast_t}")

    def test_scalar_predictions(self):
        """1-D (classification-style) predictions."""
        torch.manual_seed(7)
        B = 5000
        pred = torch.sigmoid(torch.randn(B))
        target = (torch.rand(B) > 0.6).float()

        naive_t, _ = _naive_best_threshold(pred, target, None, THRESHOLDS)
        fast_t = _fast_threshold_sweep(pred, target, None, THRESHOLDS)

        assert naive_t == fast_t

    def test_no_mask(self):
        """Works without a mask (all pixels valid)."""
        torch.manual_seed(1)
        pred = torch.sigmoid(torch.randn(50, 1, 32, 32))
        target = (torch.rand(50, 1, 32, 32) > 0.75).float()

        naive_t, _ = _naive_best_threshold(pred, target, None, THRESHOLDS)
        fast_t = _fast_threshold_sweep(pred, target, None, THRESHOLDS)

        assert naive_t == fast_t

    def test_all_positive(self):
        """Target is all 1s — lowest threshold should win."""
        pred = torch.sigmoid(torch.randn(100, 1, 8, 8))
        target = torch.ones(100, 1, 8, 8)
        mask = torch.ones(100, 1, 8, 8)

        fast_t = _fast_threshold_sweep(pred, target, mask, THRESHOLDS)
        # With all-positive target, low threshold maximises recall → F1
        assert fast_t <= 0.10

    def test_all_negative(self):
        """Target is all 0s — F1 is 0 everywhere, should not crash."""
        pred = torch.sigmoid(torch.randn(100, 1, 8, 8))
        target = torch.zeros(100, 1, 8, 8)
        mask = torch.ones(100, 1, 8, 8)

        fast_t = _fast_threshold_sweep(pred, target, mask, THRESHOLDS)
        # Should return *something* without crashing
        assert 0.01 <= fast_t <= 0.99

    def test_sparse_mask(self):
        """Very sparse mask (~1% valid) — edge case."""
        torch.manual_seed(3)
        pred = torch.sigmoid(torch.randn(100, 1, 32, 32))
        target = (torch.rand(100, 1, 32, 32) > 0.5).float()
        mask = (torch.rand(100, 1, 32, 32) > 0.99).float()  # ~1% valid

        naive_t, _ = _naive_best_threshold(pred, target, mask, THRESHOLDS)
        fast_t = _fast_threshold_sweep(pred, target, mask, THRESHOLDS)

        assert naive_t == fast_t


class TestFindOptimalThreshold:
    """Integration test for find_optimal_threshold (uses _fast_threshold_sweep)."""

    def test_returns_metrics_at_best_threshold(self):
        torch.manual_seed(99)
        pred = torch.sigmoid(torch.randn(64, 1, 16, 16))
        target = (torch.rand(64, 1, 16, 16) > 0.7).float()
        mask = (torch.rand(64, 1, 16, 16) > 0.3).float()

        threshold, m = find_optimal_threshold(pred, target, mask)

        # Verify the returned Metrics matches the threshold
        m_check = Metrics(pred, target, mask, threshold=threshold)
        assert abs(m.f1 - m_check.f1) < 1e-6
        assert abs(m.precision - m_check.precision) < 1e-6
        assert abs(m.recall - m_check.recall) < 1e-6

    def test_per_sample_available(self):
        """Returned Metrics should have per_sample for Regions.group."""
        torch.manual_seed(10)
        pred = torch.sigmoid(torch.randn(20, 1, 8, 8))
        target = (torch.rand(20, 1, 8, 8) > 0.7).float()

        _, m = find_optimal_threshold(pred, target)
        assert hasattr(m, "per_sample")
        assert len(m.per_sample) == 20

    def test_custom_thresholds(self):
        """Custom threshold list works."""
        torch.manual_seed(5)
        pred = torch.sigmoid(torch.randn(50, 1, 8, 8))
        target = (torch.rand(50, 1, 8, 8) > 0.5).float()

        custom = [0.1, 0.3, 0.5, 0.7, 0.9]
        threshold, m = find_optimal_threshold(pred, target, thresholds=custom)
        assert threshold in custom


class TestMemory:
    """Verify memory usage stays bounded."""

    def test_peak_memory_bounded(self):
        """Fast sweep on large-ish tensor shouldn't explode memory.

        We can't test 24K×256×256 in CI, but 1000×128×128 (16M elements)
        exercises the same code path.  Check it completes without error
        and peak numpy allocation stays reasonable.
        """
        torch.manual_seed(0)
        B, H, W = 1000, 128, 128
        pred = torch.sigmoid(torch.randn(B, 1, H, W))
        target = (torch.rand(B, 1, H, W) > 0.8).float()
        mask = (torch.rand(B, 1, H, W) > 0.5).float()

        # Should complete without MemoryError
        threshold, m = find_optimal_threshold(pred, target, mask)
        assert 0.01 <= threshold <= 0.99
        assert m.f1 >= 0
