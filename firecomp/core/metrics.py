"""
core/metrics.py — One-call, immutable metrics.

Replaces:
  - general/metrics.py        (493 lines, accumulator-style Metrics + SampleMetrics)
  - general/evaluation.py     (MetricsAnalysis subset, find_optimal_threshold)

Design:
  m = Metrics(pred, target, mask)
  m.f1, m.precision, m.recall, m.iou, m.auc
  m.per_sample -> list[SampleMetrics]

  No .update(). No .reset(). No state. Give it tensors, get results.
  Works for both spatial (B, 1, H, W) and scalar (B,) predictions.
"""

from dataclasses import dataclass

import torch
import numpy as np


# ---------------------------------------------------------------------------
# SampleMetrics — per-sample confusion matrix
# ---------------------------------------------------------------------------

@dataclass
class SampleMetrics:
    """Per-sample confusion counts + derived metrics. Immutable."""
    idx: int
    tp: int
    fp: int
    fn: int
    tn: int
    brier_sum: float = 0.0   # sum of (prob - target)² over valid pixels
    brier_n: int = 0         # number of valid pixels (for micro-average)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def iou(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def brier(self) -> float:
        return self.brier_sum / self.brier_n if self.brier_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Metrics — the main class
# ---------------------------------------------------------------------------

class Metrics:
    """
    Immutable evaluation result computed in one pass.

    Handles both segmentation and classification:
      - segmentation: pred/target are (B, 1, H, W), mask is (B, 1, H, W)
      - classification: pred/target are (B,), mask/weights are (B,) or None

    Usage:
        m = Metrics(pred, target)
        m = Metrics(pred, target, mask, threshold=0.4)
        print(m.f1, m.precision, m.recall)
        print(m)  # Metrics(F1=0.523, P=0.612, R=0.456, n=1842)

    For region/size grouping, use m.per_sample with Regions.group() or
    FireSize.group().
    """

    def __init__(self, pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor = None, threshold: float = 0.5,
                 compute_auc: bool = False):
        """
        Args:
            pred:        predicted probabilities (after sigmoid)
            target:      ground truth (binary)
            mask:        optional. 1 = include, 0 = ignore.
                         spatial (B, 1, H, W) for segmentation.
                         scalar (B,) for per-sample weights (values > 0 = include).
            threshold:   binarization threshold for predictions
            compute_auc: if True, compute AUC (expensive for large spatial
                         tensors — 15M+ elements through sklearn).  Off by
                         default; enable explicitly for paper figures.
        """
        with torch.no_grad():
            pred = pred.detach()
            target = target.detach()

            # flatten spatial dims if present: (B, ...) -> (B, N)
            B = pred.shape[0]
            is_spatial = pred.ndim > 1
            pred_flat = pred.reshape(B, -1)           # (B, N)
            target_flat = target.reshape(B, -1)       # (B, N)

            if mask is not None:
                mask_flat = mask.detach().reshape(B, -1).bool()
            else:
                mask_flat = torch.ones_like(pred_flat, dtype=torch.bool)

            # binarize
            pred_pos = pred_flat > threshold
            true_pos = target_flat > 0.5
            valid = mask_flat

            # per-sample confusion: (B,)
            tp = (pred_pos & true_pos & valid).sum(dim=1)
            fp = (pred_pos & ~true_pos & valid).sum(dim=1)
            fn = (~pred_pos & true_pos & valid).sum(dim=1)
            tn = (~pred_pos & ~true_pos & valid).sum(dim=1)

            # Brier score: mean squared error on valid pixels (before binarization)
            sq_err = (pred_flat - target_flat) ** 2
            self._brier_sum_per = (sq_err * mask_flat.float()).sum(dim=1)  # (B,)
            self._brier_n_per = mask_flat.sum(dim=1)                      # (B,)

            # filter out samples with no valid predictions at all
            has_content = (tp + fp + fn) > 0
            self._tp = tp[has_content]
            self._fp = fp[has_content]
            self._fn = fn[has_content]
            self._tn = tn[has_content]
            self._all_tp = tp   # unfiltered, for per_sample
            self._all_fp = fp
            self._all_fn = fn
            self._all_tn = tn

            # aggregate (micro-average)
            sum_tp = self._tp.sum().item()
            sum_fp = self._fp.sum().item()
            sum_fn = self._fn.sum().item()
            sum_tn = self._tn.sum().item()

            self._precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
            self._recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
            self._f1 = (2 * self._precision * self._recall / (self._precision + self._recall)
                        if (self._precision + self._recall) > 0 else 0.0)
            self._iou = (sum_tp / (sum_tp + sum_fp + sum_fn)
                         if (sum_tp + sum_fp + sum_fn) > 0 else 0.0)
            self._n_samples = int(self._tp.numel())

            # Brier: micro-average over all valid pixels (filtered samples only)
            brier_total = self._brier_sum_per[has_content].sum().item()
            brier_count = self._brier_n_per[has_content].sum().item()
            self._brier = brier_total / brier_count if brier_count > 0 else 0.0

            # AUC: expensive for spatial data (GPU→CPU copy + sklearn).
            # Threshold-independent, so skip during threshold sweeps.
            if compute_auc:
                self._auc = self._compute_auc(pred_flat, target_flat, mask_flat)
            else:
                self._auc = 0.0

    # --- public properties ---

    @property
    def f1(self) -> float: return self._f1
    @property
    def precision(self) -> float: return self._precision
    @property
    def recall(self) -> float: return self._recall
    @property
    def iou(self) -> float: return self._iou
    @property
    def auc(self) -> float: return self._auc
    @property
    def brier(self) -> float: return self._brier
    @property
    def n_samples(self) -> int: return self._n_samples

    @property
    def per_sample(self) -> list[SampleMetrics]:
        """Per-sample metrics for grouping (by region, fire size, etc.)."""
        return [
            SampleMetrics(
                idx=i,
                tp=self._all_tp[i].item(),
                fp=self._all_fp[i].item(),
                fn=self._all_fn[i].item(),
                tn=self._all_tn[i].item(),
                brier_sum=self._brier_sum_per[i].item(),
                brier_n=self._brier_n_per[i].item(),
            )
            for i in range(self._all_tp.shape[0])
        ]

    def to_dict(self) -> dict:
        return {
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "iou": round(self.iou, 4),
            "brier": round(self.brier, 4),
            "auc": round(self.auc, 4),
            "n_samples": self.n_samples,
        }

    def __repr__(self):
        return (f"Metrics(F1={self.f1:.3f}, P={self.precision:.3f}, "
                f"R={self.recall:.3f}, Brier={self.brier:.4f}, n={self.n_samples})")

    # --- internals ---

    @staticmethod
    def _compute_auc(pred_flat, target_flat, mask_flat):
        """ROC AUC from flattened predictions. Returns 0 if not computable."""
        try:
            # pool all valid predictions across all samples
            valid = mask_flat.any(dim=0) if mask_flat.ndim == 2 else mask_flat
            p = pred_flat[mask_flat].cpu().numpy()
            t = target_flat[mask_flat].cpu().numpy()
            if len(np.unique(t)) < 2:
                return 0.0
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(t, p))
        except Exception:
            return 0.0

    @staticmethod
    def pixel_confusion(
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """Per-pixel TP/FP/FN/valid boolean arrays for visualization.

        All inputs/outputs are numpy (H, W).  Use this for figure overlays
        so they match the same masking logic as training evaluation.

        Returns:
            {"tp": (H,W) bool, "fp": ..., "fn": ..., "valid": ...}
        """
        pred_bin = pred > threshold
        gt_bin = target > 0.5
        valid = mask > 0.5 if mask is not None else np.ones_like(pred, dtype=bool)
        return {
            "tp": pred_bin & gt_bin & valid,
            "fp": pred_bin & ~gt_bin & valid,
            "fn": ~pred_bin & gt_bin & valid,
            "valid": valid,
        }

    @classmethod
    def from_subset(cls, per_sample: list[SampleMetrics]) -> "Metrics":
        """
        Reconstruct Metrics from a subset of per-sample results.
        Used by Regions.group and FireSize.group after filtering samples.

        This avoids re-running the model — just recompute aggregates from
        the per-sample confusion counts.
        """
        m = cls.__new__(cls)
        if not per_sample:
            m._f1 = m._precision = m._recall = m._iou = m._auc = m._brier = 0.0
            m._n_samples = 0
            m._all_tp = m._all_fp = m._all_fn = m._all_tn = torch.tensor([])
            m._tp = m._fp = m._fn = m._tn = torch.tensor([])
            m._brier_sum_per = torch.tensor([])
            m._brier_n_per = torch.tensor([])
            return m

        tp = torch.tensor([s.tp for s in per_sample])
        fp = torch.tensor([s.fp for s in per_sample])
        fn = torch.tensor([s.fn for s in per_sample])
        tn = torch.tensor([s.tn for s in per_sample])

        m._all_tp, m._all_fp, m._all_fn, m._all_tn = tp, fp, fn, tn
        m._brier_sum_per = torch.tensor([s.brier_sum for s in per_sample])
        m._brier_n_per = torch.tensor([s.brier_n for s in per_sample])

        has_content = (tp + fp + fn) > 0
        m._tp = tp[has_content]
        m._fp = fp[has_content]
        m._fn = fn[has_content]
        m._tn = tn[has_content]

        sum_tp = m._tp.sum().item()
        sum_fp = m._fp.sum().item()
        sum_fn = m._fn.sum().item()

        m._precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        m._recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        m._f1 = (2 * m._precision * m._recall / (m._precision + m._recall)
                 if (m._precision + m._recall) > 0 else 0.0)
        m._iou = (sum_tp / (sum_tp + sum_fp + sum_fn)
                  if (sum_tp + sum_fp + sum_fn) > 0 else 0.0)
        m._n_samples = int(m._tp.numel())
        m._auc = 0.0   # can't recompute AUC from confusion counts alone

        # Brier: micro-average from per-sample components (filtered samples)
        brier_total = m._brier_sum_per[has_content].sum().item()
        brier_count = m._brier_n_per[has_content].sum().item()
        m._brier = brier_total / brier_count if brier_count > 0 else 0.0
        return m


# ---------------------------------------------------------------------------
# Threshold search — standalone function
# ---------------------------------------------------------------------------

def find_optimal_threshold(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor = None,
                           thresholds: list[float] = None,
                           compute_auc: bool = False) -> tuple[float, "Metrics"]:
    """
    Sweep thresholds, return (best_threshold, best_metrics).
    No model forward pass — just tensor ops on existing predictions.

    Uses a vectorised numpy sweep: flatten valid pixels once, bucket
    prediction values into threshold bins, then compute cumulative TP/FP/FN
    across all thresholds in one pass.  ~1 s for 1.6 B elements on CPU
    (vs ~100 s for the naive 99× Metrics loop).

    Args:
        compute_auc: compute AUC at the best threshold. Default False — AUC
                     is expensive on large spatial tensors (GPU→CPU + sklearn)
                     and threshold-independent, so skip during training val.

    Usage:
        val = ds.val(batch_size=-1)
        pred = torch.sigmoid(model(val.x))
        threshold, m = find_optimal_threshold(pred, val.y, val.loss_mask)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]

    best_threshold = _fast_threshold_sweep(pred, target, mask, thresholds)

    best_metrics = Metrics(pred, target, mask, threshold=best_threshold,
                           compute_auc=compute_auc)
    return best_threshold, best_metrics


def _fast_threshold_sweep(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    thresholds: list[float],
) -> float:
    """Find the threshold that maximises micro-averaged F1.

    Instead of constructing 99 Metrics objects (each doing full-tensor ops),
    we:
      1. Flatten to valid pixels only (drops ~50-90% of data).
      2. Bucket predictions into threshold bins via np.digitize (one pass).
      3. Compute cumulative TP/FP/FN with np.cumsum (one pass).
      4. Vectorised F1 across all thresholds simultaneously.

    Total: ~2 full reads of the valid-pixel array.  For 1.6B elements with
    ~20% valid, that's ~2 × 320M × 4 bytes ≈ 2.5 GB — a few seconds on CPU.
    """
    with torch.no_grad():
        p = pred.detach().reshape(-1)
        t = target.detach().reshape(-1)
        if mask is not None:
            m = mask.detach().reshape(-1).bool()
            p = p[m]
            t = t[m]

    # Move to numpy (zero-copy if already contiguous CPU)
    p_np = p.numpy() if p.is_cpu else p.cpu().numpy()
    t_np = t.numpy() if t.is_cpu else t.cpu().numpy()
    del p, t

    t_bool = t_np > 0.5  # ground truth positive

    # thresholds sorted ascending; np.digitize gives bin index
    # bin 0 = pred < thresholds[0], bin k = pred >= thresholds[k-1]
    thresh_arr = np.array(thresholds, dtype=np.float32)
    bins = np.digitize(p_np, thresh_arr)  # 0..len(thresholds)
    del p_np

    n_bins = len(thresholds) + 1  # one extra bin for pred >= max threshold

    # Count positives and negatives per bin
    pos_counts = np.bincount(bins[t_bool], minlength=n_bins).astype(np.int64)
    neg_counts = np.bincount(bins[~t_bool], minlength=n_bins).astype(np.int64)
    del bins, t_bool, t_np

    # For threshold t[i], predicted-positive = bins > i  (pred >= t[i])
    # TP[i] = sum of pos_counts for bins > i = total_pos - cumsum_pos[i]
    # FP[i] = sum of neg_counts for bins > i = total_neg - cumsum_neg[i]
    cum_pos = np.cumsum(pos_counts)  # cum_pos[i] = pos in bins 0..i
    cum_neg = np.cumsum(neg_counts)
    total_pos = cum_pos[-1]

    # For threshold index i: predicted positive = bins >= i+1
    # TP[i] = total_pos - cum_pos[i]
    # FP[i] = total_neg - cum_neg[i]
    # FN[i] = total_pos - TP[i] = cum_pos[i]
    tp = total_pos - cum_pos[:len(thresholds)]
    fp = cum_neg[-1] - cum_neg[:len(thresholds)]
    fn = cum_pos[:len(thresholds)]

    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)

    best_idx = int(np.argmax(f1))
    return thresholds[best_idx]
