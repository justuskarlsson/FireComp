"""
core/ablation.py — Input-group zeroing ablation.

Not a framework. Just a tiny helper that runs the model with each input
group zeroed and reports F1 drop.

Usage in a training script:
    ab = Ablation(model, batch, ds.input_groups, threshold=0.4)
    for name, m in ab.results.items():
        print(f"{name}: F1={m.f1:.3f}  drop={m.f1_drop:+.3f}")

Replaces the per-channel-group zeroing block scattered at the bottom of
each task's train() function.
"""

from dataclasses import dataclass

import torch

from firecomp.core.metrics import Metrics


# ---------------------------------------------------------------------------
# AblationResult — single-group result
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """One group's ablation result. Immutable."""
    name: str
    metrics: Metrics
    baseline_f1: float

    @property
    def f1(self) -> float:
        return self.metrics.f1

    @property
    def f1_drop(self) -> float:
        """Drop in F1 when this group is zeroed (positive = group matters)."""
        return self.baseline_f1 - self.metrics.f1


# ---------------------------------------------------------------------------
# Ablation — run zero-channel ablation per input group
# ---------------------------------------------------------------------------

class Ablation:
    """
    Zero each input group, run forward pass, compute F1 drop.

    Args:
        model:         torch model in eval mode. forward(x) → logits
        batch:         a single Batch (.x, .y, optional .loss_mask) on device
        input_groups:  {group_name: [channel_indices]} from ds.input_groups
        threshold:     binarization threshold for predictions
        is_classification: if True, model output is squeezed to (B,) for scalar metrics
    """

    def __init__(self, model, batch, input_groups: dict[str, list[int]],
                 threshold: float = 0.5, is_classification: bool = False):
        self._results: dict[str, AblationResult] = {}

        with torch.no_grad():
            # baseline (no zeroing)
            pred = self._forward(model, batch.x, is_classification)
            mask = getattr(batch, "loss_mask", None)
            baseline = Metrics(pred, batch.y, mask, threshold=threshold)
            self._baseline = baseline

            # per-group zeroed
            for name, channels in input_groups.items():
                x_zeroed = batch.x.clone()
                x_zeroed[:, channels] = 0.0
                pred_z = self._forward(model, x_zeroed, is_classification)
                m = Metrics(pred_z, batch.y, mask, threshold=threshold)
                self._results[name] = AblationResult(name, m, baseline.f1)

    @staticmethod
    def _forward(model, x, is_classification: bool):
        logits = model(x)
        if is_classification and logits.ndim > 1:
            logits = logits.squeeze(1)
        return torch.sigmoid(logits)

    @property
    def results(self) -> dict[str, AblationResult]:
        return self._results

    @property
    def baseline(self) -> Metrics:
        return self._baseline

    def __repr__(self):
        rows = [f"  {r.name:20s}  F1={r.f1:.3f}  drop={r.f1_drop:+.3f}"
                for r in self._results.values()]
        return f"Ablation(baseline F1={self._baseline.f1:.3f})\n" + "\n".join(rows)
