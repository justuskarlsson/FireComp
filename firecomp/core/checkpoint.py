"""
core/checkpoint.py — Lightweight checkpoint tracking.

Just enough boilerplate to keep "save best" out of the training loop body.
Not a framework — the training script still owns the for-loop and the
condition for what counts as "best".

Usage:
    ckpt = BestCheckpoint(run_dir, metric_name="f1")
    for epoch in range(num_epochs):
        ...
        if ckpt.update(model, score=m.f1, epoch=epoch, threshold=t, cfg=cfg):
            print("  >> new best")
    best = ckpt.load()  # dict: {model, epoch, threshold, cfg, ...}
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BestCheckpoint:
    """
    Track best score across epochs, save a single checkpoint.

    The training script calls update() each epoch with the current score.
    If it beats the previous best, the model state_dict + extra metadata is
    saved to {run_dir}/best.pt and update() returns True.

    Loading later: BestCheckpoint(run_dir).load() returns the saved dict.
    """
    run_dir: Path | str
    metric_name: str = "f1"
    higher_is_better: bool = True

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._best: float = float("-inf") if self.higher_is_better else float("inf")
        self._best_epoch: int = -1
        self._path = self.run_dir / "best.pt"

    @property
    def best_score(self) -> float:
        return self._best

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def path(self) -> Path:
        return self._path

    def update(self, model: torch.nn.Module, score: float, epoch: int,
               **extras) -> bool:
        """
        Save model + extras if score is the new best. Returns True if saved.

        Extras are persisted alongside the state_dict (e.g. threshold, cfg).
        """
        better = score > self._best if self.higher_is_better else score < self._best
        if not better:
            return False

        self._best = score
        self._best_epoch = epoch
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            self.metric_name: score,
            **extras,
        }, self._path)
        return True

    def load(self, map_location=None) -> dict:
        """Load the saved best.pt (raises FileNotFoundError if missing)."""
        return torch.load(self._path, map_location=map_location, weights_only=False)
