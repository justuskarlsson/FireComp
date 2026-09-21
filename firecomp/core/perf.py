"""
core/perf.py — Lightweight GPU-aware performance timer.

Usage in a training loop:

    timer = PerfTimer(device, enabled=cfg.perf)

    for epoch in range(num_epochs):
        timer.reset()
        for batch in loader:
            timer.stamp("data")

            logits = model(batch.x)
            loss = loss_fn(logits, batch.y)
            timer.stamp("fwd")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            timer.stamp("bwd")

        print(timer.summary())
        # -> data=45.2s  fwd=8.1s  bwd=12.3s  total=65.6s

When ``enabled=False``, all methods are no-ops (zero overhead).
Handles cuda.synchronize() internally so the call sites stay clean.
"""

import time

import torch


class PerfTimer:
    """Accumulates wall-clock time across named sections within an epoch.

    Call ``stamp(key)`` at section boundaries.  Each call records the
    elapsed time since the previous stamp (or since ``reset()``) under
    ``key``.  On CUDA devices, synchronizes before reading the clock so
    async kernel launches don't skew the numbers.

    When ``enabled=False``, all methods are no-ops — no syncs, no timing,
    ``summary()`` returns ``""``.
    """

    def __init__(self, device: torch.device | str = "cpu",
                 enabled: bool = True):
        self._enabled = enabled
        self._cuda = enabled and str(device).startswith("cuda")
        self._totals: dict[str, float] = {}
        self._t0: float = 0.0
        if enabled:
            self.reset()

    def reset(self):
        """Clear all accumulators. Call at the start of each epoch."""
        if not self._enabled:
            return
        self._totals.clear()
        self._sync()
        self._t0 = time.monotonic()

    def stamp(self, key: str):
        """Record elapsed time since last stamp under ``key``."""
        if not self._enabled:
            return
        self._sync()
        now = time.monotonic()
        self._totals[key] = self._totals.get(key, 0.0) + (now - self._t0)
        self._t0 = now

    def summary(self) -> str:
        """One-line summary: ``data=12.3s  fwd=4.5s  bwd=6.7s  total=23.5s``."""
        if not self._enabled:
            return ""
        parts = [f"{k}={v:.1f}s" for k, v in self._totals.items()]
        total = sum(self._totals.values())
        parts.append(f"total={total:.1f}s")
        return "  ".join(parts)

    def _sync(self):
        if self._cuda:
            torch.cuda.synchronize()
