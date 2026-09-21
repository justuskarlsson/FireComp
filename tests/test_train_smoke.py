"""
Smoke test: run a short training loop on the 100-sample desktop dataset.

Verifies the full train() pipeline end-to-end without crashing:
  dataset -> model -> loss -> backward -> val metrics -> checkpoint -> eval

Uses max_batches=1 to run a single batch per epoch -- fast enough for CI.

Skipped if the dataset is not available.

Run:  pytest tests/test_train_smoke.py -v -s
"""

from pathlib import Path

import pytest

DATASET_DIR = Path("data/tasks/next_day/100/")
H5_FILE = DATASET_DIR / "v2" / "dataset_0.h5"

pytestmark = pytest.mark.skipif(
    not H5_FILE.exists(),
    reason="Desktop dataset not available",
)


def test_train_smoke():
    """1 batch x 3 epochs -- train loop, val, checkpoint, eval all run."""
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.dl_2d import train

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        device="cuda",
        num_workers=4,
        batch_size=8,
        num_epochs=3,
        lr=1e-3,
        model_type="unet",
        loss_type="focal",
        tag="smoke_test",
    )
    result = train(cfg, max_batches=1)

    # Pipeline ran without crashing and produced results
    assert len(result.losses) == 3
    assert result.checkpoint_path is not None


@pytest.mark.slow
def test_train_5_epochs():
    """Full 5-epoch run on 100-fire dataset -- verify loss decreases.

    With only 100 fires (~936 train samples), F1 is too noisy to assert on
    (fire is ~1% of valid pixels, threshold search is unstable).  The real
    learning signal is loss: it should drop meaningfully over 5 epochs.
    """
    from firecomp.next_day.config import NextDayConfig
    from firecomp.next_day.implementations.dl_2d import train

    cfg = NextDayConfig(
        dataset_dir=str(DATASET_DIR),
        dataset_version="v2",
        device="cuda",
        num_workers=4,
        batch_size=16,
        num_epochs=5,
        lr=1e-3,
        model_type="unet",
        loss_type="focal",
        tag="train_5ep",
    )
    result = train(cfg)

    # Loss must decrease: final epoch loss < first epoch loss
    assert (
        result.losses[-1] < result.losses[0]
    ), f"Loss did not decrease: {result.losses[0]:.4f} -> {result.losses[-1]:.4f}"
    # Should drop by at least 50% (typical: 0.18 -> 0.024 = 87% drop)
    drop = 1 - result.losses[-1] / result.losses[0]
    assert drop > 0.5, f"Loss only dropped {drop*100:.0f}% (expected >50%)"
