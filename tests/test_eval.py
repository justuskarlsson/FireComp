"""
Eval tests: run eval() on a saved checkpoint from test_train_smoke.

Relies on data/runs/next_day_train_5ep/best.pt existing — created by
test_train_5_epochs. Run that first if the checkpoint is missing.

Run:  pytest tests/test_eval.py -v -s
"""

from pathlib import Path

import pytest

DATASET_DIR = Path("data/tasks/next_day/100")
H5_FILE = DATASET_DIR / "dataset_0.h5"
CHECKPOINT = Path("data/runs/next_day_train_5ep/best.pt")

pytestmark = pytest.mark.skipif(
    not H5_FILE.exists(),
    reason="Desktop dataset not available",
)


def _ensure_checkpoint():
    """Train 5 epochs if checkpoint doesn't exist yet."""
    if CHECKPOINT.exists():
        return
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
    train(cfg)


def test_eval_loads_and_runs():
    """eval() loads checkpoint, runs test split, prints metrics without crash."""
    _ensure_checkpoint()
    from firecomp.next_day.implementations.dl_2d import eval

    eval(CHECKPOINT)


def test_eval_regions_not_unknown():
    """Region grouping should produce real names, not 'unknown_0'."""
    _ensure_checkpoint()

    import torch
    from firecomp.core.checkpoint import BestCheckpoint
    from firecomp.core.metrics import Metrics
    from firecomp.core.regions import Regions
    from firecomp.core.torch_utils import get_device
    from firecomp.next_day.dataset import NextDayDataset
    from firecomp.next_day.implementations.dl_2d import _build_fire_regions
    from firecomp.models.segmentation_models import model_factory

    device = get_device()
    ckpt = BestCheckpoint(CHECKPOINT.parent).load(map_location=device)
    cfg = ckpt["cfg"]
    threshold = ckpt["threshold"]

    ds = NextDayDataset(cfg)
    model = model_factory[cfg.model_type](
        in_channels=ds.num_channels,
        out_channels=1,
        encoder_name=cfg.encoder_name,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_preds, test_ys, test_masks = [], [], []
    with torch.no_grad():
        for batch in ds.test():
            pred = torch.sigmoid(model(batch.x))
            test_preds.append(pred)
            test_ys.append(batch.y)
            test_masks.append(batch.loss_mask)

    m = Metrics(
        torch.cat(test_preds), torch.cat(test_ys),
        torch.cat(test_masks), threshold=threshold,
    )

    fire_regions = _build_fire_regions(ds.test_samples)
    assert fire_regions is not None, "RegionRaster should be available on desktop"

    by_region = Regions.group(m, ds.test_samples, fire_regions=fire_regions)

    # No region should be "unknown_*"
    for name in by_region:
        assert not name.startswith("unknown"), f"Got unknown region: {name}"

    # Should have at least 2 distinct regions in a global 100-fire dataset
    assert len(by_region) >= 2, f"Expected >=2 regions, got {list(by_region.keys())}"

    print(f"\n{len(by_region)} regions: {sorted(by_region.keys())}")
    for name, rm in sorted(by_region.items(), key=lambda x: x[1].f1, reverse=True):
        print(f"  {name:20s}  F1={rm.f1:.3f}  n={rm.n_samples}")


def test_eval_fire_sizes():
    """Fire size grouping should not crash and produce at least one bucket."""
    _ensure_checkpoint()
    from firecomp.next_day.implementations.dl_2d import eval

    # Just run eval — it prints fire sizes. No crash = pass.
    eval(CHECKPOINT)
