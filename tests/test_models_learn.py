"""
tests/test_models_learn.py — Can each Paper 1 model learn a toy segmentation task?

Generates a tiny synthetic dataset (circles on noise), trains each model for
a handful of steps on GPU, and checks that:
  1. Loss drops well below the initial value.
  2. IoU on the training batch exceeds a minimum threshold.

GPU-only — skipped when CUDA is unavailable.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.optim import Adam

from firecomp.models.segmentation_models import model_factory

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)

# ---------------------------------------------------------------------------
# Synthetic data — circles on random noise
# ---------------------------------------------------------------------------

def _make_circle_batch(batch: int, channels: int, size: int,
                       device: str = "cuda"):
    """Return (x, mask) where mask has a random circle per sample.

    The first channel of x contains the mask + noise so the task is
    trivially learnable; remaining channels are pure noise.
    """
    x = torch.randn(batch, channels, size, size, device=device) * 0.1
    mask = torch.zeros(batch, 1, size, size, device=device)

    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    for i in range(batch):
        cx = torch.randint(size // 4, 3 * size // 4, (1,)).item()
        cy = torch.randint(size // 4, 3 * size // 4, (1,)).item()
        r = torch.randint(size // 8, size // 4, (1,)).item()
        circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        mask[i, 0] = circle.float()

    # leak the answer into the first channel so the model can learn fast
    x[:, 0:1] += mask * 2.0
    return x, mask


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

MODELS = [
    ("unet",   None),
    ("unet++", "resnet18"),
    ("vit",    None),
]

IN_CHANNELS = 6
IMG_SIZE = 128  # smaller than 256 — faster, still exercises the arch
BATCH = 8
STEPS = 40


@pytest.mark.parametrize("model_type,encoder_name", MODELS)
def test_model_learns_circles(model_type, encoder_name):
    """Train on a tiny circle-segmentation task; loss must drop, IoU must rise."""
    device = "cuda"

    # --- build model ---
    kwargs = {"in_channels": IN_CHANNELS, "out_channels": 1, "encoder_weights": None}
    if encoder_name is not None:
        kwargs["encoder_name"] = encoder_name
    model = model_factory[model_type](**kwargs).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # --- fixed training batch (overfit it) ---
    torch.manual_seed(42)
    x, mask = _make_circle_batch(BATCH, IN_CHANNELS, IMG_SIZE, device)

    # --- train ---
    initial_loss = None
    for step in range(STEPS):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, mask)
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # --- evaluate ---
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum() - intersection
        iou = (intersection / union.clamp(min=1)).item()

    assert final_loss < initial_loss * 0.5, (
        f"{model_type}: loss didn't drop enough "
        f"({initial_loss:.4f} -> {final_loss:.4f})"
    )
    assert iou > 0.5, (
        f"{model_type}: IoU too low ({iou:.3f}), model didn't learn the task"
    )
