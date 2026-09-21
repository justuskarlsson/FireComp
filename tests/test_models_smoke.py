"""
tests/test_models_smoke.py — Model forward-pass smoke tests.

Exercises model_factory for all Paper 1 models (unet, unet++, vit) on both
CPU and GPU, with varying input channel counts.  No data files required.
"""

import pytest
import torch

from firecomp.models.segmentation_models import model_factory

IMG_SIZE = 256
BATCH = 2

PAPER1_MODELS = [
    ("unet",   None),
    ("unet++", "resnet18"),
    ("vit",    None),
]

CHANNEL_COUNTS = [5, 10]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# Core smoke test — model × channels × device
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("in_channels", CHANNEL_COUNTS)
@pytest.mark.parametrize("model_type,encoder_name", PAPER1_MODELS)
def test_forward_pass(model_type, encoder_name, in_channels, device):
    """Instantiate model, forward a random tensor, check output shape and finiteness."""
    kwargs = {"in_channels": in_channels, "out_channels": 1, "encoder_weights": None}
    if encoder_name is not None:
        kwargs["encoder_name"] = encoder_name

    model = model_factory[model_type](**kwargs).to(device)
    model.eval()

    x = torch.randn(BATCH, in_channels, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (BATCH, 1, IMG_SIZE, IMG_SIZE), (
        f"{model_type} on {device}: expected (2,1,256,256), got {out.shape}"
    )
    assert torch.isfinite(out).all(), f"{model_type} on {device}: output contains NaN/Inf"


# ---------------------------------------------------------------------------
# ViT ignores encoder_name — quick sanity check
# ---------------------------------------------------------------------------

def test_vit_encoder_name_ignored():
    """encoder_name is accepted but silently ignored for vit — always uses mit_b2."""
    model = model_factory["vit"](
        in_channels=10, out_channels=1,
        encoder_name="resnet18",  # should be ignored
        encoder_weights=None,
    )
    x = torch.randn(1, 10, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, IMG_SIZE, IMG_SIZE)
