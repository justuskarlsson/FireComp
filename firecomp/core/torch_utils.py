"""
core/torch_utils.py — Small torch helpers.

Replaces scattered device/precision setup from both task train() functions.
"""

import torch


def get_device(preference: str = "cuda") -> torch.device:
    """Resolve device — crash if CUDA is requested but unavailable.

    Never silently falls back to CPU when CUDA is requested, because
    CPU training is ~100× slower and wastes hours of cluster time.
    """
    if preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preference)
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_precision(device: torch.device):
    """Set matmul + conv precision for Ampere+ GPUs. Call once at script start."""
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")       # TF32 for matmuls
        torch.backends.cudnn.allow_tf32 = True            # TF32 for convolutions
        torch.backends.cuda.matmul.allow_tf32 = True      # explicit (redundant w/ "high")
