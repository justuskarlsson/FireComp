"""
core/losses.py — Loss functions.

FocalLoss, DiceLoss, HybridLoss and the build_loss_fn factory.
Segmentation passes (B, 1, H, W) masks; the losses also accept (B,)
sample weights for image-level use.
"""

import torch
import torch.nn.functional as F
from typing import Literal


LossType = Literal["bce", "focal", "dice", "hybrid"]


# ---------------------------------------------------------------------------
# Loss classes — unchanged from current, they're fine
# ---------------------------------------------------------------------------

class FocalLoss:
    """Per-element focal loss for class imbalance.

    Uses alpha for class balancing (standard focal loss paper, Lin et al.):
      - alpha weights positive class, (1 - alpha) weights negative class.
    Does NOT use a separate pos_weight — alpha already fills that role.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        """Returns unreduced loss, same shape as input."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return alpha_t * focal_weight * bce


class DiceLoss:
    """Per-sample dice loss. Returns (B,)."""

    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask
            targets = targets * mask
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class HybridLoss:
    """Focal + Dice combined."""

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0,
                 bce_weight=0.3, dice_weight=0.7):
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def __call__(self, logits, targets, mask=None):
        """
        Returns (total_loss, focal_component, dice_component).
        Three values for logging. Only .backward() on total_loss.
        """
        focal_raw = self.focal(logits, targets)
        focal_loss = masked_reduce(focal_raw, mask)
        dice_loss = self.dice(logits, targets, mask).mean()
        total = self.bce_weight * focal_loss + self.dice_weight * dice_loss
        return total, focal_loss, dice_loss


# ---------------------------------------------------------------------------
# Masked reduction — shared utility
# ---------------------------------------------------------------------------

def masked_reduce(per_element_loss, mask=None):
    """
    Reduce loss with optional mask.

    mask can be:
      - None            -> simple mean
      - (B, 1, H, W)   -> spatial: sum(loss * mask) / sum(mask)
      - (B,)            -> per-sample weights: (loss.mean(spatial) * weights).mean()
    """
    if mask is None:
        return per_element_loss.mean()

    if mask.ndim == per_element_loss.ndim:
        # spatial mask: same shape as loss
        return (per_element_loss * mask).sum() / mask.sum().clamp(min=1)
    else:
        # per-sample weights: reduce spatial first, then weight
        spatial_dims = tuple(range(1, per_element_loss.ndim))
        per_sample = per_element_loss.mean(dim=spatial_dims)     # (B,)
        return (per_sample * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss_fn(loss_type: LossType = "hybrid", pos_weight=1.0,
                  focal_alpha=0.25, focal_gamma=2.0,
                  bce_weight=0.3, dice_weight=0.7):
    """
    Returns a callable: loss_fn(logits, targets, mask) -> loss tensor

    For hybrid, returns a tuple (total, focal, dice) for logging.
    For others, returns a scalar tensor.

    Class-balancing semantics differ by loss type:
      - bce:    uses pos_weight (PyTorch native — scales positive-class gradients)
      - focal:  uses alpha (from the original paper — alpha for pos, 1-alpha for neg)
      - hybrid: same as focal (focal + dice)
      - dice:   region-based, no class weighting parameter

    Usage in training script:
        loss_fn = build_loss_fn("bce", pos_weight=5.0)
        loss_fn = build_loss_fn("focal", focal_alpha=0.5)
    """
    match loss_type:
        case "focal":
            focal = FocalLoss(focal_alpha, focal_gamma)
            return lambda logits, targets, mask=None: masked_reduce(focal(logits, targets), mask)

        case "dice":
            dice = DiceLoss()
            return lambda logits, targets, mask=None: dice(logits, targets, mask).mean()

        case "hybrid":
            return HybridLoss(focal_alpha, focal_gamma, bce_weight, dice_weight)

        case "bce":
            pw = torch.tensor(pos_weight)
            return lambda logits, targets, mask=None: masked_reduce(
                F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none",
                    pos_weight=pw.to(logits.device)), mask)
