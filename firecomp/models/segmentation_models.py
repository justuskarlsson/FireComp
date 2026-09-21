from typing import Callable, Literal

import segmentation_models_pytorch as smp
from torch import nn

from firecomp.models.unet import ShallowUnet, Unet, DeepUnet


def create_unet_shallow(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
) -> nn.Module:
    """Shallow custom UNet (3 encoder stages)."""
    return ShallowUnet(in_channels=in_channels, out_channels=out_channels)


def create_unet(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
) -> nn.Module:
    """Standard custom UNet (4 encoder stages)."""
    return Unet(in_channels=in_channels, out_channels=out_channels)


def create_unet_deep(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
) -> nn.Module:
    """Deep custom UNet (5 encoder stages)."""
    return DeepUnet(in_channels=in_channels, out_channels=out_channels)


UNETPP_INCOMPATIBLE_ENCODERS = frozenset([
    "tu-convnext_tiny", "tu-convnextv2_tiny", "tu-convnextv2_base",
    "tu-convnext_small", "tu-convnext_base", "tu-convnext_large",
])


def create_unet_plusplus(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
) -> nn.Module:
    """UNet++ with dense skip connections for precise boundary detection."""
    if encoder_name in UNETPP_INCOMPATIBLE_ENCODERS:
        raise ValueError(
            f"UNet++ is incompatible with encoder '{encoder_name}'. "
            f"These timm ConvNeXt encoders have 0-channel stages that cause decoder failures. "
            f"Use a different encoder (resnet18, efficientnet-b3, etc.) or a different model type."
        )
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_deeplabv3plus(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
) -> nn.Module:
    """DeepLabV3+ with atrous convolutions for multi-scale context."""
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_manet(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
) -> nn.Module:
    """MAnet with multi-scale attention for focusing on fire regions."""
    return smp.MAnet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_fpn(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
) -> nn.Module:
    """FPN with feature pyramid for multi-scale feature extraction."""
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_upernet(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
) -> nn.Module:
    """UPerNet for unified local and global context understanding."""
    return smp.UPerNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


def create_vit(
    in_channels: int,
    out_channels: int = 1,
    encoder_name: str | None = None,  # ignored — mit_b2 hardcoded
    encoder_weights: str | None = None,
) -> nn.Module:
    """ViT-style segmentation: MiT-B2 (Mix Transformer) encoder + UNet decoder.

    MiT-B2 is a hierarchical Vision Transformer encoder from SegFormer.
    Paired with the SMP UNet decoder it gives a full SETR-style encoder-decoder.
    ``encoder_name`` is accepted but ignored so the factory signature stays
    uniform — mit_b2 is always used.
    """
    return smp.Unet(
        encoder_name="mit_b2",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


type ModelType = Literal[
    "unet_shallow",
    "unet",
    "unet_deep",  # custom unets (no encoder)
    "unet++",
    "deeplabv3+",
    "manet",
    "fpn",
    "upernet",
    "vit",  # MiT-B2 encoder + UNet decoder (ViT/SETR-style)
]

type EncoderType = Literal[
    "resnet18",  # lightweight, fast (default)
    "resnet34",  # lightweight, fast
    "resnet50",  # good balance of speed/accuracy
    "resnext50_32x4d",  # better features than resnet
    "efficientnet-b3",  # excellent accuracy/efficiency for satellite data
    "efficientnet-b4",  # more capacity
    "mit_b2",  # Mix Transformer, good for dense prediction
    "tu-convnext_tiny",  # ConvNeXt V1 tiny, modern CNN
    "tu-convnextv2_tiny",  # ConvNeXt V2 tiny, ~28M params
    "tu-convnextv2_base",  # ConvNeXt V2 base, ~89M params
]

model_factory: dict[ModelType, Callable[..., nn.Module]] = {
    # Custom UNets
    "unet_shallow": create_unet_shallow,
    "unet": create_unet,
    "unet_deep": create_unet_deep,
    # SMP models
    "unet++": create_unet_plusplus,
    "deeplabv3+": create_deeplabv3plus,
    "manet": create_manet,
    "fpn": create_fpn,
    "upernet": create_upernet,
    # ViT/SETR-style
    "vit": create_vit,
}
