from typing import Literal, Sequence, Type

import torch
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn import functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _BaseUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_channels: Sequence[int],
        *,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        channels = tuple(encoder_channels)
        if not channels:
            raise ValueError("encoder_channels must contain at least one stage")

        self.stem = _DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList()
        prev_channels = channels[0]
        for out_channel in channels[1:]:
            self.downs.append(_DownBlock(prev_channels, out_channel))
            prev_channels = out_channel

        self.ups = nn.ModuleList()
        decoder_in_channels = prev_channels
        for skip_channels in reversed(channels[:-1]):
            self.ups.append(_UpBlock(decoder_in_channels, skip_channels, skip_channels))
            decoder_in_channels = skip_channels

        self.head = nn.Conv2d(decoder_in_channels, out_channels, kernel_size=1)

    def _forward_features(
        self, x: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b 1 h w"]:
        skips: list[Tensor] = []
        out = self.stem(x)
        for down in self.downs:
            skips.append(out)
            out = down(out)

        for up, skip in zip(self.ups, reversed(skips)):
            out = up(out, skip)

        return self.head(out)

    def forward(
        self,
        x: Float[Tensor, "b c h w"],
    ):
        logits = self._forward_features(x)
        return logits


class ShallowUnet(_BaseUnet):
    def __init__(
        self, in_channels: int, base_channels: int = 64, out_channels: int = 1
    ) -> None:
        super().__init__(
            in_channels,
            encoder_channels=(base_channels, base_channels * 2, base_channels * 4),
            out_channels=out_channels,
        )


class Unet(_BaseUnet):
    def __init__(
        self, in_channels: int, base_channels: int = 32, out_channels: int = 1
    ) -> None:
        super().__init__(
            in_channels,
            encoder_channels=(
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
            ),
            out_channels=out_channels,
        )


class DeepUnet(_BaseUnet):
    def __init__(
        self, in_channels: int, base_channels: int = 32, out_channels: int = 1
    ) -> None:
        super().__init__(
            in_channels,
            encoder_channels=(
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            ),
            out_channels=out_channels,
        )


type UnetType = Literal["shallow", "unet", "deep"]
unet_type_mapping: dict[UnetType, Type[_BaseUnet]] = {
    "shallow": ShallowUnet,
    "unet": Unet,
    "deep": DeepUnet,
}
