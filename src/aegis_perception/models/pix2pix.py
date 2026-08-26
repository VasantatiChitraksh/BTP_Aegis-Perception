from __future__ import annotations

import torch
from torch import nn


class AttentionGate(nn.Module):
    """Additive attention for filtering an encoder skip connection."""

    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int):
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Attention inputs must have the same spatial size: {gate.shape} vs {skip.shape}"
            )
        projected = self.gate_projection(gate) + self.skip_projection(skip)
        weights = self.mask(self.activation(projected))
        return skip * weights


class AttentionUNetGenerator(nn.Module):
    """Six-level Pix2Pix U-Net with switchable attention-gated skip connections.

    This preserves the historical notebook topology for a fair attention on/off
    ablation. The highest-resolution skip remains ungated, matching that notebook.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.attention = attention

        self.down1 = self._down(in_channels, features, normalize=False)
        self.down2 = self._down(features, features * 2)
        self.down3 = self._down(features * 2, features * 4)
        self.down4 = self._down(features * 4, features * 8)
        self.down5 = self._down(features * 8, features * 8)
        self.down6 = self._down(features * 8, features * 8)

        self.up1 = self._up(features * 8, features * 8, dropout=True)
        self.up2 = self._up(features * 16, features * 8, dropout=True)
        self.up3 = self._up(features * 16, features * 4)
        self.up4 = self._up(features * 8, features * 2)
        self.up5 = self._up(features * 4, features)

        self.att5 = AttentionGate(features * 8, features * 8, features * 4) if attention else None
        self.att4 = AttentionGate(features * 8, features * 8, features * 4) if attention else None
        self.att3 = AttentionGate(features * 4, features * 4, features * 2) if attention else None
        self.att2 = AttentionGate(features * 2, features * 2, features) if attention else None

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _down(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _up(in_channels: int, out_channels: int, dropout: bool = False) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def _filter(
        self, gate: torch.Tensor, skip: torch.Tensor, module: nn.Module | None
    ) -> torch.Tensor:
        return module(gate, skip) if module is not None else skip

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(inputs)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat((u1, self._filter(u1, d5, self.att5)), dim=1))
        u3 = self.up3(torch.cat((u2, self._filter(u2, d4, self.att4)), dim=1))
        u4 = self.up4(torch.cat((u3, self._filter(u3, d3, self.att3)), dim=1))
        u5 = self.up5(torch.cat((u4, self._filter(u4, d2, self.att2)), dim=1))
        return self.final(torch.cat((u5, d1), dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, image_channels: int = 3, features: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(image_channels * 2, features, normalize=False),
            *block(features, features * 2),
            *block(features * 2, features * 4),
            *block(features * 4, features * 8),
            nn.Conv2d(features * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, condition: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((condition, candidate), dim=1))


def initialize_pix2pix_weights(module: nn.Module) -> None:
    """Weight initialization from the original Pix2Pix implementation."""
    name = module.__class__.__name__
    if "Conv" in name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
