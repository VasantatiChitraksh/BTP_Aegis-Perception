"""Version 2 implementation, separate from the original contributor module."""

import torch
import torch.nn as nn


class LocalResidualBlock(nn.Module):
    """Linear-cost spatial mixing for the high-resolution feature maps."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class LiteWeatherTransformerV2(nn.Module):
    """Project-specific compact Transformer; not an official paper reproduction.

    Restormer channel-attention blocks operate at 1/4 and 1/8 resolution.
    A content-derived soft mixture of learned prompt vectors needs no weather labels.
    The legacy convolution-only class remains in the original module.
    """

    spatial_multiple = 8

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=24,
        num_weather_types=4,
        num_blocks=(2, 4),
        heads=(4, 8),
        ffn_expansion_factor=2.0,
    ):
        super().__init__()
        from .restormer import TransformerBlock

        if in_channels != out_channels or dim <= 0 or num_weather_types <= 0:
            raise ValueError(
                "Residual restoration needs equal input/output channels and positive widths"
            )
        if len(num_blocks) != 2 or len(heads) != 2:
            raise ValueError("Specify two Transformer block counts and two head counts")
        if any(b <= 0 for b in num_blocks) or ffn_expansion_factor <= 0:
            raise ValueError("Transformer block counts and expansion must be positive")
        if any(h <= 0 or (dim * (4 << i)) % h for i, h in enumerate(heads)):
            raise ValueError("Attention heads must divide their stage channels")

        def blocks(channels, count, head_count):
            return nn.Sequential(
                *[
                    TransformerBlock(channels, head_count, ffn_expansion_factor, False, "WithBias")
                    for _ in range(count)
                ]
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1), LocalResidualBlock(dim)
        )
        self.down1 = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1)
        self.enc1 = LocalResidualBlock(dim * 2)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1)
        self.enc2 = blocks(dim * 4, num_blocks[0], heads[0])
        self.down3 = nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1)
        self.prompt_router = nn.Linear(dim * 8, num_weather_types)
        self.weather_prompts = nn.Parameter(torch.empty(num_weather_types, dim * 8))
        nn.init.normal_(self.weather_prompts, std=0.02)
        self.latent = blocks(dim * 8, num_blocks[1], heads[1])
        self.up3 = nn.Sequential(nn.Conv2d(dim * 8, dim * 16, 1), nn.PixelShuffle(2))
        self.fuse2 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.dec2 = blocks(dim * 4, 1, heads[0])
        self.up2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 1), nn.PixelShuffle(2))
        self.fuse1 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.dec1 = LocalResidualBlock(dim * 2)
        self.up1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 1), nn.PixelShuffle(2))
        self.fuse0 = nn.Conv2d(dim * 2, dim, 1)
        self.refine = LocalResidualBlock(dim)
        self.head = nn.Conv2d(dim, out_channels, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError("Use inference.predict to pad native images to multiples of eight")
        s0 = self.stem(x)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))
        latent = self.down3(s2)
        weights = self.prompt_router(latent.mean(dim=(-2, -1))).softmax(dim=-1)
        prompt = (weights @ self.weather_prompts).unsqueeze(-1).unsqueeze(-1)
        latent = self.latent(latent + prompt)
        d2 = self.dec2(self.fuse2(torch.cat((self.up3(latent), s2), dim=1)))
        d1 = self.dec1(self.fuse1(torch.cat((self.up2(d2), s1), dim=1)))
        d0 = self.refine(self.fuse0(torch.cat((self.up1(d1), s0), dim=1)))
        return x + self.head(d0)
