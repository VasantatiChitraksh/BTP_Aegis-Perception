import torch
import torch.nn as nn

class RAMiTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Spatial attention (simplified)
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Sigmoid()
        )
        # Channel attention (simplified)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        sa = self.spatial(x) * x
        ca = self.channel(sa) * sa
        return x + self.proj(ca)

class RAMiTGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=32, num_blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.blocks = nn.Sequential(*[RAMiTBlock(dim) for _ in range(num_blocks)])
        self.head = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.blocks(feat)
        out = self.head(feat)
        return out + x
