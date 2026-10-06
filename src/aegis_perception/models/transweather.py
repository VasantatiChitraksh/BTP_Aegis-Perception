import torch
import torch.nn as nn

class TransWeatherGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_blocks=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            *[nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1)
            ) for _ in range(num_blocks)]
        )
        
        self.decoder = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1)
            ) for _ in range(num_blocks)],
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        )
        
        # Learnable weather queries (simplified)
        self.weather_queries = nn.Parameter(torch.randn(1, dim, 1, 1))

    def forward(self, x):
        features = self.encoder(x)
        # Add weather queries to features
        features = features + self.weather_queries
        out = self.decoder(features)
        # Residual connection
        return out + x
