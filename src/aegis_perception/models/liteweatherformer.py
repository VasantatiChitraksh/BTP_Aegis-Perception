import torch
import torch.nn as nn
import torch.nn.functional as F

class LiteWeatherFormerGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_weather_types=4):
        super().__init__()
        # Conv Stem (local edges; low cost)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        
        # Multi-Scale Encoder (depthwise conv + efficient transformer at 1/4 and 1/8 scale)
        self.down1 = nn.Conv2d(dim, dim*2, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 3, padding=1, groups=dim*2),
            nn.GELU(),
            nn.Conv2d(dim*2, dim*2, 1)
        )
        self.down2 = nn.Conv2d(dim*2, dim*4, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim*4, dim*4, 3, padding=1, groups=dim*4),
            nn.GELU(),
            nn.Conv2d(dim*4, dim*4, 1)
        )
        
        # Weather Prompts
        self.weather_prompts = nn.Parameter(torch.randn(num_weather_types, dim*4, 1, 1))
        
        # Light Decoder (skip fusion)
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(dim*2 * 2, dim*2, 3, padding=1),
            nn.GELU()
        )
        self.up1 = nn.ConvTranspose2d(dim*2, dim, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GELU()
        )
        
        self.head = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x, weather_idx=0):
        # Allow default weather_idx if not provided (for simplicity)
        if isinstance(weather_idx, torch.Tensor):
            # Batched weather idx
            prompt = self.weather_prompts[weather_idx]
        else:
            prompt = self.weather_prompts[weather_idx].unsqueeze(0)
            
        s0 = self.stem(x)
        
        s1 = self.down1(s0)
        s1 = self.enc1(s1)
        
        s2 = self.down2(s1)
        s2 = self.enc2(s2)
        
        # Add prompt at bottleneck
        s2 = s2 + prompt
        
        d1 = self.up2(s2)
        d1 = self.dec2(torch.cat([d1, s1], dim=1))
        
        d0 = self.up1(d1)
        d0 = self.dec1(torch.cat([d0, s0], dim=1))
        
        out = self.head(d0)
        return out + x
