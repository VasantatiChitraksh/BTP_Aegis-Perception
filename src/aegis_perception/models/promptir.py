import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptGenModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prompt_embed = nn.Parameter(torch.randn(1, dim, 1, 1))
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        prompt = self.prompt_embed.expand(b, -1, h, w)
        return self.conv(prompt)

class PromptIRGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dim, 3, 1, 1)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GELU()
            ) for _ in range(num_blocks)
        ])
        
        self.prompt_gen = PromptGenModule(dim)
        self.head = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.stem(x)
        prompt = self.prompt_gen(feat)
        
        for block in self.blocks:
            feat = block(feat + prompt)
            
        out = self.head(feat)
        return out + x
