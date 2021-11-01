import torch
import torch.nn as nn


class SEConvBlock(nn.Module):
    def __init__(self,
                 n_channels: int,
                 r: int = 16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(n_channels, n_channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // r, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = x.shape
        sq = self.sq(x).view(bs, c)
        ex = self.ex(sq).view(bs, c, 1, 1)
        x = x * ex.expand_as(x)
        return x
