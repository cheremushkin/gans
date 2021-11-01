import torch
import torch.nn as nn


class SeparableConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = None):
        super().__init__()
        self.dw = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=(3, 3),
                            groups=in_channels,
                            bias=False)
        self.pw = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=(1, 1),
                            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
