import typing as tp
import torch
import torch.nn as nn


class ConvTransposeBlock(nn.Module):
    """
    ConvTranspose2d -> BatchNorm2d -> ReLU
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tp.Tuple[int, int] = (4, 4),
                 stride: int = 2,
                 padding: int = 1,
                 bias: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """
    Generator for GAN
    """
    def __init__(self,
                 z_size: int,
                 n_features: int,
                 n_layers: int,
                 n_channels: int):
        """
        :param z_size: size of noise vector z
        :param n_features: size of the last feature map
        :param n_layers: number of layers in the generator
        :param n_channels: number of the output channels
        """
        super().__init__()

        in_channels: int = z_size
        out_channels: int = n_features * (n_layers ** 2)
        self.blocks: nn.ModuleList = nn.ModuleList([
            ConvTransposeBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=1,
                               padding=0)
        ])
        in_channels = out_channels
        for n_layer in reversed(range(n_layers - 1)):
            out_channels = n_features * (2 ** n_layer)
            self.blocks.append(ConvTransposeBlock(in_channels=in_channels, out_channels=out_channels))
            in_channels = out_channels

        self.activation: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=n_features,
                out_channels=n_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.activation(x)
