import typing as tp
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """
    Generator for GAN
    """
    def __init__(self,
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

        self.blocks: nn.ModuleList = nn.ModuleList()
        in_channels = n_channels
        for n_layer in range(n_layers):
            out_channels = n_features * (2 ** n_layer)
            self.blocks.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
            in_channels = out_channels

        self.activation = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.activation(x)
