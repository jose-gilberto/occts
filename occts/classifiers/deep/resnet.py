import torch
from torch import nn
from .utils import ConvBlock, Conv1dSamePadding


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=1
            ) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False

        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1
                ),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResnetClassifier(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int = 64, latent_dim: int = 32) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.layers = nn.Sequential(*[
            ResNetBlock(
                in_channels=in_channels,
                out_channels=mid_channels
            ),
            ResNetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels * 2
            ),
            ResNetBlock(
                in_channels=mid_channels * 2,
                out_channels=mid_channels * 2
            )
        ])

        self.head = nn.Linear(mid_channels * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.head(x.mean(dim=-1))

