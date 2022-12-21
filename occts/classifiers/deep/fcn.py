import torch
from torch import nn
from occts.classifiers.deep.utils import ConvBlock


class FCNClassifier(nn.Module):
    """ The Fully Connected Network (FCN) Classifier implements the convolutional
    blocks to operate over the time series.

    The network layers are composed of:

    ConvBlock      |
        Conv       |
        BatchNorm  |
        ReLU       |
    ConvBlock      |
        Conv       |
        BatchNorm  |
        ReLU       |
    ConvBlock      |
        Conv       |
        BatchNorm  |
        ReLU       |
    Linear         v
    """

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.layers = nn.Sequential(*[
            ConvBlock(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=8,
                stride=1,
                bias=False,
            ),
            ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                bias=False
            ),
            ConvBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                bias=False
            )
        ])

        self.head = nn.Linear(in_features=128, out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.head(x.mean(dim=-1))
