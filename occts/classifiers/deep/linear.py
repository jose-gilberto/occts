""" Linear baseline module.
"""
import torch
from torch import nn


class LinearBlock(nn.Module):
    """ Linear block used in the linear baseline model layers,
    that block is composed of a sequence with 3 layers, a linear
    fully connected layer, a ReLU activation layer and a Dropout used
    in the training phase.

    That way we have a pipeline composed for:
    (LinearBlock)
    Linear => ReLU => Dropout
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
                bias=False
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearClassifier(nn.Module):
    """ Linear classifier used for one class classification.
    The last layer generate a latent output in a latent space used
    for the svdd algorithm to generate an hypersphere that will be used
    to solve the one class classification task.

    That model is composed by 1 dropout layer and 3 linear blocks. The
    pipeline respect the following order:
    Dropout -> LinearBlock -> LinearBlock -> LinearBlock

    Args:
        - input_dim (int): Input dimension, given a time series data x
            the input_dim size is the time series length or x.shape[1] in
            a data with 2 dimensions or x.shape[2] in a data with 3 dims.
        - latent_dim (int): Latent space output dimension.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.layers = nn.Sequential(
            nn.Dropout(0.1),
            LinearBlock(input_dim, 500, 0.2),
            LinearBlock(500, 500, 0.2),
            LinearBlock(500, latent_dim, 0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.view(x.shape[0], -1))
