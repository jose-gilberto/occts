import torch
import torch.nn as nn


class LinearBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearClassifier(nn.Module):

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