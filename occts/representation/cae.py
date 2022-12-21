import torch
import torch.nn as nn


def swish(x: torch.Tensor) -> torch.Tensor:
    """"""
    return x * torch.sigmoid(x)


class ConvEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            padding='same'
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=7,
            padding='same'
        )

        self.dropout = nn.Dropout1d(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = swish(self.conv1(x))
        x = self.dropout(x)
        x = swish(self.conv2(x))
        return x


class ConvDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=16,
            out_channels=16,
            kernel_size=7,
            padding='same'
        )

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=7,
            padding='same'
        )

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=7,
            padding='same'
        )

        self.dropout = nn.Dropout1d(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = self.dropout(x)
        return self.conv3(x)


class CAE(nn.Module):

    def __init__(self, device: torch.cuda.device) -> None:
        super().__init__()

        self.encoder = ConvEncoder().to(device)
        self.decoder = ConvDecoder().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))