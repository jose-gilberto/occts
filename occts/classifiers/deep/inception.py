import torch
from torch import nn

from .utils import Conv1dSamePadding
from typing import cast, List, Union


class InceptionBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool,
        stride: int = 1,
        bottleneck_channels: int = 32,
        kernel_size: int = 41,
    ) -> None:
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                bias=False
            )

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3

        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size_s[i],
                stride=stride,
                bias=False
            ) for i in range(len(kernel_size_s))
        ])

        self.batch_norm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual

        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_x = x

        if self.use_bottleneck:
            x = self.bottleneck(x)

        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)

        return x


class InceptionModel(nn.Module):

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: Union[List[int], int],
        bottleneck_channels: Union[List[int], int],
        kernel_sizes: Union[List[int], int],
        use_residuals: Union[List[bool], bool, str] = 'default',
        latent_dim: int = 32
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = bottleneck_channels
        self.kernel_sizes = kernel_sizes
        self.use_residuals = use_residuals

        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels, num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))

        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]

        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks
        ))

        self.blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                residual=use_residuals[i],
                bottleneck_channels=bottleneck_channels[i],
                kernel_size=kernel_sizes[i]
            ) for i in range(num_blocks)
        ])

        self.linear = nn.Linear(
            in_features=channels[-1],
            out_features=latent_dim
        )

    @staticmethod
    def _expand_to_blocks(
        value: Union[int, bool, List[int], List[bool]],
        num_blocks: int
    ) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expect length {num_blocks}, got {len(value)}.'
        else:
            value = [value] * num_blocks

        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x).mean(dim=-1)
        return self.linear(x)
