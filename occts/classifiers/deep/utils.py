""" Utils module that implement some methods
to operate convolutions in time series, like same padding and other ones.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Optional


# TODO: fix the type declaration of stride, dilation, groups
def _conv1d_same_padding(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: Any,
    dilation: Any,
    groups: Any,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = x.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)

    if padding % 2 != 0:
        x = F.pad(x, [0, 1])

    return F.conv1d(
        input=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups
    )


class Conv1dSamePadding(nn.Conv1d):
    """ Implements the same padding functionality from TensorFlow
    See: https://github.com/pytorch/pytorch/issues/3867
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-renamed
        return _conv1d_same_padding(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups
        )


class ConvBlock(nn.Module):
    """ Implement a conv block that will operate over the time series
    data. The block consist in a Conv1d layer, a batch norm layer and
    a ReLu activation function.

    The data flow implemented is the following one:
        Conv1d(padding=`same`) -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: Optional[bool] = True
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
