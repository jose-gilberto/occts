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
    l_out = l_in = input.size(2)
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
