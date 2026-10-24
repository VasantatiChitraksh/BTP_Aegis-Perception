"""Small PyTorch equivalents of upstream compatibility-only dependencies."""

import torch
from torch import nn
from torch.nn import functional as F

trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(value):
    return (value, value) if isinstance(value, int) else value


def register_notrace_function(function):
    return function  # Only an FX tracing annotation upstream; training does not trace.


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        keep = 1 - self.drop_prob
        mask = (
            keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        ).floor()
        return x.div(keep) * mask


class TensorTransforms:
    @staticmethod
    def normalize(x, mean, std):
        mean = x.new_tensor(mean).reshape(-1, 1, 1)
        std = x.new_tensor(std).reshape(-1, 1, 1)
        return (x - mean) / std

    @staticmethod
    def pad(x, padding):
        left, top, right, bottom = padding
        return F.pad(x, (left, right, top, bottom))


def denormalize(x, mean, std):
    return x * x.new_tensor(std).reshape(-1, 1, 1) + x.new_tensor(mean).reshape(-1, 1, 1)


def safe_depthwise_forward(conv, inputs):
    """Preserve full spatial context while avoiding old CUDA kernel index limits."""
    from . import restormer

    if inputs.numel() < restormer._DEPTHWISE_INDEX_LIMIT:
        return nn.Conv2d.forward(conv, inputs)
    return restormer._depthwise_conv(conv, inputs)
