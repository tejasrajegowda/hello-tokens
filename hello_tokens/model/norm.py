"""Normalisation: keep each token's vector at a steady size as it passes through the blocks.

v1 uses LayerNorm: subtract the vector's mean, divide by its standard deviation, then apply a learned
scale and shift. RMSNorm (Zhang and Sennrich, 2019) drops the mean and the shift: divide by the root
mean square, apply a learned scale. It is simpler and cheaper, and works as well in practice, which
is why modern models (LLaMA and its successors) use it.
"""

import torch
from torch import nn

from hello_tokens.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # keeps the division safe when a vector is all zeros
        self.weight = nn.Parameter(torch.ones(width))  # the learned scale, one per dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Root mean square over the last dimension; rsqrt = 1 / sqrt.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def make_norm(config: ModelConfig) -> nn.Module:
    if config.norm == "layernorm":
        return nn.LayerNorm(config.width)
    return RMSNorm(config.width)
