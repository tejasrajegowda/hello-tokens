"""One transformer block: attention, then a feed-forward network, each wrapped in a residual."""

import torch
from torch import nn

from hello_tokens.model.attention import CausalSelfAttention
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.norm import make_norm


class FeedForward(nn.Module):
    """Two layers applied to each token on its own: widen 4x, bend with GELU, narrow back."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up = nn.Linear(config.width, 4 * config.width)
        self.gelu = nn.GELU()
        self.down = nn.Linear(4 * config.width, config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.gelu(self.up(x)))


class Block(nn.Module):
    """Attention lets tokens share information; the feed-forward network thinks about each token.

    Each is applied to a normalised copy of x and *added* back onto x (a residual connection), so
    every block refines the running vector instead of replacing it.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(config)
        self.attention = CausalSelfAttention(config)
        self.norm2 = make_norm(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
