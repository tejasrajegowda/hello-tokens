"""One transformer block: attention, then a feed-forward network, each wrapped in a residual."""

import torch
import torch.nn.functional as F
from torch import nn

from hello_tokens.model.attention import CausalSelfAttention
from hello_tokens.model.cache import KVCache
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


def swiglu_hidden(width: int) -> int:
    """2/3 of the GELU network's 4x, so three matrices hold as many weights as its two; rounded up
    to a multiple of 64, which GPUs handle efficiently. For width 384: exactly 1,024."""
    return -(-(8 * width // 3) // 64) * 64


class SwiGLU(nn.Module):
    """The gated feed-forward network of modern models (Shazeer, 2020).

    Two layers widen the token: one gives values (`up`), the other a gate (`gate`). The gate passes
    through SiLU (x * sigmoid(x)), a smooth switch, and multiplies the values, so the network learns
    per number how much to let through. Then `down` narrows back.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = swiglu_hidden(config.width)
        self.gate = nn.Linear(config.width, hidden)
        self.up = nn.Linear(config.width, hidden)
        self.down = nn.Linear(hidden, config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


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
        self.feed_forward = SwiGLU(config) if config.feed_forward == "swiglu" else FeedForward(config)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), cache, layer)
        x = x + self.feed_forward(self.norm2(x))
        return x

    def decode(self, x: torch.Tensor, cache: KVCache, layer: int, position: torch.Tensor,
               visible: torch.Tensor) -> torch.Tensor:
        """The fixed-shape one-token step (see CausalSelfAttention.decode)."""
        x = x + self.attention.decode(self.norm1(x), cache, layer, position, visible)
        x = x + self.feed_forward(self.norm2(x))
        return x
