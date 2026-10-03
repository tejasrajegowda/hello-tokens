"""Causal self-attention, written out by hand.

Every token produces three vectors: a query (what am I looking for?), a key (what do I offer?) and
a value (what do I pass on if chosen). A token's output is a weighted average of the values of the
tokens it may see, weighted by how well its query matches their keys. "Causal" means a token may
see itself and earlier tokens only, never later ones: the model must predict the next token
without reading it.
"""

import math

import torch
from torch import nn

from hello_tokens.model.config import ModelConfig
from hello_tokens.model.rope import RotaryPositions


def causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention for (..., time, head_width) inputs. Returns (output, weights)."""
    length, head_width = query.shape[-2], query.shape[-1]
    # How well each query matches each key: (..., time, time). Dividing by sqrt(head_width) keeps
    # the scores at a steady size as head_width grows, so softmax doesn't saturate.
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
    # Block the future: position i may not look at any position j > i.
    future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=query.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    # softmax turns each row of scores into weights that are positive and sum to 1;
    # the -inf scores become exactly 0.
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights


class CausalSelfAttention(nn.Module):
    """Several attention heads side by side, each free to look for something different."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        # One layer makes queries, keys and values for every head at once.
        self.qkv = nn.Linear(config.width, 3 * config.width)
        # Mixes what the heads found back into one vector per token.
        self.out = nn.Linear(config.width, config.width)
        # RoPE rotates queries and keys by their position (it has no learned weights).
        self.rope = RotaryPositions(config.head_width, config.context) if config.position == "rope" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).split(width, dim=-1)

        # (batch, time, width) -> (batch, heads, time, head_width): each head gets its own slice.
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

        query, key, value = split_heads(query), split_heads(key), split_heads(value)
        if self.rope is not None:
            query, key = self.rope(query), self.rope(key)  # values carry content, not position
        output, _ = causal_attention(query, key, value)
        # Put the heads back side by side: (batch, heads, time, head_width) -> (batch, time, width).
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.out(output)
