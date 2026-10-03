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
        self.kv_heads = config.kv_head_count
        self.head_width = config.head_width
        # One layer makes queries, keys and values for every head at once. With grouped-query
        # attention (GQA) there are fewer key/value heads than query heads, so the layer is narrower:
        # width for the queries, then kv_heads * head_width each for keys and values. With
        # kv_heads == heads this is v1's 3 * width, so v1 checkpoints still load.
        self.kv_width = self.kv_heads * config.head_width
        self.qkv = nn.Linear(config.width, config.width + 2 * self.kv_width)
        # Mixes what the heads found back into one vector per token.
        self.out = nn.Linear(config.width, config.width)
        # RoPE rotates queries and keys by their position (it has no learned weights).
        self.rope = RotaryPositions(config.head_width, config.context) if config.position == "rope" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).split([width, self.kv_width, self.kv_width], dim=-1)

        # (batch, time, heads * head_width) -> (batch, heads, time, head_width): one slice per head.
        def split_heads(t: torch.Tensor, heads: int) -> torch.Tensor:
            return t.view(batch, length, heads, self.head_width).transpose(1, 2)

        query = split_heads(query, self.heads)
        key, value = split_heads(key, self.kv_heads), split_heads(value, self.kv_heads)
        if self.rope is not None:
            query, key = self.rope(query), self.rope(key)  # values carry content, not position
        if self.kv_heads != self.heads:
            # GQA: each key/value head serves a group of query heads. Repeating it once per query
            # head in its group lines them up: query heads 0-2 use kv head 0, heads 3-5 use kv head 1.
            group = self.heads // self.kv_heads
            key, value = key.repeat_interleave(group, dim=1), value.repeat_interleave(group, dim=1)
        output, _ = causal_attention(query, key, value)
        # Put the heads back side by side: (batch, heads, time, head_width) -> (batch, time, width).
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.out(output)
