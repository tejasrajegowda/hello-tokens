"""Causal self-attention, written out by hand.

Every token produces three vectors: a query (what am I looking for?), a key (what do I offer?) and
a value (what do I pass on if chosen). A token's output is a weighted average of the values of the
tokens it may see, weighted by how well its query matches their keys. "Causal" means a token may
see itself and earlier tokens only, never later ones: the model must predict the next token
without reading it.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from hello_tokens.model.cache import KVCache
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.rope import RotaryPositions


def causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention for (..., queries, head_width) against (..., keys, head_width). Returns (output, weights).

    The queries are the *last* tokens of the sequence the keys cover. Without a cache there are as
    many queries as keys; with a cache, a few new queries meet every stored key.
    """
    queries, keys, head_width = query.shape[-2], key.shape[-2], query.shape[-1]
    # How well each query matches each key: (..., queries, keys). Dividing by sqrt(head_width) keeps
    # the scores at a steady size as head_width grows, so softmax doesn't saturate.
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
    # Block the future. Query i sits at position (keys - queries + i), so it may see keys up to there.
    # With as many queries as keys this is the usual triangle: position i sees 0..i.
    future = torch.triu(torch.ones(queries, keys, dtype=torch.bool, device=query.device),
                        diagonal=keys - queries + 1)
    scores = scores.masked_fill(future, float("-inf"))
    # softmax turns each row of scores into weights that are positive and sum to 1;
    # the -inf scores become exactly 0.
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights


def fused_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gqa: bool) -> torch.Tensor:
    """The same attention, done by PyTorch's fused kernel (scaled_dot_product_attention).

    One kernel instead of several (scores, mask, softmax, weighted sum), and it never stores the full
    score table. Its built-in causal mask lines up with the *first* key, which is only right when there
    are as many queries as keys, so the other cases pass their own mask:
    - one new query (decoding with a cache): it may see every key, so no mask at all;
    - a few new queries against a longer cache (checking drafted tokens): the usual bottom-right mask.
    With GQA the kernel shares each key/value head across its group itself (enable_gqa), with no copies.
    """
    queries, keys = query.shape[-2], key.shape[-2]
    if queries == keys:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=gqa)
    mask = None
    if queries > 1:  # True = may attend
        mask = ~torch.triu(torch.ones(queries, keys, dtype=torch.bool, device=query.device),
                           diagonal=keys - queries + 1)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=gqa)


def masked_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    """Attention of one new query over a whole fixed-size cache; `visible` (1, keys) marks the filled
    positions. The rest of the buffer holds zeros or stale tokens, and its scores become -inf, so its
    weights are exactly 0. Used by the fixed-shape step (lesson 17b), where the number of keys is always
    the full context, so that the shapes never change between steps.
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~visible, float("-inf"))
    return torch.softmax(scores, dim=-1) @ value


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
        # Which implementation computes attention: ours (False) or PyTorch's fused kernel (True). Same
        # weights, same results; a runtime choice, not part of the model's shape.
        self.fused = False

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries, keys and values for every head: each (batch, heads, time, head_width)."""
        batch, length, width = x.shape
        query, key, value = self.qkv(x).split([width, self.kv_width, self.kv_width], dim=-1)

        # (batch, time, heads * head_width) -> (batch, heads, time, head_width): one slice per head.
        def split_heads(t: torch.Tensor, heads: int) -> torch.Tensor:
            return t.view(batch, length, heads, self.head_width).transpose(1, 2)

        return split_heads(query, self.heads), split_heads(key, self.kv_heads), split_heads(value, self.kv_heads)

    def _combine(self, output: torch.Tensor) -> torch.Tensor:
        # Put the heads back side by side: (batch, heads, time, head_width) -> (batch, time, width).
        batch, _, length, _ = output.shape
        return self.out(output.transpose(1, 2).reshape(batch, length, self.heads * self.head_width))

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        offset = cache.length if cache is not None else 0  # position of the first new token
        query, key, value = self._project(x)
        if self.rope is not None:
            # Values carry content, not position. With a cache, the new tokens sit after the stored ones.
            query, key = self.rope(query, offset), self.rope(key, offset)
        if cache is not None:
            # Keys are stored already rotated, so stored ones never need rotating again.
            key, value = cache.store(layer, key, value)
        if self.fused:
            output = fused_attention(query, key, value, gqa=self.kv_heads != self.heads)
        else:
            if self.kv_heads != self.heads:
                # GQA: each key/value head serves a group of query heads. Repeating it once per query
                # head in its group lines them up: query heads 0-2 use kv head 0, heads 3-5 kv head 1.
                group = self.heads // self.kv_heads
                key, value = key.repeat_interleave(group, dim=1), value.repeat_interleave(group, dim=1)
            output, _ = causal_attention(query, key, value)
        return self._combine(output)

    def decode(self, x: torch.Tensor, cache: KVCache, layer: int, position: torch.Tensor,
               visible: torch.Tensor) -> torch.Tensor:
        """One new token at `position` (a 1-element tensor), attending over the whole cache buffer.

        The same result as `forward` with a cache, but every shape is fixed and the position lives in a
        tensor, so a CUDA graph can record it once and replay it at every step (lesson 17b). The price
        is a little extra arithmetic: attention always covers all `context` slots, the unfilled ones
        weighted 0. At this model's size a step is dominated by launching work, not by doing it.
        """
        query, key, value = self._project(x)
        if self.rope is not None:
            query, key = self.rope.at(query, position), self.rope.at(key, position)
        key, value = cache.store_at(layer, key, value, position)
        if self.fused:
            output = F.scaled_dot_product_attention(query, key, value, attn_mask=visible,
                                                    enable_gqa=self.kv_heads != self.heads)
        else:
            if self.kv_heads != self.heads:
                group = self.heads // self.kv_heads
                key, value = key.repeat_interleave(group, dim=1), value.repeat_interleave(group, dim=1)
            output = masked_attention(query, key, value, visible)
        return self._combine(output)
