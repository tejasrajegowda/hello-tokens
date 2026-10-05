"""Turning token ids into vectors the transformer can work with."""

import torch
from torch import nn

from hello_tokens.model.config import ModelConfig


class Embedding(nn.Module):
    """Token embedding plus learned position embedding.

    Each token id looks up a learned vector (what the token is). Each position 0..context-1 looks up
    another learned vector (where it is). Attention on its own cannot tell positions apart, so the
    two are added together.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.context = config.context
        self.token = nn.Embedding(config.vocab_size, config.width)
        # With RoPE, position is applied inside attention instead, so there is no position table.
        self.position = nn.Embedding(config.context, config.width) if config.position == "learned" else None

    def forward(self, ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # ids: (batch, time) integers  ->  (batch, time, width) vectors. `offset` is the position of
        # the first id (non-zero when earlier tokens are already in a KV cache).
        length = ids.shape[1]
        if offset + length > self.context:
            raise ValueError(f"sequence of {offset + length} tokens is longer than the context of {self.context}")
        if self.position is None:
            return self.token(ids)
        positions = torch.arange(offset, offset + length, device=ids.device)
        return self.token(ids) + self.position(positions)

    def at(self, ids: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """(batch, 1) ids of one new token at `position`, a 1-element tensor (see KVCache.store_at)."""
        if self.position is None:
            return self.token(ids)
        return self.token(ids) + self.position(position)
