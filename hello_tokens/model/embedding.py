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
        self.position = nn.Embedding(config.context, config.width)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (batch, time) integers  ->  (batch, time, width) vectors
        length = ids.shape[1]
        if length > self.context:
            raise ValueError(f"sequence of {length} tokens is longer than the context of {self.context}")
        positions = torch.arange(length, device=ids.device)
        return self.token(ids) + self.position(positions)
