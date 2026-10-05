"""The full decoder-only transformer."""

import math

import torch
from torch import nn

from hello_tokens.model.block import Block
from hello_tokens.model.cache import KVCache
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.norm import make_norm
from hello_tokens.model.embedding import Embedding


class GPT(nn.Module):
    """Token ids in, a score for every possible next token out, at every position."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = make_norm(config)
        # Turns each token's vector into one score per vocabulary entry (the "logits").
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        # Weight tying: the output layer reuses the token embedding table. Reading a token in and
        # predicting it out use the same vectors, and it saves 4,096 x 384 = 1.57M parameters.
        self.output.weight = self.embedding.token.weight
        self.apply(self._init_weights)
        # Each block adds two results onto the running vector; shrink those two layers' starting
        # weights so the sum doesn't grow with depth (the GPT-2 recipe).
        for block in self.blocks:
            for layer in (block.attention.out, block.feed_forward.down):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Small random starting weights (standard deviation 0.02) and zero biases.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        # ids: (batch, time)  ->  logits: (batch, time, vocab_size). With a cache, ids are only the new
        # tokens: everything before them is read from the cache, and they are added to it.
        x = self.embedding(ids, cache.length if cache is not None else 0)
        for layer, block in enumerate(self.blocks):
            x = block(x, cache, layer)
        if cache is not None:
            cache.advance(ids.shape[1])
        return self.output(self.final_norm(x))

    def new_cache(self, batch: int = 1) -> KVCache:
        """An empty KV cache on this model's device, in its precision."""
        weight = self.output.weight
        return KVCache(self.config, batch, weight.device, weight.dtype)

    def parameter_count(self) -> int:
        # .parameters() lists the tied table once, so it is counted once.
        return sum(p.numel() for p in self.parameters())
