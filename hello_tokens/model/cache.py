"""The KV cache: keep every layer's keys and values, so each new token is computed once.

Without a cache, writing token n re-reads all n tokens so far through the whole model, although only
the newest one is new. Each earlier token's key and value never change once computed (a token can't
see later tokens), so they can be stored and reused: a step then runs the model on one token.

The buffers are allocated once, at full context size, with a length pointer for how much is filled.
Fixed buffers are what CUDA graphs need later, and going back (speculative decoding rejecting drafted
tokens) is just moving the pointer: the stale entries beyond it are overwritten by the next write.
"""

import torch

from hello_tokens.model.config import ModelConfig


class KVCache:
    def __init__(self, config: ModelConfig, batch: int = 1, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32):
        # (layers, batch, key/value heads, positions, head width). With GQA only the key/value heads
        # are stored, which is why GQA makes the cache smaller.
        shape = (config.layers, batch, config.kv_head_count, config.context, config.head_width)
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.capacity = config.context
        self.length = 0  # positions filled, in every layer

    def store(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a layer's new keys and values after the stored ones; return everything so far.

        The pointer itself only moves once all layers are done (`advance`), because every layer writes
        the same positions.
        """
        start, end = self.length, self.length + key.shape[-2]
        if end > self.capacity:
            raise ValueError(f"the cache holds {self.capacity} positions; {end} were needed")
        self.keys[layer, :, :, start:end] = key
        self.values[layer, :, :, start:end] = value
        return self.keys[layer, :, :, :end], self.values[layer, :, :, :end]

    def store_at(self, layer: int, key: torch.Tensor, value: torch.Tensor,
                 position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one new token's key and value at `position` (a 1-element tensor); return the whole
        buffers, filled or not.

        The fixed-shape twin of `store`, for a CUDA graph (lesson 17b). A graph replays recorded GPU
        work with the numbers it saw while recording, so a position held in Python would be frozen in.
        Held in a tensor, it is read on the GPU at every replay. The returned buffers always have the
        full context length, so their shape never changes either; the caller hides the unfilled part.
        """
        self.keys[layer].index_copy_(2, position, key)
        self.values[layer].index_copy_(2, position, value)
        return self.keys[layer], self.values[layer]

    def advance(self, count: int) -> None:
        self.length += count

    def rollback(self, length: int) -> None:
        """Forget everything after the first `length` positions."""
        if not 0 <= length <= self.length:
            raise ValueError(f"can't roll back to {length}: the cache holds {self.length}")
        self.length = length

    def bytes(self) -> int:
        return 2 * self.keys.numel() * self.keys.element_size()
