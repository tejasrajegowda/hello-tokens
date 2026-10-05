"""Rotary position embedding (RoPE; Su et al., 2021).

v1 adds a learned "position vector" to each token. RoPE instead encodes position by *rotating* the
query and key vectors: the head's 64 numbers are treated as 32 pairs, and the pair i of the token at
position p is rotated by the angle p * f_i, where each pair has its own frequency f_i (fast for the
first pairs, slow for the last).

Why it works: the attention score is a dot product, and the dot product of two rotated vectors depends
only on the *difference* of their angles. So the score between a query at position m and a key at
position n depends on m - n, the distance between them, not on where they are. Nothing is learned and
no parameters are added.
"""

import torch
from torch import nn


class RotaryPositions(nn.Module):
    def __init__(self, head_width: int, context: int, base: float = 10_000.0):
        super().__init__()
        if head_width % 2:
            raise ValueError("RoPE needs an even head width: it rotates pairs of numbers")
        # One frequency per pair: base^(-0/hw), base^(-2/hw), ... from 1 down to almost 1/base.
        frequencies = 1.0 / base ** (torch.arange(0, head_width, 2).float() / head_width)
        angles = torch.outer(torch.arange(context).float(), frequencies)  # (context, head_width / 2)
        # Computed once. Not saved in checkpoints (persistent=False): they are fixed by the shape, and
        # leaving them out keeps the checkpoint format the same as v1's.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate (..., time, head_width) vectors whose first token sits at position `offset`."""
        length = x.shape[-2]
        return self._rotate(x, self.cos[offset : offset + length], self.sin[offset : offset + length])

    def at(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Rotate one token's (..., 1, head_width) vectors to `position`, a 1-element tensor.

        The same rotation with the position read from a tensor instead of a Python number, so a CUDA
        graph can replay it at a new position each time (see KVCache.store_at).
        """
        return self._rotate(x, self.cos.index_select(0, position), self.sin.index_select(0, position))

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]  # the two numbers of each pair
        # The 2-D rotation of (even, odd) by the angle: (e cos - o sin, e sin + o cos).
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)  # interleave the pairs back into place
