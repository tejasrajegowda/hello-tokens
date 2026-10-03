"""Cutting training examples out of the token file."""

import numpy as np
import torch


def get_batch(
    tokens: np.ndarray, batch_size: int, context: int, generator: np.random.Generator, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick batch_size random windows of `context` tokens. Returns (inputs, targets).

    The target at each position is the token that comes next, so one window of 256 tokens is
    256 next-token predictions at once.
    """
    starts = generator.integers(0, len(tokens) - context - 1, size=batch_size)
    windows = np.stack([tokens[s : s + context + 1] for s in starts]).astype(np.int64)
    windows = torch.from_numpy(windows).to(device)
    return windows[:, :-1], windows[:, 1:]  # inputs: tokens 0..T-1, targets: tokens 1..T
