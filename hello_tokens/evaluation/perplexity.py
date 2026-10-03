"""Perplexity over a whole token file.

Training checks itself on a fixed sample of held-out windows. This scores every token in the file
exactly once: the file is cut into back-to-back windows of `context` predictions, where the last
token of one window is the first input of the next, so no target is skipped or counted twice.

Perplexity is exp(average loss). A perplexity of 3.6 means the model is, on average, as unsure as
if it were choosing evenly among 3.6 tokens; a model that knows nothing scores the vocabulary size.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from hello_tokens.model.gpt import GPT


@dataclass(frozen=True)
class Score:
    loss: float  # average cross-entropy per token, in nats
    perplexity: float
    tokens: int  # how many next-token predictions were scored


@torch.no_grad()
def perplexity(
    model: GPT, tokens: np.ndarray, batch_size: int = 64, device: str = "cpu", precision=None
) -> Score:
    """Score every next-token prediction in `tokens` once. Returns the average loss and perplexity."""
    model.eval()
    context = model.config.context
    total_loss = 0.0
    total_tokens = 0

    def score(windows: np.ndarray) -> None:
        nonlocal total_loss, total_tokens
        windows = torch.from_numpy(windows.astype(np.int64)).to(device)
        inputs, targets = windows[:, :-1], windows[:, 1:]
        with torch.autocast(device, dtype=precision or torch.float32, enabled=precision is not None):
            logits = model(inputs)
        # Add up the losses rather than averaging per batch, in float32, so the short last window
        # carries exactly its share and rounding doesn't build up over thousands of batches.
        total_loss += F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
        ).item()
        total_tokens += targets.numel()

    # Window i reads tokens[i*context : (i+1)*context] and predicts the tokens one place later.
    full_windows = (len(tokens) - 1) // context
    for first in range(0, full_windows, batch_size):
        starts = range(first * context, min(first + batch_size, full_windows) * context, context)
        score(np.stack([tokens[s : s + context + 1] for s in starts]))
    if (len(tokens) - 1) % context:  # the few predictions left at the end, as one shorter window
        score(np.asarray(tokens[full_windows * context :])[None])

    loss = total_loss / total_tokens
    return Score(loss=loss, perplexity=math.exp(loss), tokens=total_tokens)
