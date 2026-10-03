"""The loss, the optimizer, the learning-rate schedule, and one training step."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy: how surprised the model was by each true next token, averaged.

    logits (batch, time, vocab) are flattened to (batch*time, vocab) so every position counts as
    one prediction.
    """
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float = 0.1) -> torch.optim.AdamW:
    """AdamW, with weight decay only on the matrices.

    Weight decay gently pulls weights toward zero so none grows needlessly large. It is applied to
    the 2-D weight matrices (the layers and the embedding table) but not to biases and LayerNorm
    knobs, which are 1-D: pulling those toward zero only hurts.
    """
    matrices = [p for p in model.parameters() if p.dim() >= 2]
    vectors = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": matrices, "weight_decay": weight_decay},
        {"params": vectors, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))


def learning_rate_at(step: int, peak: float, warmup: int, total: int, floor_fraction: float = 0.1) -> float:
    """Warm up linearly to `peak`, then follow a cosine curve down to floor_fraction * peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 at the start, 0 at the end
    floor = floor_fraction * peak
    return floor + (peak - floor) * cosine


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    learning_rate: float,
    clip: float = 1.0,
    precision: torch.dtype | None = None,
) -> float:
    """One update: predict, measure the loss, work out the gradients, nudge every weight.

    precision=torch.bfloat16 runs the forward pass in 16-bit numbers on the GPU (half the memory,
    faster maths); the weights themselves and their updates stay in full 32-bit precision.
    """
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    with torch.autocast(inputs.device.type, dtype=precision or torch.float32, enabled=precision is not None):
        loss = next_token_loss(model(inputs), targets)
    optimizer.zero_grad(set_to_none=True)  # forget the previous step's gradients
    loss.backward()  # backpropagation: the gradient of the loss for every weight
    # If the gradients are unusually large all at once, scale them down so one bad batch
    # can't throw the weights far off.
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()  # move every weight a little against its gradient
    return loss.item()
