"""Weight-only quantization, written by hand.

A weight matrix is stored as small integers plus a few scales; the scales restore the real values:
    weight ≈ integer × scale
- int8: one scale per output row. Each weight becomes an integer from -127 to 127 (one byte).
- int4: one scale per group of 64 weights along a row. Each weight becomes -7 to 7, and two of them
  share one byte (4 bits each). Smaller groups follow the weights' local size more closely, which
  matters with only 15 levels.

Symmetric: the scale is the largest absolute value divided by the top integer, so zero stays exactly
zero and no offset is stored. Rounding to the nearest integer means each weight is off by at most half
a scale step.

Only the layers inside the blocks are quantized. The word table is shared by the input and the output
layer and is kept in full precision (a common choice: the output layer is the most sensitive).

The matrices are turned back into normal weights at every use, which adds work in each step: a gain in
memory, not in speed, on this hardware.
"""

import torch
import torch.nn.functional as F
from torch import nn

INT8_MAX = 127
INT4_MAX = 7
GROUP = 64


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) weights -> (int8 values, one scale per row)."""
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT8_MAX
    values = torch.round(weight / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return values, scale


def dequantize_int8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return values.to(scale.dtype) * scale


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Integers in -7..7 -> bytes holding two each (shifted to 1..15 so they fit in 4 unsigned bits).

    The even positions go in the low 4 bits, the odd positions in the high 4 bits.
    """
    shifted = (values + 8).to(torch.uint8)
    return shifted[..., 0::2] | (shifted[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8) - 8
    high = (packed >> 4).to(torch.int8) - 8
    return torch.stack((low, high), dim=-1).flatten(-2)  # interleave back: 0, 1, 2, 3, ...


def quantize_int4(weight: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) weights -> (packed bytes (out, in/2), one scale per group of `group` along each row)."""
    out_features, in_features = weight.shape
    if in_features % group:
        raise ValueError(f"the input width {in_features} is not a multiple of the group size {group}")
    groups = weight.reshape(out_features, in_features // group, group)
    scale = groups.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / INT4_MAX
    values = torch.round(groups / scale).clamp(-INT4_MAX, INT4_MAX).to(torch.int8)
    return pack_int4(values.reshape(out_features, in_features)), scale


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_features, groups = scale.shape[0], scale.shape[1]
    values = unpack_int4(packed).reshape(out_features, groups, -1)
    return (values.to(scale.dtype) * scale).reshape(out_features, -1)


class QuantizedLinear(nn.Module):
    """A linear layer whose weights are stored in int8 or int4 and expanded at each use."""

    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        self.bits = bits
        weight = linear.weight.detach()
        values, scale = quantize_int8(weight) if bits == 8 else quantize_int4(weight)
        # Buffers, not parameters: stored and moved with the model, never trained.
        self.register_buffer("values", values)
        self.register_buffer("scale", scale.to(weight.dtype))
        self.bias = linear.bias

    def weight(self) -> torch.Tensor:
        if self.bits == 8:
            return dequantize_int8(self.values, self.scale)
        return dequantize_int4(self.values, self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight().to(x.dtype), self.bias)


def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    """Replace every linear layer inside the blocks with a quantized one, in place. Returns the model."""
    if bits not in (8, 4):
        raise ValueError("bits must be 8 or 4")
    for block in model.blocks:
        for parent in (block.attention, block.feed_forward):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, QuantizedLinear(child, bits))
    return model
