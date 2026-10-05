"""The model's shape, in one place."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096  # token ids, from the tokenizer
    context: int = 256  # the most tokens the model looks at at once
    width: int = 384  # numbers used to represent each token
    layers: int = 8  # transformer blocks stacked on top of each other
    heads: int = 6  # attention heads per block, each width // heads = 64 wide
    # v2 switches. The defaults are v1's design, so a v1 checkpoint (which stores only the fields
    # above) rebuilds exactly the v1 model.
    norm: str = "layernorm"  # or "rmsnorm"
    position: str = "learned"  # learned position table, or "rope" (rotary)
    feed_forward: str = "gelu"  # or "swiglu"
    kv_heads: int | None = None  # key/value heads; None = one per query head (v1). Fewer = GQA

    def __post_init__(self):
        if self.width % self.heads:
            raise ValueError("width must divide evenly between the heads")
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.position not in ("learned", "rope"):
            raise ValueError(f"unknown position {self.position!r}")
        if self.feed_forward not in ("gelu", "swiglu"):
            raise ValueError(f"unknown feed_forward {self.feed_forward!r}")
        if self.heads % self.kv_head_count:
            raise ValueError("the query heads must divide evenly between the key/value heads")

    @property
    def head_width(self) -> int:
        return self.width // self.heads

    @property
    def kv_head_count(self) -> int:
        return self.kv_heads or self.heads


# Named shapes for `train --model`. Each adds one change to the one before it, so training them in
# order shows what each change does on its own (an ablation); "v2" has all four.
PRESETS = {
    "v1": ModelConfig(),
    "rmsnorm": ModelConfig(norm="rmsnorm"),
    "rope": ModelConfig(norm="rmsnorm", position="rope"),
    "swiglu": ModelConfig(norm="rmsnorm", position="rope", feed_forward="swiglu"),
    "v2": ModelConfig(norm="rmsnorm", position="rope", feed_forward="swiglu", kv_heads=2),
}

# The draft model for speculative decoding: tiny, so each guess is cheap. Classic parts on purpose:
# a step's cost here is the number of GPU launches, and LayerNorm (one fused kernel) and a learned
# position table launch fewer than RMSNorm and RoPE. Same vocabulary and context as the big models.
DRAFT = ModelConfig(width=192, layers=2, heads=3)

# Every shape `train --model` accepts.
MODELS = {**PRESETS, "draft": DRAFT}
