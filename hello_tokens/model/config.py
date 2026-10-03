"""The model's shape, in one place."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096  # token ids, from the tokenizer
    context: int = 256  # the most tokens the model looks at at once
    width: int = 384  # numbers used to represent each token
    layers: int = 8  # transformer blocks stacked on top of each other
    heads: int = 6  # attention heads per block, each width // heads = 64 wide

    def __post_init__(self):
        if self.width % self.heads:
            raise ValueError("width must divide evenly between the heads")

    @property
    def head_width(self) -> int:
        return self.width // self.heads
