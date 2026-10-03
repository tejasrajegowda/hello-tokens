"""Turn a TinyStories text file into a file of token ids, two bytes per token.

The vocabulary has 4,096 ids, all below 65,536, so every id fits in an unsigned 16-bit integer
(uint16). That halves the size of 32-bit storage: the training text becomes one flat array of ids
that training can read directly from disk.
"""

from pathlib import Path

import numpy as np

from hello_tokens.tokenizer.tokenizer import END_OF_TEXT, Tokenizer

BLOCK = 16_000_000  # read about 16 MB of text at a time


def encode_file(source: Path, destination: Path, tokenizer: Tokenizer, block: int = BLOCK) -> int:
    """Encode source into destination as uint16 token ids. Returns the number of tokens written."""
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("the vocabulary is too large for 16-bit token ids")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    total = 0
    carry = ""  # the unfinished story at the end of a block, saved for the next block
    # newline="" reads line endings exactly as stored (no Windows translation).
    with open(source, encoding="utf-8", newline="") as text, open(partial, "wb") as out:
        while piece := text.read(block):
            piece = carry + piece
            # Encode only whole stories, so a block boundary never splits a word or a story.
            cut = piece.rfind(END_OF_TEXT)
            if cut == -1:
                carry = piece
                continue
            cut += len(END_OF_TEXT)
            ids = tokenizer.encode(piece[:cut])
            np.array(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
            carry = piece[cut:]
        if carry:
            ids = tokenizer.encode(carry)
            np.array(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
    partial.replace(destination)
    return total


def load_tokens(path: Path) -> np.ndarray:
    """Open a token file without reading it into memory (the operating system pages it in)."""
    return np.memmap(path, dtype=np.uint16, mode="r")
