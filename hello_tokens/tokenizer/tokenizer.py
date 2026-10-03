"""A byte-level BPE tokenizer that trains in reasonable time on real text.

The core algorithm is in bpe.py. Two changes make it practical on millions of stories:

1. Pre-split. Text is first cut into chunks (a word with its leading space, a number, punctuation,
   whitespace), and merges never cross a chunk boundary. Tokens then line up with words, and a word
   repeated a million times is stored once, with its count.
2. The end-of-story marker is a special token: one fixed id, never split, never merged across.
"""

import re
from collections import Counter
from pathlib import Path

from hello_tokens.tokenizer.bpe import Pair, merge

END_OF_TEXT = "<|endoftext|>"

# Every character falls into exactly one branch, so the chunks always join back into the input:
#   letters (with an optional leading space) | digits | other symbols | underscores | whitespace.
# "\s+(?!\S)" keeps the last space of a run of spaces free to start the next word.
SPLIT = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+")

FORMAT = "hello-tokens bpe v1"


def split_chunks(text: str) -> list[str]:
    """Cut text into the chunks that merges are not allowed to cross."""
    return SPLIT.findall(text)


class Tokenizer:
    def __init__(self, merges: dict[Pair, int]):
        self.merges = merges
        # Ids 0-255 are the bytes, then one id per merge, then the end-of-story token last.
        self.end_of_text_id = 256 + len(merges)
        self.vocab_size = self.end_of_text_id + 1
        self.pieces = {i: bytes([i]) for i in range(256)}
        for (left, right), new_id in merges.items():  # merges are in learned order
            self.pieces[new_id] = self.pieces[left] + self.pieces[right]
        # A word repeats millions of times in a corpus; work out its tokens once, then look it up.
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "Tokenizer":
        """Learn merges from text so the vocabulary, special token included, has vocab_size ids."""
        if vocab_size < 257:
            raise ValueError("vocab_size must leave room for the 256 bytes and the special token")
        # Count each distinct chunk once: "the" may occur a million times but is stored once.
        counts = Counter(
            chunk
            for story in text.split(END_OF_TEXT)
            for chunk in split_chunks(story)
        )
        chunks = [list(chunk.encode("utf-8")) for chunk in counts]
        weights = list(counts.values())

        merges: dict[Pair, int] = {}
        for new_id in range(256, vocab_size - 1):
            pair_totals: Counter[Pair] = Counter()
            for ids, weight in zip(chunks, weights):
                for pair in zip(ids, ids[1:]):
                    pair_totals[pair] += weight
            if not pair_totals:
                break  # every chunk is a single token: nothing left to merge
            # Ties go to the pair seen first, so training is repeatable.
            best = max(pair_totals, key=pair_totals.__getitem__)
            chunks = [merge(ids, best, new_id) if len(ids) > 1 else ids for ids in chunks]
            merges[best] = new_id
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if (cached := self._cache.get(chunk)) is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present, the order training used.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids. The end-of-story marker becomes its single special id."""
        ids: list[int] = []
        for i, story in enumerate(text.split(END_OF_TEXT)):
            if i > 0:
                ids.append(self.end_of_text_id)
            for chunk in split_chunks(story):
                ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""
        out = bytearray()
        for i in ids:
            out += END_OF_TEXT.encode("utf-8") if i == self.end_of_text_id else self.pieces[i]
        return out.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        """Write the merges as plain text, one "left right" pair per line, in learned order."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [FORMAT] + [f"{left} {right}" for left, right in self.merges]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        merges = {}
        for new_id, line in enumerate(lines[1:], start=256):
            left, right = map(int, line.split())
            merges[(left, right)] = new_id
        return cls(merges)
