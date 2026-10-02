"""Byte-pair encoding, the core algorithm.

Text is first turned into bytes, so every one of the 256 possible byte values is a token from the
start and any text can be encoded. Training then repeats one step: find the most frequent pair of
neighbouring tokens and replace every occurrence with a new token. Each new token is recorded as a
merge, (left, right) -> new id, and those merges are the whole tokenizer.
"""

Pair = tuple[int, int]


def pair_counts(ids: list[int]) -> dict[Pair, int]:
    """Count how often each pair of neighbouring ids occurs."""
    counts: dict[Pair, int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every occurrence of pair in ids with new_id, scanning left to right."""
    merged = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def train(text: str, vocab_size: int) -> dict[Pair, int]:
    """Learn merges from text until the vocabulary has vocab_size tokens."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256, one token per byte value")
    ids = list(text.encode("utf-8"))
    merges: dict[Pair, int] = {}
    for new_id in range(256, vocab_size):
        counts = pair_counts(ids)
        if not counts:
            break  # the text is a single token: nothing left to merge
        # The most frequent pair. On a tie, max keeps the pair seen first, so training is repeatable.
        best = max(counts, key=counts.get)
        ids = merge(ids, best, new_id)
        merges[best] = new_id
    return merges


def encode(text: str, merges: dict[Pair, int]) -> list[int]:
    """Turn text into token ids by applying the learned merges, earliest-learned first."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # Of the pairs present, apply the one learned earliest: the order training used.
        pair = min(pair_counts(ids), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # no learned merge applies any more
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids: list[int], merges: dict[Pair, int]) -> str:
    """Turn token ids back into text."""
    pieces = {i: bytes([i]) for i in range(256)}
    for (left, right), new_id in merges.items():  # merges are in learned order
        pieces[new_id] = pieces[left] + pieces[right]
    return b"".join(pieces[i] for i in ids).decode("utf-8", errors="replace")
