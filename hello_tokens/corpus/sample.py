"""Read a sample of whole stories from the start of a TinyStories file."""

from pathlib import Path

from hello_tokens.tokenizer.tokenizer import END_OF_TEXT


def read_sample(path: Path, megabytes: float) -> str:
    """Return about `megabytes` of text, cut at the last complete story."""
    with open(path, "rb") as f:
        raw = f.read(int(megabytes * 1_000_000))
    # The cut may land inside a multi-byte character; drop the broken tail, then the partial story.
    text = raw.decode("utf-8", errors="ignore")
    end = text.rfind(END_OF_TEXT)
    return text[:end] if end != -1 else text
