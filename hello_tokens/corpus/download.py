"""Download the TinyStories text files.

TinyStories (Eldan and Li, 2023) is licensed CDLA-Sharing-1.0. The files are stored under data/,
which is never committed.
"""

import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"

# The GPT-4-only generations (the cleaner part of the dataset), with their exact sizes in bytes.
FILES = {
    "TinyStoriesV2-GPT4-train.txt": 2_227_753_162,
    "TinyStoriesV2-GPT4-valid.txt": 22_502_601,
}

CHUNK = 1 << 20  # read 1 MiB at a time, so the whole file is never held in memory


def download(
    url: str,
    destination: Path,
    expected_size: int,
    open_url: Callable[[str], BinaryIO] = urllib.request.urlopen,
) -> bool:
    """Download url to destination. Returns False if a complete copy was already there."""
    if destination.exists() and destination.stat().st_size == expected_size:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first, so an interrupted download never looks finished.
    partial = destination.with_name(destination.name + ".part")
    received = 0
    next_report = 0.0
    with open_url(url) as response, open(partial, "wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            received += len(chunk)
            if received / expected_size >= next_report:
                print(f"  {destination.name}: {received / expected_size:4.0%}", flush=True)
                next_report += 0.1

    if received != expected_size:
        raise OSError(f"{destination.name}: expected {expected_size} bytes, got {received}")
    partial.replace(destination)
    return True


def download_all(data_dir: Path, open_url: Callable[[str], BinaryIO] = urllib.request.urlopen) -> None:
    """Fetch every TinyStories file into data_dir/raw, skipping files already complete."""
    for name, size in FILES.items():
        fetched = download(BASE_URL + name, data_dir / "raw" / name, size, open_url)
        print(f"{name}: {'downloaded' if fetched else 'already present'} ({size / 1e6:,.1f} MB)")
