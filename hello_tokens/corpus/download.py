"""Download the TinyStories text files.

TinyStories (Eldan and Li, 2023) is licensed CDLA-Sharing-1.0. The files are stored under data/,
which is never committed.
"""

import time
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
ATTEMPTS = 20  # a long download over a flaky connection may drop several times

# open_url(url, start) returns a stream of the file's bytes, beginning at byte `start`.
Opener = Callable[[str, int], BinaryIO]


def open_from(url: str, start: int) -> BinaryIO:
    """Open url over HTTP, asking the server to start at byte `start` (a Range request)."""
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", f"bytes={start}-")
    response = urllib.request.urlopen(request, timeout=60)
    if start and response.status != 206:  # 206 Partial Content: the server honoured the range
        response.close()
        raise OSError("the server does not support resuming this download")
    return response


def download(
    url: str,
    destination: Path,
    expected_size: int,
    open_url: Opener = open_from,
    attempts: int = ATTEMPTS,
    wait_seconds: float = 5.0,
) -> bool:
    """Download url to destination, resuming after dropped connections.

    Returns False if a complete copy was already there.
    """
    if destination.exists() and destination.stat().st_size == expected_size:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first, so an unfinished download never looks finished.
    partial = destination.with_name(destination.name + ".part")

    for attempt in range(1, attempts + 1):
        received = partial.stat().st_size if partial.exists() else 0
        if received >= expected_size:
            break
        try:
            with open_url(url, received) as response, open(partial, "ab") as out:
                while chunk := response.read(CHUNK):
                    out.write(chunk)
                    received += len(chunk)
            if received >= expected_size:
                break
        except OSError as error:  # dropped connection, timeout, server error
            print(f"  {destination.name}: interrupted at {received / expected_size:.0%} ({error})")
        print(f"  {destination.name}: resuming from {received:,} bytes (attempt {attempt + 1})")
        time.sleep(wait_seconds)

    size = partial.stat().st_size if partial.exists() else 0
    if size != expected_size:
        raise OSError(f"{destination.name}: expected {expected_size} bytes, got {size}")
    partial.replace(destination)
    return True


def download_all(data_dir: Path, open_url: Opener = open_from) -> None:
    """Fetch every TinyStories file into data_dir/raw, skipping files already complete."""
    for name, size in FILES.items():
        print(f"{name}: fetching ({size / 1e6:,.1f} MB)", flush=True)
        fetched = download(BASE_URL + name, data_dir / "raw" / name, size, open_url)
        print(f"{name}: {'downloaded' if fetched else 'already present'}")
