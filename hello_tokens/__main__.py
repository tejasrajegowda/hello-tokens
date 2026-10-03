"""Command-line entry point: python -m hello_tokens <command>."""

import argparse
import platform
import sys
import time
from pathlib import Path

import torch

from hello_tokens.corpus.download import download_all
from hello_tokens.corpus.encode import encode_file
from hello_tokens.corpus.sample import read_sample
from hello_tokens.tokenizer.tokenizer import Tokenizer


def check() -> int:
    """Report the Python and PyTorch versions, and confirm the GPU can run PyTorch code."""
    print(f"python   {platform.python_version()}")
    print(f"torch    {torch.__version__}")
    if not torch.cuda.is_available():
        print("gpu      not available: training would fall back to the CPU")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"gpu      {props.name}, {props.total_memory / 2**30:.1f} GiB")

    # Run one real computation on the GPU. .item() copies the answer back to the CPU,
    # which forces the GPU to finish the work, so a broken setup fails here, not later.
    matrix = torch.randn(1024, 1024, device="cuda")
    (matrix @ matrix).sum().item()
    print("compute  ok: a 1024x1024 matrix product ran on the GPU")

    bf16 = "supported" if torch.cuda.is_bf16_supported() else "not supported"
    print(f"bf16     {bf16}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hello_tokens")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check that Python, PyTorch and the GPU are ready")
    prepare = commands.add_parser("prepare", help="download TinyStories and train the tokenizer")
    prepare.add_argument("--data-dir", type=Path, default=Path("data"), help="where to store it")
    prepare.add_argument("--sample-mb", type=float, default=20, help="text used to train the tokenizer")
    prepare.add_argument("--vocab-size", type=int, default=4096, help="tokenizer vocabulary size")
    args = parser.parse_args(argv)

    if args.command == "check":
        return check()
    if args.command == "prepare":
        download_all(args.data_dir)
        train_tokenizer(args.data_dir, args.sample_mb, args.vocab_size)
        encode_corpus(args.data_dir)
        return 0
    return 2


def encode_corpus(data_dir: Path) -> None:
    """Encode the training and validation text into token files, unless they already exist."""
    tokenizer = Tokenizer.load(data_dir / "tokenizer.bpe")
    for split in ("train", "valid"):
        destination = data_dir / "tokens" / f"{split}.bin"
        if destination.exists():
            print(f"tokens: {split} already present ({destination})")
            continue
        source = data_dir / "raw" / f"TinyStoriesV2-GPT4-{split}.txt"
        print(f"tokens: encoding {source.name}", flush=True)
        started = time.perf_counter()
        count = encode_file(source, destination, tokenizer)
        print(f"tokens: {split} {count:,} tokens in {time.perf_counter() - started:.0f} s -> {destination}")


def train_tokenizer(data_dir: Path, sample_mb: float, vocab_size: int) -> None:
    """Train the tokenizer on a sample of the training stories, unless it already exists."""
    path = data_dir / "tokenizer.bpe"
    if path.exists():
        print(f"tokenizer: already present ({path})")
        return
    text = read_sample(data_dir / "raw" / "TinyStoriesV2-GPT4-train.txt", sample_mb)
    print(f"tokenizer: training on {sample_mb:g} MB, vocabulary {vocab_size}", flush=True)
    started = time.perf_counter()
    tokenizer = Tokenizer.train(text, vocab_size)
    tokenizer.save(path)
    print(f"tokenizer: {tokenizer.vocab_size} tokens in {time.perf_counter() - started:.0f} s -> {path}")


if __name__ == "__main__":
    sys.exit(main())
