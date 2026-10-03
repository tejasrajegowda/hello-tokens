"""Command-line entry point: python -m hello_tokens <command>."""

import argparse
import platform
import sys
import time
from pathlib import Path

import torch

from hello_tokens.benchmark.harness import Setup, format_table, load_results, run_benchmark, save_result
from hello_tokens.corpus.download import download_all
from hello_tokens.corpus.encode import encode_file, load_tokens
from hello_tokens.evaluation.perplexity import perplexity
from hello_tokens.corpus.sample import read_sample
from hello_tokens.model.config import PRESETS
from hello_tokens.tokenizer.tokenizer import Tokenizer
from hello_tokens.generation.power import opt_out_of_power_throttling
from hello_tokens.generation.profile_step import profile_steps
from hello_tokens.generation.sampling import generate
from hello_tokens.training.run import TrainConfig, load_model, train


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
    train_cmd = commands.add_parser("train", help="train the model (resumes a run of the same name)")
    train_cmd.add_argument("--name", default="v1", help="run name: checkpoints/<name>.pt, runs/<name>/")
    train_cmd.add_argument("--model", default="v1", choices=list(PRESETS), help="model shape preset")
    train_cmd.add_argument("--steps", type=int, default=TrainConfig.steps)
    train_cmd.add_argument("--warmup", type=int, default=TrainConfig.warmup, help="warm-up steps")
    train_cmd.add_argument("--seed", type=int, default=TrainConfig.seed)
    train_cmd.add_argument("--data-dir", type=Path, default=Path("data"))
    write = commands.add_parser("write", help="write a story with a trained model")
    write.add_argument("prompt", nargs="?", default="Once upon a time")
    write.add_argument("--name", default="v1", help="which run's checkpoint to use")
    write.add_argument("--max-tokens", type=int, default=300)
    write.add_argument("--temperature", type=float, default=0.8)
    write.add_argument("--top-k", type=int, default=None)
    write.add_argument("--top-p", type=float, default=0.95)
    write.add_argument("--seed", type=int, default=None)
    write.add_argument("--data-dir", type=Path, default=Path("data"))
    eval_cmd = commands.add_parser("eval", help="perplexity of a trained model on every held-out token")
    eval_cmd.add_argument("--name", default="v1", help="which run's checkpoint to use")
    eval_cmd.add_argument("--data-dir", type=Path, default=Path("data"))
    profile_cmd = commands.add_parser("profile", help="where the time of one generation step goes")
    profile_cmd.add_argument("--name", default="v1", help="which run's checkpoint to use")
    profile_cmd.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    bench = commands.add_parser("bench", help="measure generation speed on the fixed workload; save a row")
    bench.add_argument("--name", default="v1", help="which run's checkpoint to use")
    bench.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    bench.add_argument("--skip-quality", action="store_true", help="don't compute held-out perplexity")
    bench.add_argument("--table", action="store_true", help="only print the table of saved rows")
    bench.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    # Without this, Windows slows a long-running process about 3x after a second (see power.py).
    opt_out_of_power_throttling()

    if args.command == "bench":
        return run_bench(args)
    if args.command == "profile":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(Path("checkpoints") / f"{args.name}.pt", device).to(getattr(torch, args.dtype))
        print(f"{args.name}, {args.dtype}, {device}: one generation step (forward pass + sampling)")
        print(" context   wall ms  (spread)   GPU busy ms   kernels   GPU idle")
        for r in profile_steps(model, contexts=[16, 64, 128, 256]):
            print(f"{r.context:>8}  {r.wall_ms:8.2f}  ({r.spread_ms:5.2f})  {r.gpu_busy_ms:12.2f}"
                  f"  {r.kernels:8.0f}  {r.gpu_idle_fraction:8.0%}")
        return 0

    if args.command == "eval":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(Path("checkpoints") / f"{args.name}.pt", device)
        started = time.perf_counter()
        # Scored in float32, not bf16, so the reported number carries no rounding from lower precision.
        score = perplexity(model, load_tokens(args.data_dir / "tokens" / "valid.bin"), device=device)
        print(f"{args.name} on held-out text: {score.tokens:,} tokens scored in "
              f"{time.perf_counter() - started:.0f} s")
        print(f"loss        {score.loss:.4f}")
        print(f"perplexity  {score.perplexity:.3f}")
        return 0

    if args.command == "write":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = Tokenizer.load(args.data_dir / "tokenizer.bpe")
        model = load_model(Path("checkpoints") / f"{args.name}.pt", device)
        new = generate(model, tokenizer.encode(args.prompt), args.max_tokens, args.temperature,
                       args.top_k, args.top_p, stop_id=tokenizer.end_of_text_id, seed=args.seed)
        print(args.prompt + tokenizer.decode(new))
        return 0
    if args.command == "train":
        config = TrainConfig(name=args.name, steps=args.steps, warmup=args.warmup, seed=args.seed)
        train(args.data_dir, Path("."), config, PRESETS[args.model])
        return 0
    if args.command == "check":
        return check()
    if args.command == "prepare":
        download_all(args.data_dir)
        train_tokenizer(args.data_dir, args.sample_mb, args.vocab_size)
        encode_corpus(args.data_dir)
        return 0
    return 2


RESULTS = Path("benchmarks") / "results"


def run_bench(args) -> int:
    """Benchmark one checkpoint as one row, save it, and print every saved row."""
    if not args.table:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(Path("checkpoints") / f"{args.name}.pt", device).to(getattr(torch, args.dtype))
        row = f"{args.name} {'fp32' if args.dtype == 'float32' else 'bf16'}"
        print(f"benchmarking {row} on {device}", flush=True)
        result = run_benchmark(row, Setup(args.name, args.dtype), model,
                               lambda prompt, n: generate(model, prompt, n, temperature=0, stop_id=None), device)
        if not args.skip_quality:
            tokens = load_tokens(args.data_dir / "tokens" / "valid.bin")
            result.perplexity = perplexity(model, tokens, device=device).perplexity
        print(f"saved {save_result(result, RESULTS)}")
    print(format_table(load_results(RESULTS)))
    return 0


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
