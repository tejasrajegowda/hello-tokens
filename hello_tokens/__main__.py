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
from hello_tokens.evaluation.calibration import CalibrationBins, save_reliability_diagram
from hello_tokens.evaluation.perplexity import perplexity
from hello_tokens.corpus.sample import read_sample
from hello_tokens.model.config import MODELS
from hello_tokens.quantization.weights import quantize_model
from hello_tokens.tokenizer.tokenizer import Tokenizer
from hello_tokens.generation.power import opt_out_of_power_throttling
from hello_tokens.generation.profile_step import profile_steps
from hello_tokens.generation.sampling import generate
from hello_tokens.generation.speculative import speculative_stream
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
    train_cmd.add_argument("--model", default="v1", choices=list(MODELS), help="model shape preset")
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
    write.add_argument("--cache", action="store_true", help="use the KV cache")
    write.add_argument("--graphs", action="store_true", help="replay each one-token step as a CUDA graph (implies --cache)")
    write.add_argument("--data-dir", type=Path, default=Path("data"))
    eval_cmd = commands.add_parser("eval", help="perplexity of a trained model on every held-out token")
    eval_cmd.add_argument("--name", default="v1", help="which run's checkpoint to use")
    eval_cmd.add_argument("--data-dir", type=Path, default=Path("data"))
    eval_cmd.add_argument("--quantize", type=int, choices=[8, 4], help="store the block weights in int8 or int4")
    profile_cmd = commands.add_parser("profile", help="where the time of one generation step goes")
    profile_cmd.add_argument("--name", default="v1", help="which run's checkpoint to use")
    profile_cmd.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    bench = commands.add_parser("bench", help="measure generation speed on the fixed workload; save a row")
    bench.add_argument("--name", default="v1", help="which run's checkpoint to use")
    bench.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    bench.add_argument("--cache", action="store_true", help="use the KV cache")
    bench.add_argument("--fused", action="store_true", help="use PyTorch's fused attention kernel")
    bench.add_argument("--graphs", action="store_true", help="replay each one-token step as a CUDA graph (implies --cache)")
    bench.add_argument("--speculative", metavar="DRAFT", help="speculative decoding with this draft checkpoint")
    bench.add_argument("--k", type=int, default=4, help="tokens drafted per round (speculative)")
    bench.add_argument("--quantize", type=int, choices=[8, 4], help="store the block weights in int8 or int4")
    bench.add_argument("--skip-quality", action="store_true", help="don't compute held-out perplexity")
    bench.add_argument("--table", action="store_true", help="only print the table of saved rows")
    bench.add_argument("--data-dir", type=Path, default=Path("data"))
    play = commands.add_parser("play", help="open the playground: the models write side by side in a browser")
    play.add_argument("--port", type=int, default=8000)
    play.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    play.add_argument("--data-dir", type=Path, default=Path("data"))
    judge = commands.add_parser("judge", help="multiple-choice test: score the options instead of writing")
    judge.add_argument("--name", default="v2", help="which run's checkpoint to use")
    judge.add_argument("--quantize", type=int, choices=[8, 4], help="store the block weights in int8 or int4")
    judge.add_argument("--questions", type=int, default=2000, help="how many questions to build")
    judge.add_argument("--target", type=float, default=0.9, help="accuracy the switch aims for")
    judge.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    # Without this, Windows slows a long-running process about 3x after a second (see power.py).
    opt_out_of_power_throttling()

    if args.command == "bench":
        return run_bench(args)
    if args.command == "play":
        return run_play(args)
    if args.command == "judge":
        return run_judge_command(args)
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
        if args.quantize:
            quantize_model(model, args.quantize)
        started = time.perf_counter()
        # Scored in float32, not bf16, so the reported number carries no rounding from lower precision.
        bins = CalibrationBins()
        score = perplexity(model, load_tokens(args.data_dir / "tokens" / "valid.bin"), device=device,
                           calibration=bins)
        calibration = bins.result()
        label = args.name + (f" int{args.quantize}" if args.quantize else "")
        print(f"{label} on held-out text: {score.tokens:,} tokens scored in "
              f"{time.perf_counter() - started:.0f} s")
        print(f"loss        {score.loss:.4f}")
        print(f"perplexity  {score.perplexity:.3f}")
        print(f"accuracy    {calibration.accuracy:.1%} (top choice = the true next token)")
        print(f"ECE         {calibration.ece:.4f} (0 = perfectly calibrated)")
        diagram = Path("runs") / args.name / f"reliability{f'-int{args.quantize}' if args.quantize else ''}.png"
        save_reliability_diagram(calibration, diagram, label)
        print(f"diagram     {diagram}")
        return 0

    if args.command == "write":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = Tokenizer.load(args.data_dir / "tokenizer.bpe")
        model = load_model(Path("checkpoints") / f"{args.name}.pt", device)
        new = generate(model, tokenizer.encode(args.prompt), args.max_tokens, args.temperature,
                       args.top_k, args.top_p, stop_id=tokenizer.end_of_text_id, seed=args.seed, cache=args.cache,
                       graphs=args.graphs)
        print(args.prompt + tokenizer.decode(new))
        return 0
    if args.command == "train":
        config = TrainConfig(name=args.name, steps=args.steps, warmup=args.warmup, seed=args.seed)
        train(args.data_dir, Path("."), config, MODELS[args.model])
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
        model = load_model(Path("checkpoints") / f"{args.name}.pt", device)
        if args.quantize:  # round the full-precision weights, then cast the rest
            quantize_model(model, args.quantize)
        model.to(getattr(torch, args.dtype)).use_fused_attention(args.fused)
        if args.graphs and args.speculative:
            print("--graphs applies to normal decoding only")
            return 2
        setup = Setup(args.name, args.dtype, cache="kv" if args.cache or args.graphs or args.speculative else "none",
                      attention="fused" if args.fused else "ours",
                      graphs=("cuda graph" if device == "cuda" else "fixed-shape step, no graph (cpu)")
                      if args.graphs else "none",
                      decoding=f"speculative ({args.speculative}, k={args.k})" if args.speculative else "normal",
                      quantization=f"int{args.quantize}" if args.quantize else "none")
        # The row's name lists what differs from the plain model, e.g. "v2 bf16 cache fused".
        row = " ".join([args.name, "fp32" if args.dtype == "float32" else "bf16"]
                       + ["cache"] * (args.cache or args.graphs) + ["fused"] * args.fused + ["graphs"] * args.graphs
                       + ([f"spec-k{args.k}"] if args.speculative else [])
                       + ([f"int{args.quantize}"] if args.quantize else []))
        print(f"benchmarking {row} on {device}", flush=True)
        write = lambda prompt, n: generate(model, prompt, n, temperature=0, stop_id=None, cache=args.cache,
                                           graphs=args.graphs)
        stats: dict = {}
        if args.speculative:
            draft = load_model(Path("checkpoints") / f"{args.speculative}.pt", device).to(getattr(torch, args.dtype))
            draft.use_fused_attention(args.fused)
            write = lambda prompt, n: [s.id for s in speculative_stream(
                model, draft, prompt, n, temperature=0, stop_id=None, k=args.k, stats=stats)]
        result = run_benchmark(row, setup, model, write, device)
        if stats:
            print(f"draft tokens accepted: {stats['accepted'] / stats['drafted']:.1%}")
        if not args.skip_quality:
            tokens = load_tokens(args.data_dir / "tokens" / "valid.bin")
            bins = CalibrationBins()
            result.perplexity = perplexity(model, tokens, device=device, calibration=bins).perplexity
            result.ece = bins.result().ece
        print(f"saved {save_result(result, RESULTS)}")
    print(format_table(load_results(RESULTS)))
    return 0


def run_judge_command(args) -> int:
    """Build the question set once (data/judge/), score every question, save the result."""
    import json

    from hello_tokens.judge.evaluate import run_judge
    from hello_tokens.judge.questions import build_questions, load_questions, save_questions

    tokenizer = Tokenizer.load(args.data_dir / "tokenizer.bpe")
    path = args.data_dir / "judge" / f"questions-{args.questions}.json"
    if not path.exists():
        text = (args.data_dir / "raw" / "TinyStoriesV2-GPT4-valid.txt").read_text(encoding="utf-8")
        save_questions(build_questions(text, tokenizer, args.questions, seed=0), path)
    questions = load_questions(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path("checkpoints") / f"{args.name}.pt", device)
    if args.quantize:
        quantize_model(model, args.quantize)
    if device == "cuda":
        model.to(torch.bfloat16)
    label = args.name + (f" int{args.quantize}" if args.quantize else "")
    print(f"judge: {label} on {len(questions)} questions ({device})", flush=True)
    result = run_judge(model, tokenizer, questions, args.target,
                       progress=lambda n: print(f"  {n} scored", flush=True))
    result.update(row=label, device=device)
    out = Path("benchmarks") / "judge" / f"{label.replace(' ', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"accuracy, answering everything: {result['accuracy_by_method'][result['method']]:.1%} "
          f"({result['method']} scoring)")
    print(f"the judge's own calibration (ECE): {result['option_ece_raw']:.3f} raw, "
          f"{result['option_ece']:.3f} after temperature scaling (T = {result['temperature']:.1f})")
    print(f"switch at confidence {result['threshold']:.1%}: answers {result['answered']:.1%} of questions, "
          f"{result['accuracy_when_answering']:.1%} of them right (target {args.target:.0%})")
    print(f"{result['ms_per_question']:.1f} ms per question · saved {out}")
    return 0


PLAYGROUND_MODELS = {"v1": "classic GPT", "v2": "modern: RoPE, RMSNorm, SwiGLU, GQA"}


def run_play(args) -> int:
    """Load every trained model that exists, warm it up, and serve the playground on this machine only."""
    import json
    import threading
    import webbrowser

    import uvicorn

    from hello_tokens.serving.app import Entry, create_app

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # bf16 is slow and imprecise on a CPU
    tokenizer = Tokenizer.load(args.data_dir / "tokenizer.bpe")
    benchmarks = {r["row"]: r for r in load_results(RESULTS)} if RESULTS.exists() else {}
    models = {}
    for name, label in PLAYGROUND_MODELS.items():
        path = Path("checkpoints") / f"{name}.pt"
        if not path.exists():
            continue
        model = load_model(path, device).to(dtype)
        generate(model, tokenizer.encode("Once upon a time"), 8, seed=0)  # warm-up: start-up costs paid here
        row = benchmarks.get(f"{name} {'bf16' if dtype == torch.bfloat16 else 'fp32'}")
        # Benchmark rows were measured on the GPU: only show one next to a GPU run.
        speed = next((p["tokens_per_s"] for p in row["points"] if p["new_tokens"] == 224), None) \
            if row and device == "cuda" else None
        judge_file = Path("benchmarks") / "judge" / f"{name}.json"
        judge = json.loads(judge_file.read_text(encoding="utf-8")) if judge_file.exists() else None
        models[name] = Entry(model, label, speed, judge)
    if not models:
        print("no trained models found in checkpoints/: run `train` first")
        return 1
    url = f"http://127.0.0.1:{args.port}"
    print(f"playground: {url}  ({', '.join(models)} on {device}; Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    # 127.0.0.1: reachable from this computer only, never from the network.
    uvicorn.run(create_app(models, tokenizer, device), host="127.0.0.1", port=args.port, log_level="warning")
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
