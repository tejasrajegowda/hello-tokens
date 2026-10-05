"""Run every GPU measurement of v2 in order, resumably, with one command.

    uv run python scripts/gpu_benchmarks.py            # run (or resume) the batch
    uv run python scripts/gpu_benchmarks.py --dry-run  # only list the steps and which are done

Each step is one `python -m hello_tokens` command (or the GPU-only tests). A finished step is recorded in
runs/gpu-benchmarks-done.txt and never runs again, so an interrupted batch resumes where it stopped. The
first failure stops the batch. Every line of output also goes to runs/gpu-benchmarks.log.

Quality (perplexity and ECE) is measured for rows that change the numbers the model computes: the
precision, the attention kernel and quantization. The KV cache, CUDA graphs and speculative decoding give
the same distribution as their model's plain row (exact in fp32, as the tests show), so those rows skip it.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs")
DONE = RUNS / "gpu-benchmarks-done.txt"
LOG = RUNS / "gpu-benchmarks.log"
BF16 = ["--dtype", "bfloat16"]
SAME_QUALITY = ["--skip-quality"]


def steps() -> list[tuple[str, list[str]]]:
    """(name, command) pairs, in order. A command starting with "pytest" runs the tests."""
    plan: list[tuple[str, list[str]]] = [
        # The CUDA graph replay can only be checked on a GPU; nothing is measured until it passes.
        ("graph tests", ["pytest", "-q", "tests/generation/test_graphs.py"]),
        ("train draft", ["train", "--name", "draft", "--model", "draft", "--steps", "20000"]),
        ("eval draft", ["eval", "--name", "draft"]),
    ]
    for model in ("v1", "v2"):
        bench = ["bench", "--name", model, *BF16]
        plan += [
            (f"bench {model} bf16", bench),  # again, now that ECE is measured too
            (f"bench {model} cache", bench + ["--cache", *SAME_QUALITY]),
            (f"bench {model} fused", bench + ["--fused"]),
            (f"bench {model} cache fused", bench + ["--cache", "--fused", *SAME_QUALITY]),
            (f"bench {model} graphs", bench + ["--graphs", *SAME_QUALITY]),
            (f"bench {model} graphs fused", bench + ["--graphs", "--fused", *SAME_QUALITY]),
        ]
    v2 = ["bench", "--name", "v2", *BF16]
    for k in (2, 4, 6):
        plan.append((f"bench v2 speculative k{k}", v2 + ["--speculative", "draft", "--k", str(k), *SAME_QUALITY]))
    plan.append(("bench v2 speculative k4 fused", v2 + ["--speculative", "draft", "--k", "4", "--fused", *SAME_QUALITY]))
    for bits in (8, 4):
        plan += [
            (f"eval v2 int{bits}", ["eval", "--name", "v2", "--quantize", str(bits)]),
            (f"bench v2 int{bits}", v2 + ["--quantize", str(bits)]),
            # Quantized weights are expanded at every use, adding launches; graphs remove launch cost.
            (f"bench v2 int{bits} graphs", v2 + ["--quantize", str(bits), "--graphs", *SAME_QUALITY]),
        ]
    plan.append(("report", ["report"]))
    return plan


def done_steps() -> set[str]:
    return set(DONE.read_text(encoding="utf-8").splitlines()) if DONE.exists() else set()


def run(name: str, command: list[str], log) -> bool:
    if command[0] == "pytest":
        argv = [sys.executable, "-m", *command]
    else:
        argv = [sys.executable, "-m", "hello_tokens", *command]
    header = f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {name}: {' '.join(command)}"
    print(header, flush=True)
    log.write(header + "\n")
    started = time.perf_counter()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace")
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    ok = process.wait() == 0
    footer = f"=== {name}: {'done' if ok else 'FAILED'} in {time.perf_counter() - started:.0f} s"
    print(footer, flush=True)
    log.write(footer + "\n")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list the steps and which are done")
    args = parser.parse_args()
    finished = done_steps()
    plan = steps()
    if args.dry_run:
        for name, command in plan:
            print(f"{'done' if name in finished else 'todo':<5} {name:<32} {' '.join(command)}")
        return 0

    import torch

    if not torch.cuda.is_available():
        print("no CUDA GPU visible: this batch measures GPU speed, so it does not run on a CPU")
        return 1
    RUNS.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        for name, command in plan:
            if name in finished:
                continue
            if not run(name, command, log):
                print(f"\nstopped at '{name}'. Fix it and run the script again: finished steps are skipped.")
                return 1
            with DONE.open("a", encoding="utf-8") as done:
                done.write(name + "\n")
    print("\nall steps done. The tables are in docs/benchmarks.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
