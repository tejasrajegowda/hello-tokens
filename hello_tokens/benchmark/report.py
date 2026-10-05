"""The benchmark report: every saved result, as Markdown, with nothing typed by hand.

`bench` saves one JSON file per row (benchmarks/results/) and `judge` one per model
(benchmarks/judge/). This module turns them into tables, so every number in the README traces back to a
saved measurement, and re-measuring a row is enough to update the report.

- Rows are grouped by model (v1, then v2, then any other), then listed from the fewest switches to the
  most, and in measuring order after that. So each model's plain row comes first, and each switch can be
  read against it.
- "Speed-up" compares a row with the same model in plain bf16 (no cache, no other switch): how much each
  technique adds on its own model, at the long-answer point (16-token prompt, 224 new tokens).
- A value that was not measured shows as "–", never as 0.
"""

import json
import re
from pathlib import Path

from hello_tokens.benchmark.harness import POINTS, load_results

LONG = POINTS[1]  # (16, 224): the point quoted in the README
MODEL_ORDER = ("v1", "v2")
START, END = "<!-- benchmarks:start -->", "<!-- benchmarks:end -->"


def _order(result: dict) -> tuple:
    model = result["setup"]["model"]
    rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    return rank, len(result["row"].split()), result["measured_at"]


def _point(result: dict, point: tuple[int, int]) -> dict | None:
    return next((p for p in result["points"] if (p["prompt_tokens"], p["new_tokens"]) == point), None)


def _number(value: float | None, digits: int) -> str:
    return "–" if value is None else f"{value:,.{digits}f}"


def speed_table(results: list[dict]) -> str:
    """One row per saved result, in report order."""
    results = sorted(results, key=_order)
    plain = {r["row"]: r for r in results}
    names = [f"{p}+{n}" for p, n in POINTS]
    lines = [
        "| Row | " + " | ".join(f"tok/s {n}" for n in names)
        + " | Speed-up | First token ms | ms / token | Size MB | Peak MB | Perplexity | ECE |",
        "|---|" + "---:|" * (len(POINTS) + 7),
    ]
    for r in results:
        long = _point(r, LONG)
        base = plain.get(f"{r['setup']['model']} bf16")
        base_long = _point(base, LONG) if base else None
        speedup = (long["tokens_per_s"] / base_long["tokens_per_s"]) if long and base_long else None
        cells = [f"`{r['row']}`"]
        cells += [_number(p["tokens_per_s"], 0) if (p := _point(r, pt)) else "–" for pt in POINTS]
        cells += [
            "–" if speedup is None else f"{speedup:.2f}×",
            _number(long["first_token_ms"] if long else None, 2),
            _number(long["per_token_ms"] if long else None, 2),
            _number(r["model_mb"], 1),
            _number(r["peak_memory_mb"] or None, 0),  # 0 means a CPU run: no GPU memory to report
            _number(r.get("perplexity"), 3),
            _number(r.get("ece"), 4),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def judge_table(judges: list[dict]) -> str:
    lines = [
        "| Model | Questions | Accuracy (all) | Judge ECE raw → scaled | Threshold | Answers | Accuracy when answering | Device |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for j in sorted(judges, key=lambda j: j["row"]):
        lines.append(
            f"| `{j['row']}` | {j['questions']:,} | {j['accuracy_by_method'][j['method']]:.1%} "
            f"| {j['option_ece_raw']:.3f} → {j['option_ece']:.3f} | {j['threshold']:.1%} | {j['answered']:.0%} "
            f"| {j['accuracy_when_answering']:.1%} | {j['device']} |"
        )
    return "\n".join(lines)


def machines(results: list[dict]) -> str:
    """The distinct machines the rows were measured on, so a reader knows what the numbers mean."""
    seen = []
    for r in results:
        m = r.get("machine", {})
        line = ", ".join(x for x in (m.get("gpu") or "CPU only", f"PyTorch {m.get('torch', '?')}",
                                     f"Python {m.get('python', '?')}") if x)
        if line not in seen:
            seen.append(line)
    return "; ".join(seen)


def build_report(results_dir: Path, judge_dir: Path) -> str:
    results = load_results(results_dir) if results_dir.exists() else []
    judges = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(judge_dir.glob("*.json"))] \
        if judge_dir.exists() else []
    parts = [
        "### Speed and quality",
        "",
        f"Greedy writing, no stop token, median of 5 runs; tok/s at (prompt + new tokens). Speed-up, first "
        f"token and ms / token are at {LONG[0]}+{LONG[1]}. Speed-up is against the same model in plain bf16. "
        "Perplexity and ECE are on the held-out text.",
        "",
        speed_table(results) if results else "_No rows measured yet._",
        "",
        f"Measured on: {machines(results) or '–'}.",
    ]
    if judges:
        parts += ["", "### Judge mode", "",
                  "Four-option next-sentence questions from held-out stories; the threshold is chosen on one "
                  "half and reported on the other.", "", judge_table(judges)]
    return "\n".join(parts) + "\n"


def replace_between_markers(text: str, content: str) -> str:
    """Put `content` between the README's start and end markers, replacing what was there."""
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"the markers {START} and {END} were not found")
    return pattern.sub(lambda _: f"{START}\n{content}{END}", text, count=1)
