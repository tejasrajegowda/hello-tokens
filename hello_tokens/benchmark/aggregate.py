"""Combine several benchmark batches into one result per row.

A single batch can be disturbed by other work on the machine, and its own spread (five runs within
minutes) does not show that: a batch can be consistently slow. So when a machine cannot be kept quiet,
the same batch is run several times and combined here:

- A batch is used only if its anchor row (v1 bf16, the simplest GPU row) is within a tolerance of a
  reference measured on a quiet machine. The rule is fixed before the batches are combined, and every
  batch is kept on disk, so the selection can be checked.
- Each value at each point is the median across the used batches.
- The spread between batches (interquartile range of tokens/s, as a share of the median) is saved with
  each point and shown in the report, so the uncertainty of every number is visible.
"""

import statistics
from pathlib import Path

from hello_tokens.benchmark.harness import load_results

ANCHOR_ROW = "v1 bf16"
ANCHOR_POINT = (16, 224)
ANCHOR_REFERENCE_MS = 5.91  # v1 bf16 ms/token at 16+224, measured on a quiet machine
ANCHOR_TOLERANCE = 0.10

POINT_VALUES = ("first_token_ms", "per_token_ms", "tokens_per_s", "spread_ms")


def _median_and_iqr(values: list[float]) -> tuple[float, float]:
    quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return statistics.median(values), quartiles[2] - quartiles[0]


def anchor_ms(batch: list[dict], row: str = ANCHOR_ROW, point: tuple[int, int] = ANCHOR_POINT) -> float:
    """The anchor row's per-token time in one batch."""
    result = next((r for r in batch if r["row"] == row), None)
    if result is None:
        raise ValueError(f"the batch has no {row!r} row")
    return next(p["per_token_ms"] for p in result["points"] if (p["prompt_tokens"], p["new_tokens"]) == point)


def passes_anchor(batch: list[dict], reference_ms: float = ANCHOR_REFERENCE_MS,
                  tolerance: float = ANCHOR_TOLERANCE) -> bool:
    return abs(anchor_ms(batch) - reference_ms) <= tolerance * reference_ms


def combine(batches: list[list[dict]]) -> list[dict]:
    """One result per row: the median of every point value across the batches.

    Every batch must hold the same rows with the same setup, size and points; anything else would
    combine different measurements into one number."""
    if not batches:
        raise ValueError("no batches to combine")
    rows = [{r["row"]: r for r in batch} for batch in batches]
    names = set(rows[0])
    for other in rows[1:]:
        if set(other) != names:
            raise ValueError(f"the batches hold different rows: {sorted(names ^ set(other))}")

    combined = []
    for name in sorted(names):
        versions = [batch[name] for batch in rows]
        first = versions[0]
        for v in versions[1:]:
            if v["setup"] != first["setup"] or v["model_mb"] != first["model_mb"]:
                raise ValueError(f"{name!r} was measured with different setups across batches")
        points = []
        for i, point in enumerate(first["points"]):
            same = [v["points"][i] for v in versions]
            if any((p["prompt_tokens"], p["new_tokens"]) != (point["prompt_tokens"], point["new_tokens"])
                   for p in same):
                raise ValueError(f"{name!r} was measured at different points across batches")
            merged = {"prompt_tokens": point["prompt_tokens"], "new_tokens": point["new_tokens"]}
            for key in POINT_VALUES:
                merged[key] = statistics.median(p[key] for p in same)
            speed, iqr = _median_and_iqr([p["tokens_per_s"] for p in same])
            merged["batch_spread_pct"] = 100 * iqr / speed
            merged["identical_runs"] = all(p["identical_runs"] for p in same)
            points.append(merged)
        quality = {key: statistics.median(vals) if (vals := [v[key] for v in versions if v.get(key) is not None])
                   else None for key in ("perplexity", "ece")}
        combined.append({
            "row": name,
            "setup": first["setup"],
            "points": points,
            "model_mb": first["model_mb"],
            "peak_memory_mb": statistics.median(v["peak_memory_mb"] for v in versions),
            **quality,
            "machine": first["machine"],
            # The order of measuring is kept: a row takes its time from the last batch.
            "measured_at": versions[-1]["measured_at"],
            "batches": len(versions),
        })
    return combined


def select_and_combine(directory: Path, reference_ms: float = ANCHOR_REFERENCE_MS,
                       tolerance: float = ANCHOR_TOLERANCE) -> tuple[list[dict], dict[str, float], list[str]]:
    """Read every batch folder in `directory`, combine those that pass the anchor check.

    Returns the combined rows, each batch's anchor time, and the names of the batches used."""
    folders = sorted(p for p in directory.iterdir() if p.is_dir())
    batches = {f.name: load_results(f) for f in folders}
    anchors = {name: anchor_ms(batch) for name, batch in batches.items()}
    used = [name for name, batch in batches.items() if passes_anchor(batch, reference_ms, tolerance)]
    return combine([batches[name] for name in used]), anchors, used
