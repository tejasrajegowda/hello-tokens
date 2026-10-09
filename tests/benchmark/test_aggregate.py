import json

import pytest

from hello_tokens.benchmark.aggregate import combine, passes_anchor, select_and_combine


def row(name, per_token_ms, measured_at="2026-01-01T00:00:00", perplexity=None, model_mb=28.6):
    return {
        "row": name,
        "setup": {"model": name.split()[0], "dtype": "bfloat16"},
        "points": [{"prompt_tokens": 16, "new_tokens": 224, "first_token_ms": 10.0, "per_token_ms": per_token_ms,
                    "tokens_per_s": 1000 / per_token_ms, "spread_ms": 1.0, "identical_runs": True}],
        "model_mb": model_mb,
        "peak_memory_mb": 40.0,
        "perplexity": perplexity,
        "ece": None,
        "machine": {"gpu": "Test GPU"},
        "measured_at": measured_at,
    }


def test_each_value_is_the_median_across_batches_and_the_spread_between_them_is_kept():
    batches = [[row("v1 bf16", ms, f"2026-01-0{i}T00:00:00", perplexity=3.6)]
               for i, ms in enumerate([5.0, 6.0, 5.5, 8.0, 5.2], start=1)]
    [result] = combine(batches)
    point = result["points"][0]
    assert point["per_token_ms"] == 5.5  # the slow 8.0 batch does not pull the number up
    assert point["tokens_per_s"] == pytest.approx(1000 / 5.5)
    assert 0 < point["batch_spread_pct"] < 50
    assert result["batches"] == 5 and result["perplexity"] == 3.6
    assert result["measured_at"] == "2026-01-05T00:00:00"  # keeps the measuring order of the last batch


def test_batches_with_different_rows_or_setups_are_refused():
    with pytest.raises(ValueError, match="different rows"):
        combine([[row("v1 bf16", 6.0)], [row("v1 bf16", 6.0), row("v2 bf16", 13.0)]])
    with pytest.raises(ValueError, match="different setups"):
        combine([[row("v1 bf16", 6.0)], [row("v1 bf16", 6.0, model_mb=16.1)]])


def test_a_batch_is_used_only_when_its_anchor_row_matches_the_quiet_reference():
    assert passes_anchor([row("v1 bf16", 6.2)], reference_ms=5.91, tolerance=0.10)
    assert not passes_anchor([row("v1 bf16", 7.25)], reference_ms=5.91, tolerance=0.10)
    with pytest.raises(ValueError, match="no 'v1 bf16' row"):
        passes_anchor([row("v2 bf16", 13.0)])


def test_selection_reads_every_batch_folder_and_reports_each_anchor(tmp_path):
    for name, ms in {"0100": 5.9, "0200": 12.5, "0300": 6.1}.items():
        folder = tmp_path / name
        folder.mkdir()
        for r in (row("v1 bf16", ms), row("v2 bf16", ms * 2)):
            (folder / (r["row"].replace(" ", "-") + ".json")).write_text(json.dumps(r), encoding="utf-8")
    rows, anchors, used = select_and_combine(tmp_path, reference_ms=5.91, tolerance=0.10)
    assert used == ["0100", "0300"] and anchors["0200"] == 12.5
    assert {r["row"]: r["points"][0]["per_token_ms"] for r in rows} == {"v1 bf16": 6.0, "v2 bf16": 12.0}
