import json

import pytest

from hello_tokens.benchmark.report import (
    END, START, build_report, judge_table, replace_between_markers, speed_table,
)


def row(name, model, tok_s, measured_at, perplexity=None, ece=None, peak=40.0, points=None):
    points = points or [(16, 64), (16, 224), (128, 128)]
    return {
        "row": name,
        "setup": {"model": model},
        "points": [{"prompt_tokens": p, "new_tokens": n, "first_token_ms": 10.0, "per_token_ms": 1000 / tok_s,
                    "tokens_per_s": tok_s, "spread_ms": 1.0, "identical_runs": True} for p, n in points],
        "model_mb": 28.6,
        "peak_memory_mb": peak,
        "perplexity": perplexity,
        "ece": ece,
        "machine": {"gpu": "Test GPU", "torch": "2.0", "python": "3.14"},
        "measured_at": measured_at,
    }


def table_rows(table):
    return [line.split("|")[1].strip().strip("`") for line in table.splitlines()[2:]]


def test_rows_are_grouped_by_model_then_by_number_of_switches():
    results = [
        row("v2 bf16 cache graphs", "v2", 300, "2026-01-05"),
        row("v1 bf16", "v1", 150, "2026-01-02"),
        row("draft bf16", "draft", 900, "2026-01-01"),
        row("v2 bf16", "v2", 75, "2026-01-03"),
        row("v1 fp32", "v1", 140, "2026-01-01"),
        row("v2 bf16 cache", "v2", 80, "2026-01-04"),
    ]
    assert table_rows(speed_table(results)) == [
        "v1 fp32", "v1 bf16", "v2 bf16", "v2 bf16 cache", "v2 bf16 cache graphs", "draft bf16"]


def test_speed_up_is_against_the_same_model_in_plain_bf16():
    table = speed_table([row("v2 bf16", "v2", 75, "1"), row("v2 bf16 cache graphs", "v2", 300, "2"),
                         row("v1 fp32", "v1", 140, "0")])
    lines = {r: line for r, line in zip(table_rows(table), table.splitlines()[2:])}
    assert "| 1.00× |" in lines["v2 bf16"]
    assert "| 4.00× |" in lines["v2 bf16 cache graphs"]
    assert "| – |" in lines["v1 fp32"]  # no "v1 bf16" row to compare with


def test_values_that_were_not_measured_show_as_a_dash_never_as_zero():
    table = speed_table([row("v2 bf16", "v2", 75, "1", perplexity=3.568, ece=None, peak=0.0,
                             points=[(16, 64), (16, 224)])])
    cells = [c.strip() for c in table.splitlines()[2].split("|")[1:-1]]
    assert cells[3] == "–"  # the (128, 128) point is missing
    assert cells[8] == "–"  # peak memory 0: a CPU run
    assert cells[9] == "3.568"
    assert cells[10] == "–"  # no ECE


def test_the_judge_table_reports_the_held_out_split():
    judge = {"row": "v2", "questions": 2000, "method": "summed", "accuracy_by_method": {"summed": 0.72},
             "option_ece_raw": 0.22, "option_ece": 0.057, "threshold": 0.767, "answered": 0.32,
             "accuracy_when_answering": 0.922, "device": "cpu"}
    line = judge_table([judge]).splitlines()[2]
    assert "72.0%" in line and "0.220 → 0.057" in line and "32%" in line and "92.2%" in line


def test_the_report_reads_the_saved_files(tmp_path):
    results, judges = tmp_path / "results", tmp_path / "judge"
    results.mkdir()
    (results / "v2-bf16.json").write_text(json.dumps(row("v2 bf16", "v2", 75, "1")), encoding="utf-8")
    report = build_report(results, judges)
    assert "`v2 bf16`" in report and "Test GPU" in report and "Judge mode" not in report
    assert "No rows measured yet" in build_report(tmp_path / "missing", judges)


def test_markers_are_replaced_and_the_rest_of_the_text_is_kept():
    text = f"# Title\n\nintro\n{START}\nold table\n{END}\n\noutro\n"
    once = replace_between_markers(text, "new table\n")
    assert once == f"# Title\n\nintro\n{START}\nnew table\n{END}\n\noutro\n"
    assert replace_between_markers(once, "new table\n") == once  # running it again changes nothing
    with pytest.raises(ValueError):
        replace_between_markers("no markers here", "x")
