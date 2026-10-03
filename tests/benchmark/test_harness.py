import pytest
import torch

from hello_tokens.benchmark.harness import (
    POINTS, Setup, format_table, load_results, measure_point, model_megabytes, run_benchmark, save_result,
)
from hello_tokens.generation.sampling import generate
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

TINY = ModelConfig(vocab_size=64, context=16, width=32, layers=2, heads=4)
SMALL_POINTS = ((4, 6), (8, 8))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY)


def greedy_writer(model):
    return lambda prompt, n: generate(model, prompt, n, temperature=0, stop_id=None)


def test_a_benchmark_measures_every_point_with_sane_values(model):
    result = run_benchmark("tiny", Setup("tiny", "float32"), model, greedy_writer(model), "cpu",
                           points=SMALL_POINTS, repeats=3)
    assert [(p.prompt_tokens, p.new_tokens) for p in result.points] == list(SMALL_POINTS)
    for p in result.points:
        assert p.identical_runs  # greedy, fixed prompt: the same tokens every run
        assert p.first_token_ms > 0 and p.per_token_ms > 0 and p.tokens_per_s > 0
    assert result.model_mb == pytest.approx(model_megabytes(model))
    assert result.peak_memory_mb == 0  # CPU


def test_the_model_size_counts_every_weight_once(model):
    # float32: 4 bytes per parameter; the tied table is shared, so it is counted once.
    assert model_megabytes(model) == pytest.approx(model.parameter_count() * 4 / 1e6)


def test_a_writer_that_stops_early_is_rejected():
    with pytest.raises(ValueError):
        measure_point(lambda prompt, n: [0] * (n - 1), [1, 2], 5, "cpu", repeats=1)


def test_a_writer_that_varies_is_reported(model):
    calls = iter(range(1000))
    point = measure_point(lambda prompt, n: [next(calls)] * n, [1, 2], 3, "cpu", repeats=2)
    assert not point.identical_runs


def test_results_save_load_and_print(model, tmp_path):
    result = run_benchmark("tiny fp32", Setup("tiny", "float32"), model, greedy_writer(model), "cpu",
                           points=POINTS[:1], repeats=2)
    path = save_result(result, tmp_path)
    assert path.name == "tiny-fp32.json"
    (loaded,) = load_results(tmp_path)
    assert loaded["setup"]["dtype"] == "float32"
    table = format_table([loaded])
    assert "tiny fp32" in table and "-" in table  # the points not measured show as "-"


def test_rows_are_listed_in_the_order_they_were_measured(model, tmp_path):
    for row, when in (("a later", "2026-01-02T00:00:00"), ("b earlier", "2026-01-01T00:00:00")):
        result = run_benchmark(row, Setup("tiny", "float32"), model, greedy_writer(model), "cpu",
                               points=SMALL_POINTS[:1], repeats=2)
        result.measured_at = when
        save_result(result, tmp_path)
    assert [r["row"] for r in load_results(tmp_path)] == ["b earlier", "a later"]
