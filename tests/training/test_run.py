import csv

import numpy as np
import pytest
import torch

from hello_tokens.model.config import ModelConfig
from hello_tokens.training.run import TrainConfig, load_model, train

TINY = ModelConfig(vocab_size=64, context=16, width=32, layers=2, heads=4)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "tokens").mkdir()
    for split in ("train", "valid"):
        rng.integers(0, TINY.vocab_size, 5000).astype(np.uint16).tofile(tmp_path / "tokens" / f"{split}.bin")
    return tmp_path


def run(data_dir, out_dir, steps):
    config = TrainConfig(name="test", batch_size=4, steps=steps, warmup=5, eval_every=10,
                         eval_batches=2, checkpoint_every=10)
    return train(data_dir, out_dir, config, TINY, device="cpu", precision=None)


def read_log(out_dir):
    with open(out_dir / "runs" / "test" / "log.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_a_run_writes_a_log_a_chart_and_a_checkpoint(data_dir, tmp_path):
    out = tmp_path / "out"
    run(data_dir, out, steps=20)
    assert [r["step"] for r in read_log(out)] == ["10", "20"]
    assert (out / "runs" / "test" / "loss.png").exists()
    assert (out / "checkpoints" / "test.pt").exists()


def test_an_interrupted_run_resumes_where_it_stopped(data_dir, tmp_path, capsys):
    out = tmp_path / "out"
    run(data_dir, out, steps=20)  # stops at step 20, as if interrupted
    run(data_dir, out, steps=30)  # same run, longer: must continue from 20, not restart
    assert "resuming test from step 20" in capsys.readouterr().out
    assert [r["step"] for r in read_log(out)] == ["10", "20", "30"]


def test_a_saved_model_reloads_identically(data_dir, tmp_path):
    out = tmp_path / "out"
    model = run(data_dir, out, steps=10)
    loaded = load_model(out / "checkpoints" / "test.pt")
    ids = torch.randint(0, TINY.vocab_size, (1, 8))
    model.eval()
    assert torch.allclose(model(ids), loaded(ids))
