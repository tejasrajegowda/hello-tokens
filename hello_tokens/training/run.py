"""A full training run: evaluation on held-out text, a log, a chart, and resumable checkpoints."""

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from hello_tokens.corpus.encode import load_tokens
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.training.batches import get_batch
from hello_tokens.training.optimize import learning_rate_at, make_optimizer, next_token_loss, train_step


@dataclass(frozen=True)
class TrainConfig:
    name: str = "v1"
    batch_size: int = 64  # 64 windows x 256 tokens = 16,384 tokens per step
    steps: int = 20_000  # about 327M tokens: ~20 per parameter
    peak_lr: float = 1e-3
    warmup: int = 1_000
    eval_every: int = 500
    eval_batches: int = 50
    checkpoint_every: int = 1_000
    seed: int = 0


@torch.no_grad()
def evaluate(model: GPT, tokens: np.ndarray, config: TrainConfig, device: str, precision) -> float:
    """Average loss on held-out text. The same batches every time, so values are comparable."""
    model.eval()
    rng = np.random.default_rng(12345)
    losses = []
    for _ in range(config.eval_batches):
        inputs, targets = get_batch(tokens, config.batch_size, model.config.context, rng, device)
        with torch.autocast(device, dtype=precision or torch.float32, enabled=precision is not None):
            losses.append(next_token_loss(model(inputs), targets).item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model: GPT, optimizer, step: int, config: TrainConfig) -> None:
    """Save everything needed to resume. Written to .part first, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": asdict(model.config),
            "train_config": asdict(config),
        },
        partial,
    )
    partial.replace(path)


def load_model(path: Path, device: str = "cpu") -> GPT:
    """Rebuild a trained model from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = GPT(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def save_chart(log_path: Path, chart_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [float(r["train_loss"]) for r in rows], label="training text")
    ax.plot(steps, [float(r["valid_loss"]) for r in rows], label="held-out text")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title("hello-tokens: loss during training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)


def train(
    data_dir: Path,
    out_dir: Path,
    config: TrainConfig = TrainConfig(),
    model_config: ModelConfig = ModelConfig(),
    device: str = "cuda",
    precision=torch.bfloat16,
) -> GPT:
    """Train, or resume a run of the same name. Returns the trained model."""
    train_tokens = load_tokens(data_dir / "tokens" / "train.bin")
    valid_tokens = load_tokens(data_dir / "tokens" / "valid.bin")
    checkpoint_path = out_dir / "checkpoints" / f"{config.name}.pt"
    run_dir = out_dir / "runs" / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"

    torch.manual_seed(config.seed)
    model = GPT(model_config).to(device)
    optimizer = make_optimizer(model, config.peak_lr)
    step = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if ModelConfig(**checkpoint["model_config"]) != model_config:
            raise ValueError(f"checkpoint {checkpoint_path} holds a different model shape; "
                             "resume with the same --model, or choose a new --name")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        print(f"resuming {config.name} from step {step}", flush=True)
    elif log_path.exists():
        # A log with no checkpoint belongs to an abandoned run: keep it, but out of the way.
        log_path.replace(run_dir / f"log-abandoned-{int(time.time())}.csv")

    # A different seed per resume point, so a resumed run doesn't replay the same batches.
    rng = np.random.default_rng(config.seed + step)
    tokens_per_step = config.batch_size * model_config.context
    started = time.perf_counter()
    recent = []

    while step < config.steps:
        lr = learning_rate_at(step, config.peak_lr, config.warmup, config.steps)
        inputs, targets = get_batch(train_tokens, config.batch_size, model_config.context, rng, device)
        recent.append(train_step(model, optimizer, inputs, targets, lr, precision=precision))
        step += 1

        if step % config.eval_every == 0 or step == config.steps:
            valid_loss = evaluate(model, valid_tokens, config, device, precision)
            train_loss = sum(recent) / len(recent)
            recent.clear()
            new_log = not log_path.exists()
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if new_log:
                    writer.writerow(["step", "lr", "train_loss", "valid_loss", "tokens_seen", "seconds"])
                writer.writerow([step, f"{lr:.6f}", f"{train_loss:.4f}", f"{valid_loss:.4f}",
                                 step * tokens_per_step, f"{time.perf_counter() - started:.0f}"])
            save_chart(log_path, run_dir / "loss.png")
            print(f"step {step:>6}  lr {lr:.2e}  train {train_loss:.3f}  held-out {valid_loss:.3f}", flush=True)

        if step % config.checkpoint_every == 0 or step == config.steps:
            save_checkpoint(checkpoint_path, model, optimizer, step, config)
    return model
