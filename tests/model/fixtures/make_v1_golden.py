"""Record a small v1 model's weights and outputs, so later model changes can prove v1 still loads and
behaves identically.

Run once, from the repository root with it on the import path (PYTHONPATH=.), with the v1 model code
(it was generated at commit 3856056):
    uv run python tests/model/fixtures/make_v1_golden.py
Running it again after the model code changes would defeat its purpose.
"""

from dataclasses import asdict
from pathlib import Path

import torch

from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

GOLDEN = Path(__file__).with_name("v1_golden.torch")  # not .pt: checkpoints (*.pt) are ignored by git

if __name__ == "__main__":
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=50, context=16, width=24, layers=2, heads=4)
    model = GPT(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(ids)
    torch.save({"model_config": asdict(config), "model": model.state_dict(), "ids": ids, "logits": logits},
               GOLDEN)
    print(f"saved {GOLDEN}")
