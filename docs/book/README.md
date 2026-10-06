# A language model from scratch, measured

This book builds the model in this repository step by step: a byte-level tokenizer, a GPT-style
transformer and its training loop, written by hand in PyTorch and trained on a laptop GPU. It then
makes generation fast and measures every change honestly, including the ones that made it slower.

Each chapter explains one step: the idea in plain words, the real code, what would go wrong if it were
done another way, the test that proves it, and the numbers it produced. It assumes Python, and no
machine learning; every term is explained where it first appears.

Two rules keep the book true to the code. Every code excerpt is copied from the repository, and
[a test](../../tests/docs/test_book.py) fails if an excerpt stops matching its file, a link breaks, or a
command no longer exists. Every number comes from a file committed alongside the code.

## Part I: Build it

1. [Setup and the data](01-setup-and-the-data.md)
2. [A tokenizer from scratch: byte-pair encoding](02-byte-pair-encoding.md)
3. Tokenizing real stories, and the token files
4. Embeddings and attention
5. The block and the full model
6. The learning loop
7. The real training run
8. Writing: sampling
9. Measuring it: perplexity

## Part II: Make it fast

Profiling one step, the benchmark harness, the modern parts (RMSNorm, RoPE, SwiGLU and grouped-query
attention, one chapter each), training v2, the KV cache, fused attention, CUDA graphs, quantization,
speculative decoding, calibration, judge mode, serving, and the results.

Chapters without a link are still being written.
