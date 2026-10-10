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
3. [Tokenizing real stories, and the token files](03-real-text-and-token-files.md)
4. [Embeddings and attention](04-embeddings-and-attention.md)
5. [The block and the full model](05-the-block-and-the-full-model.md)
6. [The learning loop](06-the-learning-loop.md)
7. [The real training run](07-the-real-training-run.md)
8. [Writing: sampling](08-writing-sampling.md)
9. [Measuring it: perplexity](09-measuring-it-perplexity.md)

## Part II: Make it fast

10. [Profiling one step](10-profiling-one-step.md)
11. [The benchmark harness](11-the-benchmark-harness.md)
12. [RMSNorm](12-rmsnorm.md)
13. [Rotary position embedding](13-rotary-position-embedding.md)
14. [SwiGLU](14-swiglu.md)
15. [Grouped-query attention](15-grouped-query-attention.md)
16. Ablations, and training v2
17. The KV cache
18. Fused attention
19. CUDA graphs
20. int8 and int4 quantization
21. Speculative decoding
22. Calibration
23. Judge mode
24. Serving
25. The results
