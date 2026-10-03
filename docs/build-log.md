# Build log

What was built, in order, and why.

## 1. Repository setup

- **Layout.** A flat application layout, organized by capability; see
  [ADR 0001](decisions/0001-repo-structure.md).
- **Commit checks.** Every commit goes through `scripts/commit.py`, which runs the tests and a
  guard (`scripts/guard.py`) before committing. The guard refuses local-only files, absolute
  machine paths and commit-message trailers, and checks the author identity. The same checks run
  from the git hooks in `.githooks/`, including before every push.
- **License and data.** MIT license. The training data, TinyStories, is downloaded locally and
  never committed.

## 2. Environment and machine check

- **Versions.** Python 3.14 and PyTorch 2.14, the CUDA 13.0 build from PyTorch's own package
  index (CUDA builds are not published on PyPI). uv installs both and pins every version in
  `uv.lock`, so the environment can be reproduced exactly.
- **`python -m hello_tokens check`.** Reports the versions, then runs one matrix product on the GPU
  and copies the result back, which forces the GPU to finish the work. A broken CUDA setup fails
  here, before any training starts. It also reports bf16 support, used later for mixed-precision
  training.

## 3. Downloading the corpus

- **Data.** The GPT-4-only files of TinyStories: `TinyStoriesV2-GPT4-train.txt` (2.23 GB) for
  training and `TinyStoriesV2-GPT4-valid.txt` (22.5 MB), held out for evaluation.
- **`python -m hello_tokens prepare`.** Streams each file in 1 MiB pieces, so the 2 GB file is never
  held in memory. It writes to a `.part` file and renames it only once the byte count matches the
  known size, so an interrupted download never looks complete. A file that is already complete
  is skipped.
- **Testable without the network.** The function that opens the URL is passed in, so the tests use
  an in-memory stand-in: a fresh download, a skipped re-download, and a short download that is
  rejected.
- **Resuming.** The first real download dropped at 622 MB of 2.2 GB; the size check caught it and
  left the `.part` file. The download now resumes: each attempt asks the server for the remaining
  bytes only (an HTTP `Range` request, answered with `206 Partial Content`) and appends them, with
  up to 20 attempts. Tests cover resuming from a partial file, retrying after a drop, and giving
  up without leaving a finished-looking file.

## 4. Byte-pair encoding: the core algorithm

- **Bytes first.** Text is encoded as UTF-8 bytes, so the 256 byte values are the starting
  vocabulary and any text, in any script, can be tokenized. Nothing is ever "unknown".
- **Training** repeats one step: count every pair of neighbouring tokens, take the most frequent,
  and replace each occurrence with a new token. The recorded merges, `(left, right) → new id`, are
  the tokenizer. Ties go to the pair seen first, so training is repeatable.
- **Encoding** applies merges in the order they were learned, which reproduces training's
  decisions on new text. **Decoding** expands each token back into its bytes.
- **Tests** check the classic example `aaabdaaabac` against merges worked out by hand, and that
  text round-trips exactly, including emoji and characters never seen in training.

## 5. The tokenizer on real text

- **Pre-split into chunks.** A regular expression cuts text into words (with their leading space),
  numbers, symbols and whitespace, and merges never cross a chunk boundary. Tokens therefore align
  with words, and every character falls into exactly one branch, so the chunks always rejoin into
  the original text.
- **Count distinct chunks once.** Training works on each distinct chunk with its frequency instead
  of the full text: a 2 MB sample has 2,417 stories but only 5,989 distinct chunks.
- **`<|endoftext|>` is a special token.** Stories are split on it before training, so no merge spans
  two stories, and it encodes to a single fixed id, the last in the vocabulary.
- **Vocabulary.** Ids 0–255 are bytes, then the learned merges, then the special token: 4,096 ids in
  all. Training stops early if a text runs out of pairs.
- **Saved as plain text,** one merge per line in learned order, and reloaded identically. Training
  is deterministic, so `prepare` reproduces the same file.
- **Sample size, measured.** Training the full vocabulary on 5, 10 and 20 MB took 114, 113 and
  156 s, and all three compress held-out text to 3.95 bytes per token: TinyStories' vocabulary is
  covered by 5 MB. `prepare` uses 20 MB (197 s) for broader coverage of rare words. A typical
  sentence encodes to one token per word.

## 6. Encoding the corpus

- **Token files.** `prepare` encodes the training and validation text once into
  `data/tokens/{train,valid}.bin`: flat arrays of `uint16` ids, two bytes per token, which the
  4,096-id vocabulary guarantees will fit.
- **Streaming in blocks.** Text is read about 16 MB at a time and each block is cut back to its last
  `<|endoftext|>`; the unfinished story carries into the next block, so block boundaries never change
  the tokens (tested with 37-character blocks).
- **A chunk cache.** Each distinct chunk is encoded once and looked up thereafter. 20 MB of stories
  contain only 12,682 distinct chunks, so encoding runs at 5.2 MB/s on one core.
- **Memory-mapped reads.** Token files are opened with `numpy.memmap`, so training can sample from
  them without loading them into memory.
- **Result.** Training: 562,963,642 tokens (1.13 GB, 435 s); validation: 5,683,948 tokens. 3.96 bytes
  of text per token over the whole corpus. The largest id in both files is the special token, 4,095.

## 7. Embeddings and causal self-attention

- **`ModelConfig`** holds the shape in one frozen dataclass: vocabulary 4,096, context 256, width
  384, 8 layers, 6 heads of 64.
- **Embedding.** A learned token table plus a learned position table, summed, since attention alone
  is blind to order.
- **Attention, written by hand** rather than with PyTorch's fused kernel, so it can be inspected and
  later benchmarked against it. Scores are query·key / √64; a causal mask sets every future position
  to −∞ before softmax, so future tokens receive exactly zero weight; the output is the weighted sum
  of values.
- **Multi-head.** One linear layer produces queries, keys and values for all six heads; heads run in
  parallel as an extra tensor dimension and are recombined by an output projection.
- **Tests** include a hand-worked case (equal scores average the past: 0, 0.5, 1.0, 1.5) and a
  causality check: changing token 7 leaves outputs 0–6 exactly unchanged.
- **No dropout** in v1: with 563 million tokens and less than one pass over them, overfitting is not a
  risk.

## 8. The block and the full model

- **Block.** Pre-norm GPT-2 style: `x + attention(LayerNorm(x))`, then `x + feed_forward(LayerNorm(x))`.
  The feed-forward network widens each token 4× (384 → 1,536), applies GELU, and narrows back.
- **GPT.** Embedding, 8 blocks, a final LayerNorm, and an output layer giving one logit per vocabulary
  entry at every position. The output layer is tied to the token embedding table.
- **Initialisation.** Weights drawn with standard deviation 0.02, biases zero; the two residual
  projections per block are scaled by 1/√(2·layers) so the residual stream does not grow with depth.
- **15,867,648 parameters**, matched exactly against a hand-derived formula in the tests.
- **Sanity check at initialisation.** The loss on unseen random targets is ln(4,096) ≈ 8.32, the
  uniform-guess baseline. Using the inputs as targets instead gives 7.59, because a tied model
  initially favours repeating its input; the test uses independent targets.

## 9. The training loop

- **Batches.** Random 257-token windows from the memory-mapped token file; inputs are tokens 0–255
  and targets tokens 1–256, so each window holds 256 next-token predictions.
- **Loss.** Cross-entropy over every position: the negative log-probability of the true next token.
- **Optimizer.** AdamW (β = 0.9, 0.95) with weight decay 0.1 on weight matrices only; biases and
  LayerNorm parameters are not decayed.
- **Schedule.** Linear warm-up to the peak learning rate, then cosine decay to 10% of it.
- **Step.** Forward, loss, backpropagation, gradient-norm clipping at 1.0, update.
- **Proof of learning.** On one fixed batch the loss falls from about 4.2 to 0.007 in 150 steps.
- **Faster tests.** Limiting PyTorch to 4 CPU threads for the tiny test models cut the suite from
  42 s to 7 s; thread coordination outweighed the work.

## 10. The first training run

- **Throughput, measured.** On an RTX 4060 Laptop GPU (8 GB) with bf16 autocast, batch sizes 32,
  64 and 96 all ran at 73–81k tokens/s. Batch 64 (16,384 tokens per step, 3.7 GB) was chosen.
- **Run.** 20,000 steps, 327,680,000 tokens (about 20 per parameter), peak learning rate 1e-3 with
  1,000 warm-up steps and cosine decay to 1e-4. Evaluation every 500 steps on fixed held-out batches;
  resumable checkpoints every 1,000 steps.
- **Result.** Held-out loss 8.32 → **1.286** (perplexity 3.6) in 78.5 minutes. Training loss 1.280: the
  gap of 0.006 shows no overfitting. The curve was still falling at the end.

![Loss during the first training run](images/v1-loss.png)

- **Samples.** From "Once upon a time" at temperature 0.8, the model writes coherent short stories
  with named characters and an ending, and stops by emitting `<|endoftext|>` itself.

## 11. Sampling and the `write` command

- **Sampling.** Temperature scaling (0 = greedy), top-k, and top-p (nucleus) filtering, then a draw
  from the remaining distribution with a seedable generator.
- **Generation.** One token at a time from the last-position logits, cropped to the 256-token context,
  stopping at `<|endoftext|>`. Each step recomputes the whole prefix; there is no KV cache yet.
- **`python -m hello_tokens write "<prompt>"`** with temperature 0.8 and top-p 0.95 by default.
- **Baseline speed: about 152 tokens/s (6.6 ms per token)** on the RTX 4060, flat between 50 and 200
  new tokens. This is the reference point for v2's inference work, which starts by profiling where a
  step's time goes.

## 12. Perplexity and the v1 release

- **Perplexity over the whole held-out file.** `hello_tokens/evaluation/perplexity.py` cuts the
  token file into back-to-back windows of 256 predictions, where the last token of one window is the
  first input of the next, so every token after the first is scored exactly once. Losses are summed in
  float32 and divided once at the end, so the shorter final window carries exactly its share.
- **`python -m hello_tokens eval`** scores the held-out file in float32 rather than bf16, so the
  reported number carries no rounding from mixed precision.
- **Tests.** The token count is exact; a model with all-zero weights, which gives every token equal
  probability, scores exactly the vocabulary size (64 for the test model); the batch size does not
  change the result; and the result matches scoring each window by hand.
- **Result.** 5,683,947 held-out tokens in 43 s: **loss 1.2866, perplexity 3.62.** This agrees with
  the 1.286 that training measured on a fixed sample of 50 batches, so the sample was representative.
- **v1 is complete.** The README now presents the results, sample stories, the loss chart and how each
  component works.

## 13. Profiling one generation step

- **`python -m hello_tokens profile`** times one generation step (a forward pass plus sampling) at
  contexts of 16, 64, 128 and 256 tokens. The contexts are interleaved over 200 rounds, and the
  median and interquartile range are reported. The PyTorch profiler then measures GPU kernel time and
  the number of kernels launched per step.
- **Windows power throttling.** The first measurements were three times slower than the baseline and
  noisy. Generation ran at 6 ms per token for about a second, then settled at 17-18 ms: Windows
  throttles processes it considers background work. A process can ask to be exempt
  (`SetProcessInformation` with `ProcessPowerThrottling`), which affects only that process. Every
  command now does so (`hello_tokens/generation/power.py`). Generation then holds a steady
  **6.0 ms per token (166 tokens/s)**. The earlier 152 tokens/s came from short runs, partly caught
  before the throttle started.
- **Result.** With throttling off, a step takes **6.3 ms at every context from 16 to 256 tokens**,
  in both float32 and bf16. The GPU is busy for only 0.8-2.5 ms of it (float32) or 0.5-1.2 ms (bf16).
  It sits idle 60-92% of the time, while the CPU launches about 200 kernels per step at roughly 30 µs
  each.

| context | wall (ms) | GPU busy, fp32 (ms) | GPU busy, bf16 (ms) | kernels |
|---|---|---|---|---|
| 16 | 6.3 | 0.83 | 0.51 | ~200 |
| 64 | 6.3 | 1.11 | 0.58 | ~200 |
| 128 | 6.4 | 1.58 | 0.72 | ~200 |
| 256 | 6.3 | 2.52 | 1.15 | ~200 |

- **What it predicts for v2.** Generation is bound by launch overhead, not computation. A KV cache
  and lower precision reduce work the GPU is mostly not doing, so they should change speed little at
  this size. Fewer launches (fused attention, CUDA graphs) and fewer sequential steps (speculative
  decoding) should matter most. If launch overhead were removed entirely, a step would approach its
  GPU time, roughly 6-10x faster.

## 14. The benchmark harness

- **`python -m hello_tokens bench`** measures one variant as one row of a shared table and saves it to
  `benchmarks/results/<row>.json`. The results are committed, so every speed figure in this
  repository can be traced to a measurement.
- **A fixed workload.** Greedy decoding and no stop token, so every variant writes exactly the
  requested number of tokens, at three points (prompt + new tokens): 16 + 64, 16 + 224 and 128 + 128.
  All stay within the 256-token context. The prompts are fixed random tokens: with no stop token,
  speed depends only on how many tokens there are, not which.
- **Method.** The process opts out of power throttling. A warm-up run is discarded, the GPU is
  synchronized before every clock reading, and each figure is the median of five runs, reported with
  its interquartile range. The time to the first token is measured separately; the per-token time is
  the rest of the run divided by the remaining tokens. Each row records whether every run wrote
  identical tokens, its full setup (model, dtype, cache, graphs, quantization, decoding), peak GPU
  memory, model size, held-out perplexity, and the machine.
- **The first two rows.**

| row | tok/s (16+224) | ms/token | model MB | peak MB | perplexity |
|---|---|---|---|---|---|
| v1 fp32 | 168 | 5.95 | 63.5 | 83 | 3.620 |
| v1 bf16 | 169 | 5.91 | 31.7 | 46 | 3.622 |

- **Reading.** bf16 halves the memory at no measurable cost in quality (perplexity +0.002), and does
  not change the speed, as the profile in section 13 predicted: the step is bound by kernel-launch
  overhead, not arithmetic. Repeat runs agree within about 2%.

## 15. Modern parts I: RMSNorm and rotary position embedding

- **Switches, not a rewrite.** `ModelConfig` gains `norm` (`layernorm` or `rmsnorm`) and `position`
  (`learned` or `rope`). The defaults are v1's design, and a checkpoint stores its own config, so every
  v1 checkpoint rebuilds the v1 model exactly.
- **Proof that v1 is untouched.** Before the model code changed, a small v1 model's weights and
  outputs were recorded (`tests/model/fixtures/v1_golden.torch`, made by `make_v1_golden.py`). A test
  loads those weights strictly into today's code and requires identical outputs within 1e-6. Loading
  them into a RoPE model fails, as it should. The full v1 checkpoint still writes the README samples
  token for token.
- **RMSNorm** (Zhang and Sennrich, 2019): divide by the root mean square and apply a learned scale,
  with no mean subtraction and no bias. Checked against a hand computation.
- **RoPE** (Su et al., 2021): each head's 64 numbers form 32 pairs, and the pair *i* of a token at
  position *p* is rotated by the angle *p*·*f*ᵢ, applied to queries and keys only. Since a dot product
  of rotated vectors depends only on the difference of their angles, attention scores depend on the
  distance between tokens, not their absolute positions. The test places the same query and key at
  distances 2 and 2 (equal scores) and 5 (different). The rotation tables are fixed by the shape, so
  they are not saved in checkpoints. An offset argument rotates a slice as if it sat later in a
  sequence, which the KV cache will need.
- **Parameters.** Together the two changes remove the 98,304-parameter position table and the 6,528
  LayerNorm biases (104,832 in all, pinned by a test).
