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
