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
