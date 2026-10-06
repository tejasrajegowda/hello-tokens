# hello-tokens

A language model built from scratch in PyTorch, then made fast.

hello-tokens is a decoder-only transformer written from first principles. The tokenizer, attention,
training loop, sampling and evaluation are implemented by hand; PyTorch is used only for tensors,
automatic differentiation, the optimizer and the GPU. Version 1, complete, is a 15.9-million-parameter
model trained on a laptop GPU that writes short, coherent stories. Version 2 turns to inference
engineering: making generation fast, and measuring what each optimization costs in quality.

## Status

**v1 is complete.** v2's components (modern architecture, KV cache, fused attention, CUDA graphs,
quantization, speculative decoding, calibration, judge mode, local serving) are built and tested; the
remaining step is measuring them all on the GPU. Each step is recorded in
[docs/build-log.md](docs/build-log.md).

**The book.** [docs/book/](docs/book/README.md) teaches the project step by step, for a reader who
knows Python but not machine learning: the idea, the real code, the test that proves it, and the
numbers it produced. A test keeps every code excerpt identical to the repository. Part I (building
v1) is complete; Part II (making it fast) follows the final measurements.

## Results (v1)

| | |
|---|---|
| Parameters | 15,867,648 |
| Architecture | GPT-style decoder: 8 layers, width 384, 6 heads, context 256 tokens |
| Tokenizer | Byte-level BPE, 4,096 tokens, trained in this repository |
| Training | 20,000 steps, 327.7M tokens, 78.5 minutes on an RTX 4060 Laptop GPU (8 GB), bf16 |
| Held-out loss | **1.287** (cross-entropy per token, every one of 5,683,947 held-out tokens scored) |
| Held-out perplexity | **3.62** |
| Generation speed | 166 tokens/s (6.0 ms per token), no KV cache: the baseline for v2 ([how it was measured](docs/build-log.md#13-profiling-one-generation-step)) |

Training and held-out loss finished 0.006 apart, so the model generalizes rather than memorizes, and
the held-out curve was still falling when training stopped.

![Loss during training](docs/images/v1-loss.png)

### Samples

Written by the trained model from the prompt in bold, at temperature 0.8 and top-p 0.95. Each story
ends because the model emitted its own end-of-text token. Reproduce with
`python -m hello_tokens write "<prompt>" --seed <n>`.

> **Lily found a shiny key.** She put it in the lock and opened the box. Inside, she found a pretty
> flower. The flower was red and pretty. Lily was very happy. She wanted to show her mom the flower.
> But, when Lily showed the flower to her mom, something unexpected happened. The flower started to
> talk! It said, "Thank you for finding me, Lily. I am a magic flower. I will give you one wish." Lily
> was very surprised. She thought for a moment and said, "I wish for a new friend." The magic flower
> made her wish come true. Lily and the magic flower became best friends and played together every
> day. *(seed 4)*

> **The little bird was sad because** it did not have any friends. It asked the big bear for help. The
> big bear wanted to help the little bird. They both decided to play a game. The game was to roll down
> the hill. The little bird and the big bear rolled down the hill. They laughed and had fun. The little
> bird was not sad anymore. They were best friends and played together every day. And they lived
> happily ever after. *(seed 3)*

**Limitations.** The model knows only the simple vocabulary and plots of its training stories, and at
this size it still makes logical slips, such as calling a bug harmless one sentence before it bites.

## Results (v2)

The tables below are generated from the saved measurements in `benchmarks/` by
`python -m hello_tokens report`; they are never edited by hand. Rows not yet measured on the GPU are
absent rather than estimated.

<!-- benchmarks:start -->
### Speed and quality

Greedy writing, no stop token, median of 5 runs; tok/s at (prompt + new tokens). Speed-up, first token and ms / token are at 16+224. Speed-up is against the same model in plain bf16. Perplexity and ECE are on the held-out text. Rows that only add the KV cache, CUDA graphs or speculative decoding produce their model's distribution unchanged (exact in fp32, as tested), so their quality is not measured again.

| Row | tok/s 16+64 | tok/s 16+224 | tok/s 128+128 | Speed-up | First token ms | ms / token | Size MB | Peak MB | Perplexity | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1 fp32` | 168 | 168 | 166 | 0.99× | 6.45 | 5.95 | 63.5 | 83 | 3.620 | – |
| `v1 bf16` | 168 | 169 | 169 | 1.00× | 6.27 | 5.91 | 31.7 | 46 | 3.622 | – |
| `v2 bf16` | 74 | 75 | 78 | 1.00× | 12.11 | 13.27 | 28.6 | 42 | 3.568 | – |

Measured on: NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu130, Python 3.14.3.

### Judge mode

Four-option next-sentence questions from held-out stories; the threshold is chosen on one half and reported on the other.

| Model | Questions | Accuracy (all) | Judge ECE raw → scaled | Threshold | Answers | Accuracy when answering | Device |
|---|---:|---:|---:|---:|---:|---:|---|
| `v1` | 2,000 | 71.6% | 0.205 → 0.030 | 69.7% | 50% | 88.2% | cpu |
| `v2` | 2,000 | 72.0% | 0.220 → 0.057 | 76.7% | 32% | 92.2% | cpu |
<!-- benchmarks:end -->

## How it works

Every component below is implemented in this repository and covered by tests.

**Tokenizer** (`hello_tokens/tokenizer/`). Byte-level byte-pair encoding. Text starts as UTF-8 bytes,
so any input can be encoded and nothing is ever unknown. Training repeatedly merges the most frequent
neighbouring pair into a new token until the vocabulary reaches 4,096. A regular expression first
splits text into words, numbers and symbols, so merges never cross a word boundary, and
`<|endoftext|>` is a single reserved token that separates stories. The result averages 3.96 bytes of
text per token.

**Data** (`hello_tokens/corpus/`). [TinyStories](https://arxiv.org/abs/2305.07759) is downloaded with
resumable range requests, then encoded once into flat files of 16-bit token ids (563M training tokens,
5.7M held out), which training reads through memory mapping without loading them into memory.

**Model** (`hello_tokens/model/`). Token and position embeddings, then 8 pre-norm transformer blocks,
each `x + attention(LayerNorm(x))` followed by `x + feed_forward(LayerNorm(x))`. Attention is written
out by hand: scores are query·key / √64, a causal mask stops each position from seeing later ones, and
six heads run in parallel. The feed-forward network widens each token to 1,536 with GELU and narrows it
back. The output layer shares its weights with the token embedding table.

**Training** (`hello_tokens/training/`). Random 256-token windows, cross-entropy loss on every
position, AdamW with weight decay on weight matrices only, linear warm-up then cosine decay, gradient
clipping, and bf16 mixed precision. Checkpoints are written atomically and runs resume where they
stopped. A test first proves the loop can overfit a single batch.

**Generation** (`hello_tokens/generation/`). The model scores every possible next token; temperature,
top-k and top-p (nucleus) filtering shape those scores, and one token is drawn. Generation repeats
until the model emits `<|endoftext|>`. Seeds make output reproducible.

**Evaluation** (`hello_tokens/evaluation/`). Perplexity, exp(average loss), over the whole held-out
file: back-to-back windows score every token exactly once, losses are summed in float32, and the
result is computed without mixed precision. Tests check it against a hand computation and against a
model that knows nothing, which must score exactly the vocabulary size.

## How v2 works

Each v2 change is a switch, and the v1 checkpoint still loads and runs unchanged (a golden fixture saved
from the v1 code is reproduced within 1e-6).

**Modern components** (`hello_tokens/model/`). RMSNorm in place of LayerNorm; rotary position embeddings
(RoPE) in place of the learned position table; a SwiGLU feed-forward network sized so its three matrices
hold exactly as many weights as v1's two; and grouped-query attention with 2 key/value heads for 6 query
heads, which shrinks the KV cache threefold. v2 has 14,189,952 parameters, 10.6% fewer than v1, and was
trained on the identical data, steps and schedule. Short cumulative ablations, with two v1 seeds to
measure run-to-run noise, show what each change contributes.

**KV cache** (`model/cache.py`). One preallocated key and value buffer per layer at full context size,
plus a length pointer. Each step runs the model on one new token; rolling back is moving the pointer.
Past the context limit the cache restarts from the last three quarters of the text, because a cache
cannot simply slide: deeper layers' keys were computed while the dropped tokens were still visible.

**Fused attention.** A runtime switch to PyTorch's `scaled_dot_product_attention`, with explicit masks
wherever the built-in causal mask would align queries and keys incorrectly.

**CUDA graphs** (`generation/graphs.py`). Generation at this size is limited by the cost of launching
about 200 small GPU operations per token, not by arithmetic. The one-token step is rewritten so that no
shape changes and the position lives in a tensor, recorded once as a CUDA graph, and replayed for every
token.

**Quantization** (`quantization/`). Weight-only int8 (one scale per output row) and int4 (one scale per
group of 64, two values per byte), written by hand. The shared embedding table stays in bf16.

**Speculative decoding** (`generation/speculative.py`). A 1.7M-parameter draft model guesses several
tokens; v2 checks them all in one pass and keeps those it agrees with, using the accept/reject rule
that leaves the output distribution exactly v2's own.

**Calibration** (`evaluation/calibration.py`). Expected calibration error and reliability diagrams on
every held-out next-token prediction, for each precision and quantization level.

**Judge mode** (`judge/`). Instead of writing, the model scores multiple-choice options in one batched
pass. Questions are built automatically from held-out stories, with distractors taken from later in the
same story and matched in length. The model's confidence over the options is calibrated by temperature
scaling, and below a threshold chosen on a separate half of the questions it abstains and writes its own
continuation instead.

**Serving** (`serving/`). A local FastAPI app (`play`) streams stories from v1 and v2 side by side, with
switches for the cache and fused attention, and exposes judge mode at `POST /judge`. It listens on
127.0.0.1 only.

## Getting started

Requires an NVIDIA GPU and [uv](https://docs.astral.sh/uv/). Python 3.14 and PyTorch's CUDA build
are installed by uv.

```
uv sync
uv run python -m hello_tokens check      # versions, and one computation on the GPU
uv run python -m hello_tokens prepare    # download TinyStories (2.2 GB), train the tokenizer, encode
uv run python -m hello_tokens train      # about 80 minutes on an RTX 4060 Laptop; resumable
uv run python -m hello_tokens write "Once upon a time"
uv run python -m hello_tokens eval       # perplexity on every held-out token
uv run pytest                            # the test suite, on the CPU, in a few seconds
```

`write` accepts `--temperature`, `--top-k`, `--top-p`, `--max-tokens` and `--seed`, and the speed
switches `--cache` and `--graphs`.

v2 and its measurements:

```
uv run python -m hello_tokens train --name v2 --model v2      # the modern architecture, same budget
uv run python -m hello_tokens bench --name v2 --dtype bfloat16 --cache --fused   # one benchmark row
uv run python -m hello_tokens judge --name v2                  # multiple-choice accuracy and the switch
uv run python -m hello_tokens play                             # the local playground in a browser
uv run python scripts/gpu_benchmarks.py                        # every GPU measurement, resumable
uv run python -m hello_tokens report                           # rebuild the tables from benchmarks/
```

## Roadmap

### v1: a working model

- [x] Byte-pair-encoding tokenizer, trained on TinyStories
- [x] Decoder-only transformer: embeddings, multi-head self-attention, feed-forward blocks, normalization
- [x] Training loop with loss curves
- [x] Text generation with temperature, top-k and top-p sampling
- [x] Perplexity on held-out text

### v2: modern and fast

- [x] Modern components: RoPE, RMSNorm, SwiGLU, grouped-query attention
- [x] KV cache
- [x] Fused attention
- [ ] CUDA graphs for the one-token step (exact on the CPU; GPU test pending)
- [x] int8 and int4 weight quantization
- [x] Speculative decoding with a small draft model (draft training pending)
- [x] Judge mode: score a fixed set of answers in a single forward pass, falling back to generation below a confidence threshold
- [x] Calibration (expected calibration error), measured before and after quantization
- [ ] Benchmarks: tokens per second, latency, and quality retained, for every row
- [x] A small local serving API

## Design

Structural decisions are recorded as architecture decision records in
[docs/decisions/](docs/decisions/).

## Data

[TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023), licensed CDLA-Sharing-1.0.
It is downloaded locally and never committed to this repository.

## License

[MIT](LICENSE)
