# hello-tokens

A language model built from scratch in PyTorch, then made fast.

hello-tokens is a decoder-only transformer written from first principles. The tokenizer, attention,
training loop, sampling and evaluation are implemented by hand; PyTorch is used only for tensors,
automatic differentiation, the optimizer and the GPU. Version 1, complete, is a 15.9-million-parameter
model trained on a laptop GPU that writes short, coherent stories. Version 2 turns to inference
engineering: making generation fast, and measuring what each optimization costs in quality.

## Status

**v1 is complete.** v2 (modern components, KV cache, quantization, speculative decoding) is next.
Each step is recorded in [docs/build-log.md](docs/build-log.md).

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

`write` accepts `--temperature`, `--top-k`, `--top-p`, `--max-tokens` and `--seed`.

## Roadmap

### v1: a working model

- [x] Byte-pair-encoding tokenizer, trained on TinyStories
- [x] Decoder-only transformer: embeddings, multi-head self-attention, feed-forward blocks, normalization
- [x] Training loop with loss curves
- [x] Text generation with temperature, top-k and top-p sampling
- [x] Perplexity on held-out text

### v2: modern and fast

- [ ] Modern components: RoPE, RMSNorm, SwiGLU, grouped-query attention
- [ ] KV cache
- [ ] int8 and int4 weight quantization
- [ ] Speculative decoding with a small draft model
- [ ] Judge mode: score a fixed set of answers in a single forward pass, falling back to generation below a confidence threshold
- [ ] Calibration (expected calibration error), measured before and after quantization
- [ ] Benchmarks: tokens per second, latency, and quality retained
- [ ] A small local serving API

## Design

Structural decisions are recorded as architecture decision records in
[docs/decisions/](docs/decisions/).

## Data

[TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023), licensed CDLA-Sharing-1.0.
It is downloaded locally and never committed to this repository.

## License

[MIT](LICENSE)
