# hello-tokens

A language model built from scratch in PyTorch, then made fast.

hello-tokens is a decoder-only transformer written from first principles. The tokenizer, attention,
training loop and sampling are implemented by hand; PyTorch is used only for tensors, automatic
differentiation and the GPU. Once the model writes coherent text, the project turns to inference
engineering: making generation fast, and measuring what each optimization costs in quality.

## Status

Early development. The repository layout and tooling are in place, and the model is being built
one component at a time. Progress is recorded in [docs/build-log.md](docs/build-log.md).

## Roadmap

### v1: a working model

- [ ] Byte-pair-encoding tokenizer, trained on TinyStories
- [ ] Decoder-only transformer: embeddings, multi-head self-attention, feed-forward blocks, normalization
- [ ] Training loop with loss curves
- [ ] Text generation with temperature, top-k and top-p sampling
- [ ] Perplexity on held-out text

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
