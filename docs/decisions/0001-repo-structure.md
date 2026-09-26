# 0001. Repository structure

**Status:** accepted

## Context

hello-tokens is run in place from its own checkout: data preparation, training, generation and
evaluation are commands, and a later version adds a small local API. It is not published as a
library for others to install. The layout should follow the standard for that kind of project,
make each capability easy to find, and keep generated files out of version control.

## Decision

1. **Treat the project as an application.** uv's documentation describes application projects as
   "suitable for web servers, scripts, and command-line interfaces"; they have no build system and
   are not installed ([uv: creating projects](https://docs.astral.sh/uv/concepts/projects/init/)).
2. **Use the flat layout:** one package, `hello_tokens/`, at the repository root, with no `src/`
   directory. The Python Packaging Authority notes that "the src layout requires installation of
   the project to be able to run its code, and the flat layout does not"
   ([src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)).
   The src layout protects distributed packages; nothing here is distributed.
3. **Organize the package by capability**, so the top level names what a language model does:
   `tokenizer/`, `corpus/`, `model/`, `training/`, `generation/`, `evaluation/`
   ([Screaming Architecture](https://blog.cleancoder.com/uncle-bob/2011/09/30/Screaming-Architecture.html)).
   Later capabilities (`quantization/`, `judge/`, `benchmark/`, `serving/`) are added when their
   work begins, not in advance.
4. **One entry point:** `python -m hello_tokens <command>`.
5. **Tests mirror the package** under `tests/`.
6. **Generated files are never committed:** datasets, checkpoints and run logs.
7. **One name:** the repository is `hello-tokens`; the package is `hello_tokens`, because hyphens
   are not valid in Python import names.

## Consequences

- Code runs directly from a checkout; there is no build or install step.
- Each optimization lands in the capability it changes: the KV cache in `model/`, speculative
  decoding in `generation/`.
- If the project is ever published as a library, this decision should be revisited in favor of
  the src layout.
