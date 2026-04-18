# BitNet (b1.58) — interoperability spec for Rbitnet

This document describes how Rbitnet interoperates with models produced by [microsoft/BitNet](https://github.com/microsoft/BitNet) and Hugging Face → GGUF conversion pipelines.

## Common GGUF quantization labels

| BitNet / tooling name | Role |
|----------------------|------|
| `i2_s`, `tl1` | Quantization kinds used by `setup_env.py` and official conversion scripts. |
| 1.58-bit weights | Ternary {-1, 0, +1} values (often packed per block with scales). |

GGUF tensors store `ggml_type` as a raw `u32`. Rbitnet keeps that value uninterpreted so extended or future types still parse.

## Useful GGUF metadata keys

Typical keys (prefix `llama.*` when `general.architecture` declares a Llama-compatible model):

- `general.architecture` — e.g. `llama`
- `llama.context_length` — context size
- `llama.embedding_length` — hidden size
- `llama.block_count` — number of transformer blocks
- `general.alignment` — tensor data alignment (often 32)

BitNet may add layout-specific keys; the loader exposes everything via `GgufArchive::metadata`.

## Tensor naming (indicative)

Exact names follow the llama.cpp / BitNet fork conventions, for example:

- `token_embd.*`, `blk.N.attn_*`, `blk.N.ffn_*`, `output.*`

For golden validation, compare per-layer activations or final logits against **bitnet.cpp** on identical inputs.

## References

- Kernels and reports: Microsoft BitNet repo (`preset_kernels/`, `src/`).
- GGUF format: [ggml GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
