# Training and Rbitnet-compatible checkpoints

## Does Rbitnet train models?

**No.** Rbitnet is an **inference** runtime: it loads **GGUF** and runs a Llama-shaped forward pass in Rust. There is no training loop, optimizer, or dataset pipeline in this repository.

Training happens in your usual framework (**PyTorch**, **Hugging Face Transformers**, JAX, etc.). Rbitnet only consumes the **exported** artifact.

## What “compatible with Rbitnet” means

A model is compatible if, after export, it satisfies all of the following:

1. **File format:** A single **GGUF** file that Rbitnet can mmap and parse (see `[GgufArchive](../crates/bitnet-core/src/gguf/parse.rs)`).
2. **Architecture metadata:** `llama.*` keys Rbitnet reads for shapes (for example `llama.embedding_length`, `llama.block_count`, `llama.attention.head_count`, `llama.feed_forward_length`, `llama.rope.freq_base`, …). See [BITNET_SPEC.md](BITNET_SPEC.md).
3. **Tensor naming:** Llama-style names such as `token_embd.weight`, `blk.{i}.attn_norm.weight`, `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`, `blk.{i}.attn_output.weight`, `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`, `output_norm.weight`, `output.weight`. If your export uses different names, you must align them with llama.cpp conventions or extend the loader.
4. **Tokenizer:** A `**tokenizer.json`** (or compatible `**tokenizer.model**`) matching the vocabulary and special tokens of the trained model. Rbitnet does not ship tokenizers inside the GGUF; it loads them from disk (see [USAGE.md](USAGE.md)).
5. **Quantization:** Weights must use GGML types that Rbitnet can **dequantize** to `f32` for the current implementation. Exotic IQ layouts may fail until implemented; re-quantize to a supported type if needed.

## Typical workflow (train elsewhere → export → run Rbitnet)

1. **Train or fine-tune** your model in PyTorch / HF (not in this repo).
2. **Save** checkpoints in a form your **converter** accepts (often Safetensors + `config.json` on Hugging Face).
3. **Convert to GGUF** using a pipeline that targets **llama** architecture in GGUF. Rbitnet does **not** require Microsoft BitNet; you only need *a* valid GGUF file:
  - `**llama.cpp` `convert_*.py`** (or compatible exporters) for standard Llama-family checkpoints.
  - **Prebuilt GGUF** from Hugging Face or elsewhere (no converter on your machine).
  - **Microsoft BitNet** tooling only if you choose that path for BitNet-specific checkpoints ([MODEL_TESTING.md](MODEL_TESTING.md)).
4. **Copy** `tokenizer.json` (and any `tokenizer_config.json` if you rely on it elsewhere) from the Hugging Face repo next to the GGUF, or set `RBITNET_TOKENIZER`.
5. **Run** `rbitnet-server` or embed `bitnet-core`’s `Engine` as in [USAGE.md](USAGE.md).

## BitNet-specific models

For checkpoints published as **1.58-bit / BitNet** (for example [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)), Rbitnet still only needs a **GGUF** on disk. Prefer a **community `.gguf`** if available, or **llama.cpp**-compatible conversion when it applies; use **Microsoft BitNet**’s scripts only if you need their exact export path. See [MODEL_TESTING.md](MODEL_TESTING.md).

## Validating a new export

1. `cargo run -p bitnet-core --example inspect_gguf -- your.gguf`
2. Optional: `RBITNET_TEST_GGUF=your.gguf cargo test -p bitnet-core optional_gguf_from_env_smoke`
3. Run the server with `RBITNET_MODEL` and a tokenizer; send a short prompt and check the reply.

## Summary


| Phase          | Tooling                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| Training       | Your choice (PyTorch, HF, …) — **not Rbitnet**                          |
| Export to GGUF | llama.cpp, prebuilt GGUF, or optional BitNet tooling — **often Python** |
| Inference      | **Rbitnet only (Rust)** — **no Python required**                        |
