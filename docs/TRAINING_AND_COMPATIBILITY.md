# Training and Rbitnet-compatible checkpoints

## Does Rbitnet train models?

**No.** Rbitnet is an **inference** runtime: it loads **GGUF** and runs a Llama-shaped forward pass in Rust. There is no training loop, optimizer, or dataset pipeline in this repository.

Training happens in your usual framework (**PyTorch**, **Hugging Face Transformers**, JAX, etc.). Rbitnet only consumes the **exported** artifact.

## Recommended fine-tuning path (before GGUF)

For Llama-family models consumed later as **GGUF**, the usual stack is:

1. **LoRA or QLoRA + SFT** (supervised fine-tuning) on a JSONL or Hugging Face dataset — lighter than full fine-tune and sufficient for many tasks.
2. **Merge adapters** (optional) into a full Safetensors checkpoint if your exporter expects a single merged model.
3. **Convert to GGUF** with a **llama.cpp**-compatible pipeline (see below).

Full fine-tuning is possible but heavier on GPU memory and time; start with LoRA/SFT unless you have a clear need for full weights updates.

This repository ships an optional **Python recipe** under [`training/`](../training/README.md) that you can run locally or adapt — it is **not** executed by the Rust runtime.

## Working with [ml-intern](https://github.com/huggingface/ml-intern)

[ml-intern](https://github.com/huggingface/ml-intern) is a Hugging Face open-source agent oriented toward the **research and post-training loop**: literature search (arXiv, [HF Papers](https://huggingface.co/papers)), citation graphs, dataset inspection on the [Hub](https://huggingface.co/datasets), training in GPU sandboxes, **[HF Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)** when you have no local GPU, and iteration from evals. It is distributed as a [CLI](https://github.com/huggingface/ml-intern/tree/main) and a [Space (web / mobile)](https://huggingface.co/spaces/smolagents/ml-intern).

**Division of labor:**

| Stage | Tool | Typical output |
|-------|------|----------------|
| Research → data → train → eval | ml-intern (and/or your own notebooks) | Safetensors checkpoints, adapters, scripts |
| Convert checkpoint → **GGUF** | **llama.cpp** `convert_*.py` / quantize (versions vary by upstream) | `.gguf` file |
| Local inference | **Rbitnet** | `RBITNET_MODEL`, OpenAI-compatible HTTP API |

The **LiteLLM**-based model that drives the ml-intern **agent** (paper reading, code generation) is **independent** from the base model you fine-tune and from the GGUF you serve with Rbitnet.

Whether training ran on **local GPU**, **[HF Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)**, or **GPU Spaces** does not change Rbitnet: you still need a **single GGUF path** (and tokenizer files) on disk for the server.

## What “compatible with Rbitnet” means

A model is compatible if, after export, it satisfies all of the following:

1. **File format:** A single **GGUF** file that Rbitnet can mmap and parse (see [`GgufArchive`](../crates/bitnet-core/src/gguf/parse.rs)).
2. **Architecture metadata:** `llama.*` keys Rbitnet reads for shapes (for example `llama.embedding_length`, `llama.block_count`, `llama.attention.head_count`, `llama.feed_forward_length`, `llama.rope.freq_base`, …). See [BITNET_SPEC.md](BITNET_SPEC.md).
3. **Tensor naming:** Llama-style names such as `token_embd.weight`, `blk.{i}.attn_norm.weight`, `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`, `blk.{i}.attn_output.weight`, `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`, `output_norm.weight`, `output.weight`. If your export uses different names, you must align them with llama.cpp conventions or extend the loader.
4. **Tokenizer:** `tokenizer.json` (or compatible `tokenizer.model`) matching the vocabulary and special tokens of the trained model. Rbitnet does not ship tokenizers inside the GGUF; it loads them from disk (see [USAGE.md](USAGE.md)).
5. **Quantization:** Weights must use GGML types that Rbitnet can **dequantize** to `f32` for the current implementation. Exotic IQ layouts may fail until implemented; re-quantize to a supported type if needed.

Do **not** assume every checkpoint (including BitNet-specific or exotic quantizations) converts cleanly to a Llama-shaped GGUF without checking the upstream converter and running `inspect_gguf`.

## Typical workflow (train elsewhere → export → run Rbitnet)

1. **Train or fine-tune** (PyTorch / Transformers / TRL / ml-intern-assisted workflows — not in `bitnet-core`).
2. **Save** checkpoints in a form your **converter** accepts (often Safetensors + `config.json`).
3. **Convert to GGUF** using a pipeline that targets **llama** architecture in GGUF. Rbitnet does **not** require Microsoft BitNet; you only need *a* valid GGUF file:
   - **llama.cpp** `convert_hf_to_gguf.py` (name and location change with llama.cpp releases — follow the docs for your checkout).
   - **Prebuilt GGUF** from Hugging Face or elsewhere (no converter on your machine).
   - **Microsoft BitNet** tooling only if you choose that path for BitNet-specific checkpoints ([MODEL_TESTING.md](MODEL_TESTING.md)).
4. **Copy** `tokenizer.json` (and `tokenizer.model` if used) beside the GGUF, or set `RBITNET_TOKENIZER`.
5. **Run** `rbitnet-server` / `rbitnet serve`, or embed `bitnet-core`’s `Engine` as in [USAGE.md](USAGE.md).

Optional: use [`rbitnet train`](USAGE.md#fine-tuning-helper-cli) from a checkout of this repo to launch the bundled **Python** LoRA/SFT recipe; use [`rbitnet export-gguf`](USAGE.md#fine-tuning-helper-cli) for a short checklist toward GGUF conversion.

## BitNet-specific models

For checkpoints published as **1.58-bit / BitNet** (for example [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)), Rbitnet still only needs a **GGUF** on disk. Prefer a **community `.gguf`** if available, or **llama.cpp**-compatible conversion when it applies; use **Microsoft BitNet**’s scripts only if you need their exact export path. See [MODEL_TESTING.md](MODEL_TESTING.md). That workflow is **distinct** from standard Llama LoRA fine-tunes exported via llama.cpp.

## Validating a new export

1. `cargo run -p bitnet-core --example inspect_gguf -- your.gguf`
2. Optional: `RBITNET_TEST_GGUF=your.gguf cargo test -p bitnet-core optional_gguf_from_env_smoke`
3. Run the server with `RBITNET_MODEL` and a tokenizer; send a short prompt and check the reply.

## Summary


| Phase          | Tooling                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| Training       | PyTorch / HF / TRL / ml-intern / [`training/` recipes](../training/README.md) — **not** `bitnet-core` |
| Export to GGUF | llama.cpp, prebuilt GGUF, or optional BitNet tooling — **often Python** |
| Inference      | **Rbitnet only (Rust)** — **no Python required**                        |
