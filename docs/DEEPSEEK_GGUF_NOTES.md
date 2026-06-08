# DeepSeek GGUF notes (dense vs MoE)

Phase **1a** of the multi-family roadmap: identify which Hub checkpoints export to **Llama-shaped** dense GGUF vs **MoE-only** (`deepseek2`-style) in **llama.cpp**.

## Delivery order

Inference implementation order: **V4 → V3 → V2** (see [ARCHITECTURE_GGUF_MATRIX.md](ARCHITECTURE_GGUF_MATRIX.md)).

## Dense (Llama-compatible) path

When `convert_hf_to_gguf.py` produces a **standard Llama tensor naming** and `general.architecture` maps to a supported Llama-family key, the [`Llama` loader](../crates/bitnet-core/src/loaders/llama) applies directly (or `deepseek2` tags that still pass [`LlamaModel::from_gguf`](../crates/bitnet-core/src/llama/model.rs)).

**Validation checklist** (per generation when a dense GGUF exists):

1. `inspect_gguf` / metadata: architecture slug and hyperparameters.
2. `engine_smoke` with `RBITNET_BACKEND=cpu` (or CUDA for large models) and a matching tokenizer.
3. Record repo id + filenames in [MODEL_TESTING.md](MODEL_TESTING.md).

If only MoE exports exist for a generation, document **“dense path N/A”** — Rbitnet will **reject load** for non-Llama `deepseek2` GGUF until native MoE support ships.

## MoE (`deepseek2`)

Community GGUF for DeepSeek-V2/V3-class models typically uses **`general.architecture = deepseek2`** (confirm in your file with `inspect_gguf`).

If `LlamaModel::from_gguf` cannot parse the tensors, **the model does not load** (see [`roadmap_unsupported.rs`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs)) until a native MoE forward exists in Rbitnet.

## References

- llama.cpp DeepSeek converter and arch enums (upstream).
- [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md) for generic GGUF requirements.
