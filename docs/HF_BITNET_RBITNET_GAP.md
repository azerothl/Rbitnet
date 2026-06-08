# Hugging Face BitNet vs Rbitnet — gaps and install expectations

This document explains why many BitNet-related Hugging Face repositories do not map 1:1 to what [Rbitnet](../README.md) can run today, and what the CLI can automate versus what still requires human steps (tokens, gated models, or a different runtime).

## What Rbitnet needs at runtime

| Requirement | Details |
|-------------|---------|
| **Weights** | A **GGUF** file on disk (`RBITNET_MODEL`). Rbitnet does **not** load Safetensors checkpoints. |
| **Architecture** | A **Llama-shaped** graph: [`LlamaModel::from_gguf`](../crates/bitnet-core/src/llama/model.rs) and metadata keys such as `llama.*` (see [BITNET_SPEC.md](BITNET_SPEC.md)). |
| **Tokenizer** | A **`tokenizer.json`** or **`tokenizer.model`** next to the GGUF, or `RBITNET_TOKENIZER` pointing at such a file (see [USAGE.md](USAGE.md)). Raw SentencePiece protobuf `.model` files may fail to load; prefer `tokenizer.json` exported from Hugging Face. |
| **Quantization** | GGML types supported for dequantization in [`types.rs` / `dequant.rs`](../crates/bitnet-core/src/ggml/types.rs). Unknown types fail with `UnsupportedGgmlType`. Implemented paths include common `Q4_*` / `Q8_*` / `Q4_K` / `Q6_K` / `BF16` / `TQ1_0` / `TQ2_0`; several K-quants and IQ variants still return `UnsupportedGgmlType` until wired in `dequant.rs` (search may label some GGUFs `experimental_gguf` from the filename alone). |

## Typical Hugging Face BitNet layouts

| Pattern on Hub | Example | Gap for Rbitnet |
|----------------|---------|-----------------|
| **Safetensors main repo + separate GGUF repo** | [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (weights) vs `microsoft/bitnet-b1.58-2B-4T-gguf` | Search that only lists `.gguf` in the **same** tree will miss the main card unless the user installs from the **GGUF** repo. The CLI `models install` bundle resolves this pair. |
| **Tokenizer in another repo (often gated)** | [HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens) uses `AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")` | Rbitnet cannot download Meta weights without **your** Hugging Face access (gated models + `HF_TOKEN`). No tool can bypass Hub policy. |
| **Non-Llama architecture (e.g. Phi)** | [tzervas/phi-4-bitnet-1.58b](https://huggingface.co/tzervas/phi-4-bitnet-1.58b) | Rbitnet has **no Phi forward**; even with a GGUF file, inference is not supported until a Phi backend exists. Search labels these as **unsupported architecture (heuristic)**. |
| **GGUF + experimental quant names** | Community GGUFs with BitNet-specific quant labels in filenames | May use GGML types not yet implemented; may need re-export or loader work. |

## Readiness labels (`models search`)

The CLI assigns a **readiness** label to each search hit (best-effort, no full GGUF parse in search):

| Label | Meaning |
|-------|---------|
| **ready** | At least one `.gguf` and `tokenizer.json` or `tokenizer.model` appears in the Hub **siblings** list for that repo. |
| **needs_tokenizer** | `.gguf` present but no tokenizer files in siblings. |
| **needs_external_tokenizer** | Only `tokenizer_config.json` (Transformers metadata) or other hints that the tokenizer may live elsewhere / need `AutoTokenizer` setup. |
| **unsupported_arch_likely** | Repo id matches a **heuristic** for non-Llama families (e.g. Phi). |
| **experimental_gguf** | GGUF filename suggests BitNet-specific quant strings (e.g. `I2_S`); runtime support may still fail until types are implemented. |

These labels are **not** a formal compatibility guarantee; only the curated catalog in `models list` is project-tested.

## One-click install (`models install`)

For a small set of **known bundles** (e.g. official Microsoft BitNet GGUF repo + tokenizer files from the paired Safetensors repo), `rbitnet models install <bundle-id>` downloads files into one directory and writes **`rbitnet.manifest.json`** with suggested `RBITNET_MODEL` and `RBITNET_TOKENIZER` paths. Run `models install --list` to see bundle ids.

Bundles cannot fix gated third-party tokenizers without a valid **`HF_TOKEN`** and Hub access to the gated repo.

## See also

- [USAGE.md](USAGE.md) — environment variables and CLI commands.
- [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md) — export and tokenizer expectations.
- [BITNET_SPEC.md](BITNET_SPEC.md) — GGUF metadata and tensor naming.
