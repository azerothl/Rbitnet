# DeepSeek GGUF notes (dense vs MoE)

Phase **1a** of the multi-family roadmap: identify which Hub checkpoints export to **Llama-shaped** dense GGUF vs **MoE-only** (`deepseek2`-style) in **llama.cpp**.

## Delivery order

Inference implementation order: **V4 → V3 → V2** (see [ARCHITECTURE_GGUF_MATRIX.md](ARCHITECTURE_GGUF_MATRIX.md)).

## Dense (Llama-compatible) path

When `convert_hf_to_gguf.py` produces **standard Llama tensor naming** and `general.architecture` maps to a supported Llama-family key, the [`Llama` loader](../crates/bitnet-core/src/loaders/llama.rs) applies directly. The `deepseek2` tag now selects the native MLA/MoE graph rather than probing the Llama runtime.

**Validation checklist** (per generation when a dense GGUF exists):

1. `inspect_gguf` / metadata: architecture slug and hyperparameters.
2. `engine_smoke` with `RBITNET_BACKEND=cpu` (or CUDA for large models) and a matching tokenizer.
3. Record repo id + filenames in [MODEL_TESTING.md](MODEL_TESTING.md).

If only MoE exports exist for a generation, document **“dense path N/A”** and validate the actual MLA tensor layout; do not rename it to Llama.

## MoE (`deepseek2`)

Community GGUF for DeepSeek-V2/V3-class models typically uses **`general.architecture = deepseek2`** (confirm in your file with `inspect_gguf`).

The native graph supports low-rank `attn_q_a` / `attn_q_b`, compressed `attn_kv_a_mqa`, and separate `attn_k_b` / `attn_v_b` head projections, with routed and optional shared experts. **GLM-4.7-Flash Q4_K_M** is the real-model validation target. Its router uses sigmoid scores, a bias only for expert selection, normalized selected weights and scale 1.8. Other DeepSeek layouts, fused MLA weights, dense query variants and longer-context scaling are not validated merely because they share this slug; missing projections fail eagerly. See [measured results](benchmarks/2026-10-03-optimized/README.md).

## References

- llama.cpp DeepSeek converter and arch enums (upstream).
- [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md) for generic GGUF requirements.
