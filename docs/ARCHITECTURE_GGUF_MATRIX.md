# GGUF architecture matrix (Rbitnet roadmap)

Reference for Z.ai **GLM**, OpenAI **gpt-oss**, and **DeepSeek** support. Source of truth for tensor layouts and converters is **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (`llama-arch`, `convert_hf_to_gguf.py`, model loader).

## Dispatch overview

| Vendor / line | Typical `general.architecture` (GGUF) | Rbitnet status |
|----------------|----------------------------------------|----------------|
| GLM-4.5 / 4.7 / 5 (MoE exports) | `glm4moe` (verify per release on HF) | [`glm4_moe`](../crates/bitnet-core/src/glm4_moe): **Llama-compatible tensors only** → Llama runtime; else load error ([`roadmap_unsupported`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs)). |
| OpenAI gpt-oss | `gptoss` | [`gpt_oss`](../crates/bitnet-core/src/gpt_oss): same; **MXFP4** (GGML 39) in [`dequant`](../crates/bitnet-core/src/ggml/dequant.rs). |
| DeepSeek V2 / V3 / V4 MoE | `deepseek2` (verify V4 slug when GGUF available) | [`deepseek2`](../crates/bitnet-core/src/deepseek2): Llama-shaped only today; full MoE/MLA is future work. |
| DeepSeek dense Llama-shaped | `llama`, `mistral`, … | Use existing [`Llama` loader](../crates/bitnet-core/src/loaders/llama) when tensors match. |

### DeepSeek generations vs MoE slug (verify on your GGUF)

Rollout order for full inference work: **V4 → V3 → V2**. Confirm `general.architecture` with `inspect_gguf` after each Hub download — llama.cpp may evolve slugs per release.

| Generation | Typical Hub labels | Expected MoE GGUF slug (verify) |
|------------|-------------------|--------------------------------|
| V4 | DeepSeek-V4 / successor lines | Often `deepseek2` — **confirm** when weights ship |
| V3 | DeepSeek-V3, R1 distill, … | Usually `deepseek2` in community GGUF |
| V2 | DeepSeek-V2 | Usually `deepseek2` |

If a generation only publishes MoE GGUF, there is no Llama dense shortcut — Rbitnet refuses load until a native MoE graph exists or you obtain a Llama-shaped export.

## Test checkpoints (Flash / Lite first)

For CI and local smokes, prefer **Lite**, **Flash**, or **Air** quantizations when **llama.cpp** treats them as the **same** architecture slug and graph as full-weight models. Document any deviation in this table when discovered.

| Family | Suggested first GGUF for iteration | Notes |
|--------|-------------------------------------|--------|
| GLM | GLM-4.x **Flash** / **Lite** GGUF (e.g. bartowski/zai-org repos) | Confirm `general.architecture` with `cargo run -p bitnet-core --example inspect_gguf --`. |
| gpt-oss | Smallest single-file quant (e.g. ~12 GiB class) | Same `gptoss` slug; perf validation on larger quant later. |
| DeepSeek | **V4** Lite/Flash if published with `deepseek2` | Then V3, V2 per roadmap; see [DEEPSEEK_GGUF_NOTES.md](DEEPSEEK_GGUF_NOTES.md). |

## Related code

- Dispatch: [`crates/bitnet-core/src/loaders/registry.rs`](../crates/bitnet-core/src/loaders/registry.rs)
- Unsupported roadmap layout (not Llama-tensor compatible): [`crates/bitnet-core/src/loaders/roadmap_unsupported.rs`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs)
- Builders: [`deepseek2`](../crates/bitnet-core/src/deepseek2), [`gpt_oss`](../crates/bitnet-core/src/gpt_oss), [`glm4_moe`](../crates/bitnet-core/src/glm4_moe)
