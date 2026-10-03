# GGUF architecture matrix (Rbitnet roadmap)

Reference for Z.ai **GLM**, OpenAI **gpt-oss**, and **DeepSeek** support. Source of truth for tensor layouts and converters is **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (`llama-arch`, `convert_hf_to_gguf.py`, model loader).

## Dispatch overview

| Vendor / line | Typical `general.architecture` (GGUF) | Rbitnet status |
|----------------|----------------------------------------|----------------|
| Dense Qwen3 | `qwen3` | Native dense CPU path ([`qwen3`](../crates/bitnet-core/src/qwen3)); **default-CI synthetic golden** + optional Hub path in [GOLDEN_TESTS.md](GOLDEN_TESTS.md). |
| Mixtral MoE | `mixtral` | Native CPU top-k MoE ([`mixtral`](../crates/bitnet-core/src/mixtral)); default-CI synthetic golden + `/v1` e2e ([GOLDEN_TESTS.md](GOLDEN_TESTS.md)). |
| Qwen3.5 dense / MoE | `qwen35`, `qwen35moe` | Native GDN + gated GQA graph on CPU/CUDA/hybrid. Real-model validation covers **Qwen3.5-2B Q8_0 dense**, text only; other sizes and MoE exports remain unvalidated. |
| GLM-4.5 / 4.7 / 5 (MoE exports) | `glm4moe` (verify per release on HF) | [`glm4_moe`](../crates/bitnet-core/src/glm4_moe): **Llama-compatible tensors only** → Llama runtime; else **clear refuse** ([`roadmap_unsupported`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs)). |
| OpenAI gpt-oss | `gpt-oss`, alias `gptoss` | Native biased attention, sinks, alternating sliding window, YaRN and MXFP4 routed experts. CPU/CUDA/hybrid; real-model validation covers **GPT-OSS-20B Q4_K_M**. Harmony analysis is hidden until a final channel appears. |
| GLM-4.7 Flash, MLA + MoE | `deepseek2` | Native compressed MLA cache, split K/V head matrices, sigmoid router with selection bias, routed + shared experts. CPU/CUDA/hybrid; real-model validation covers **GLM-4.7-Flash Q4_K_M**. Other DeepSeek exports, fused MLA tensors, and alternate query projections are not certified. |
| DeepSeek dense Llama-shaped | `llama`, `mistral`, … | Use existing [`Llama` loader](../crates/bitnet-core/src/loaders/llama.rs) when tensors match. |

### DeepSeek generations vs MoE slug (verify on your GGUF)

Rollout order for full inference work: **V4 → V3 → V2**. Confirm `general.architecture` with `inspect_gguf` after each Hub download — llama.cpp may evolve slugs per release.

| Generation | Typical Hub labels | Expected MoE GGUF slug (verify) |
|------------|-------------------|--------------------------------|
| V4 | DeepSeek-V4 / successor lines | Often `deepseek2` — **confirm** when weights ship |
| V3 | DeepSeek-V3, R1 distill, … | Usually `deepseek2` in community GGUF |
| V2 | DeepSeek-V2 | Usually `deepseek2` |

The native `deepseek2` loader checks the split MLA tensor layout eagerly. A shared architecture slug does not prove that another generation uses the supported layout. Missing or incompatible projections fail at load rather than falling through to Llama.

## Test checkpoints (Flash / Lite first)

For CI and local smokes, prefer **Lite**, **Flash**, or **Air** quantizations when **llama.cpp** treats them as the **same** architecture slug and graph as full-weight models. Document any deviation in this table when discovered.

| Family | Suggested first GGUF for iteration | Notes |
|--------|-------------------------------------|--------|
| GLM | GLM-4.x **Flash** / **Lite** GGUF (e.g. bartowski/zai-org repos) | Confirm `general.architecture` with `cargo run -p bitnet-core --example inspect_gguf --`. |
| gpt-oss | GPT-OSS-20B Q4_K_M, with MXFP4 experts | Canonical `gpt-oss` slug; CPU and CUDA measurements use the same GGUF bytes. |
| DeepSeek | **V4** Lite/Flash if published with `deepseek2` | Then V3, V2 per roadmap; see [DEEPSEEK_GGUF_NOTES.md](DEEPSEEK_GGUF_NOTES.md). |

## Related code

- Dispatch: [`crates/bitnet-core/src/loaders/registry.rs`](../crates/bitnet-core/src/loaders/registry.rs)
- Unsupported roadmap layout (not Llama-tensor compatible): [`crates/bitnet-core/src/loaders/roadmap_unsupported.rs`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs)
- Builders: [`deepseek2`](../crates/bitnet-core/src/deepseek2), [`gpt_oss`](../crates/bitnet-core/src/gpt_oss), [`glm4_moe`](../crates/bitnet-core/src/glm4_moe)
- Shared native graph and weight residency: [`native`](../crates/bitnet-core/src/native). Measured exports, results and limits: [native inference validation](benchmarks/2026-10-03-optimized/README.md).
