# Limitations

This page sets expectations for performance, formats, and architectures. For compatibility rules when **exporting** checkpoints, see [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md).

## Performance

- **CPU-first in production today:** multi-backend architecture exists (`cpu`, `cuda`, `rocm`, `vulkan`, `metal`) but non-CPU backends are currently MVP/parity stubs unless explicitly documented otherwise.
- **Llama weights:** **`RBITNET_LLAMA_WEIGHT_MODE`** defaults to **`auto`**. When all Llama matrices use supported GGML types, weights stay **quantized in the GGUF mmap** (row-wise GEMV), giving RAM closer to Ollama/llama.cpp on the same file. **`dense`** forces full `f32` materialization at load (legacy, very high RAM).
- **BitNet weights:** `general.architecture=bitnet` now routes to the native BitNet GGUF path for Llama-shaped Microsoft b1.58 exports. Projection weights may stay mmap-quantized when they use supported GGML types including ternary **TQ1_0** and **TQ2_0**. See [BITNET_NATIVE.md](BITNET_NATIVE.md).
- **No distributed inference:** One process loads one GGUF for real generation (stub/toy modes are separate smoke paths).
- **Not a vLLM / TRT-LLM–class server (yet):** fused GPU kernels (FlashAttention-style), **GPU-filling continuous batching**, **API-style prompt caching**, **prefill/decode disaggregation**, and **CUDA graphs** are **not** the default baseline. **Opt-in CPU serving hooks:** paged KV (`RBITNET_LLAMA_PAGED_KV` / `RBITNET_KV_POOL`), **KV Q8** (`RBITNET_KV_QUANT=q8`), and **Sarathi-style stall-free batching** (`RBITNET_CONTINUOUS_BATCHING` + `RBITNET_ITERATION_TOKEN_BUDGET` — decode-first, chunked prefill). **Fused multi-seq (#46):** scheduler hook + `dense_matvec_multi_seq` behind `RBITNET_FUSED_MULTI_SEQ=1`; **e2e concurrency gain stalled** (executors still sequential) — [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md); GPU fused = #22. See [USAGE.md](USAGE.md), `scripts/bench_kv_q8.sh`, `scripts/bench_sarathi.sh`, `scripts/bench_fused_multi_seq.sh`, [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md), [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md).

## Prefix cache vs KV prompt caching

- **`RBITNET_PREFIX_CACHE`** stores **full responses** for requests where `(prompt, max_tokens, temperature)` matches exactly. It does **not** skip prefill by reusing attention KV.
- **`RBITNET_PREFIX_KV`** (opt-in) **does** reuse tensorial KV for shared token prefixes: dense or paged snapshots, longest-common-prefix agent reuse, and a radix LRU (`RBITNET_PREFIX_KV_RADIX_MAX`). Hits are exposed as `rbitnet_core_prefix_hit` on `/metrics`. Cross-replica reuse still needs sticky co-location (`X-Rbitnet-Session` / `RBITNET_PROXY_STICKY`); see [DEPLOYMENT.md](DEPLOYMENT.md#multiple-replicas-prefix-locality) and [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md).
- Implementing **hosted-API-style** prompt caching across replicas still needs block-wise policies and L7 co-location — not the default today.

## Multi-replica deployments

- Running several **rbitnet-server** instances behind a load balancer spreads requests randomly unless you add **sticky** or **prefix-aware** routing. Prefer hashing `X-Rbitnet-Session` (or enabling `RBITNET_PROXY_STICKY` on `rbitnet-proxy`). Random spreading defeats prefix KV reuse on a single worker. Document operational expectations when you scale out.

## GGUF / GGML

- **Unknown `ggml_type` values** fail with a clear error (`UnsupportedGgmlType`) once a tensor is dequantized; see `crates/bitnet-core/src/ggml/types.rs` for layout coverage.
- **Tensor names** must follow llama.cpp-style conventions; odd exports may need renaming or loader extensions.
- **BitNet native scope:** the supported native path assumes Microsoft/llama.cpp-style BitNet GGUF naming (`token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_*`, `output.weight`) and Llama-like metadata (`llama.*` or BitNet aliases for shape fields). Non-Llama BitNet research layouts are not covered yet.

### Architecture support (spike #25 — MoE / MLA gaps)

| Family / slug | Status today | Notes |
|---------------|--------------|-------|
| Llama / Mistral-shaped (`llama`, `mistral`, …) | **Supported** | Built-in Llama loader + runtime. |
| Dense **Qwen3** (`qwen3`) | **Supported** (CPU-first) | Native dense path; optional golden note in [GOLDEN_TESTS.md](GOLDEN_TESTS.md). |
| Experimental Qwen3.5 MoE (`qwen35moe`) | **Partial** | Requires `RBITNET_BACKEND=cuda` or `hybrid`; not a general MoE solution. |
| Roadmap MoE tags (`glm4moe`, `gptoss`, `deepseek2`) | **Llama-shaped only** | Run only when [`LlamaModel::from_gguf`](../crates/bitnet-core/src/llama/model.rs) succeeds; else **startup refuse** via [`roadmap_unsupported.rs`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs). |
| Pure MoE expert routing / DeepSeek **MLA** | **Not implemented** | No silent stub executor; load fails early. Full DeepSeek MoE is out of spike scope (#25). |

When a roadmap slug’s tensors match the Llama loader, inference is real while [`ModelExecutor::family`](../crates/bitnet-core/src/model/executor.rs) still reports the roadmap slug.

**gpt-oss** MXFP4 weights: GGML type **39** dequantizes via [`tensor_to_f32`](../crates/bitnet-core/src/ggml/dequant.rs).

Dense DeepSeek checkpoints that remain **Llama-shaped** are the supported path today — see [DEEPSEEK_GGUF_NOTES.md](DEEPSEEK_GGUF_NOTES.md). Matrix detail: [ARCHITECTURE_GGUF_MATRIX.md](ARCHITECTURE_GGUF_MATRIX.md).

## Tokenizer

- The bundled generation path loads **`tokenizer.json`** (Hugging Face `tokenizers`) or **`tokenizer.model`** (SentencePiece) via env / beside the GGUF. Prefer `tokenizer.json` when both exist. See [USAGE.md](USAGE.md) and [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md).

## Research directions (not shipped)

- **Training-free speculative decoding** beyond the shipped PLD / n-gram path (for example methods in the spirit of [SPECTRA, ACL 2025](https://aclanthology.org/2025.acl-long.685/)) would be a separate research or prototype track, not a dependency of the HTTP server today.

## HTTP server

- **Inference timeout:** Long generations are cut off with HTTP 504 after `RBITNET_INFERENCE_TIMEOUT_SECS`. The server **aborts** the tokio `spawn_blocking` join handle and requests cooperative cancellation in the engine; a **thread-pool worker may still run to completion** on some runs (Tokio cannot hard-kill an in-flight closure). Do not rely on 504 as a hard process-wide stop without capacity planning.
- **Concurrency:** At most `RBITNET_MAX_CONCURRENT` generations at once; extra requests receive HTTP 503.
- **Auth:** When `RBITNET_API_KEY` is set, protect upstream with TLS and a reverse proxy for anything beyond localhost.