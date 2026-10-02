# Limitations

This page sets expectations for performance, formats, and architectures. For compatibility rules when **exporting** checkpoints, see [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md).

## Performance

- **CPU-first in production today:** multi-backend architecture exists (`cpu`, `cuda`, `rocm`, `vulkan`, `metal`) but non-CPU backends are currently MVP/parity stubs unless explicitly documented otherwise.
- **Llama weights:** **`RBITNET_LLAMA_WEIGHT_MODE`** defaults to **`auto`**. When all Llama matrices use supported GGML types, weights stay **quantized in the GGUF mmap** (row-wise GEMV), giving RAM closer to Ollama/llama.cpp on the same file. **`dense`** forces full `f32` materialization at load (legacy, very high RAM).
- **BitNet weights:** `general.architecture=bitnet` now routes to the native BitNet GGUF path for Llama-shaped Microsoft b1.58 exports. Projection weights may stay mmap-quantized when they use supported GGML types including ternary **TQ1_0** and **TQ2_0**. See [BITNET_NATIVE.md](BITNET_NATIVE.md).
- **No distributed inference:** One process loads one GGUF for real generation (stub/toy modes are separate smoke paths).
- **Not a vLLM / TRT-LLM–class server (yet):** fused GPU kernels (FlashAttention-style), **GPU-filling continuous batching**, **API-style prompt caching**, **prefill/decode disaggregation**, and **CUDA graphs** are **not** the default baseline. **Opt-in CPU serving hooks:** paged KV (`RBITNET_LLAMA_PAGED_KV` / `RBITNET_KV_POOL`), **KV Q8** (`RBITNET_KV_QUANT=q8`), and **Sarathi-style stall-free batching** (`RBITNET_CONTINUOUS_BATCHING` + `RBITNET_ITERATION_TOKEN_BUDGET` — decode-first, chunked prefill). **Fused multi-seq CPU spike (#46):** `dense_matvec_multi_seq` + `generate_decode_batch` behind `RBITNET_FUSED_MULTI_SEQ=1` (scheduler decode wave can call the batch API; default executors still sequential — **not** a measured throughput ship; GPU fused = #22). See [USAGE.md](USAGE.md), `scripts/bench_kv_q8.sh`, `scripts/bench_sarathi.sh`, [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md), [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md).

## Memory / hybrid placement budgets (no FFI)

Rbitnet does **not** port akasha-os `aos-placement` (no mid-token migrate, no OS crate). It reuses the **mental model**: declare RAM/VRAM budgets, refuse load clearly when over, then retry.

### Load guardrails (RAM-style)

| Env | Effect |
|-----|--------|
| `RBITNET_MAX_WEIGHT_BYTES` | Cap on GGUF tensor payload size |
| `RBITNET_MAX_LOAD_BYTES` | Cap on weights + estimated F32 KV (via `RBITNET_BUDGET_MAX_SEQ` or GGUF context) |
| `RBITNET_MAX_VRAM_MB` | Soft VRAM-style cap on the same estimated footprint |
| `RBITNET_BUDGET_MAX_SEQ` | Tokens used for the KV estimate (default: GGUF context length) |

**Fallback:** over-budget → **load refused** with an actionable error (no hang). Fix the path/caps and `POST /v1/admin/reload` (see also LoadFailed retry). Metrics: `rbitnet_core_memory_budget_refusals_total`, `rbitnet_core_memory_budget_estimated_load_bytes`.

Example (refuse a ~1 GB GGUF on a tight host):

```bash
export RBITNET_MAX_LOAD_BYTES=$((512*1024*1024))
export RBITNET_MODEL=/path/model.gguf
# Engine::from_env / rbitnet-server fails fast with "exceeds RBITNET_MAX_LOAD_BYTES"
```

### Hybrid CPU/GPU planning (`RBITNET_BACKEND=hybrid`)

Soft **upload** budget for which Llama layers go to device memory (not a hard CUDA OOM fence):

| Env | Role |
|-----|------|
| `RBITNET_HYBRID_POLICY` | `layers` / `hotcold` / `auto` |
| `RBITNET_HYBRID_LAYERS` | Explicit layer list / ranges |
| `RBITNET_HYBRID_MAX_VRAM_MB` | Soft planning budget (default 512) |
| `RBITNET_HYBRID_MIN_ROWS` | Skip tiny matrices |
| `RBITNET_HYBRID_OUTPUT` | Optionally offload output head |

GPU backends remain MVP unless measured under #22; hybrid planning still runs so logs show the selected plan. See [ENV_REFERENCE.md](ENV_REFERENCE.md) and [USAGE.md](USAGE.md).

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

### Roadmap families (`glm4moe`, `gptoss`, `deepseek2`)

Rbitnet only runs these slugs when [`LlamaModel::from_gguf`](../crates/bitnet-core/src/llama/model.rs) succeeds — i.e. the file’s tensors match the **Llama loader** (same as how Llama/Mistral GGUF work). In that case inference is real on the in-tree stack while [`ModelExecutor::family`](../crates/bitnet-core/src/model/executor.rs) reports the roadmap slug.

If tensors are pure MoE / MLA layouts that the Llama loader cannot parse, **startup fails** with a clear error from [`roadmap_unsupported.rs`](../crates/bitnet-core/src/loaders/roadmap_unsupported.rs); there is **no** executor that loads then fails at `generate`.

**gpt-oss** MXFP4 weights: GGML type **39** dequantizes via [`tensor_to_f32`](../crates/bitnet-core/src/ggml/dequant.rs).

Dense DeepSeek checkpoints that remain **Llama-shaped** are the supported path today — see [DEEPSEEK_GGUF_NOTES.md](DEEPSEEK_GGUF_NOTES.md).

## Tokenizer

- The bundled generation path loads **`tokenizer.json`** (Hugging Face `tokenizers`) or **`tokenizer.model`** (SentencePiece) via env / beside the GGUF. Prefer `tokenizer.json` when both exist. See [USAGE.md](USAGE.md) and [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md).

## Research directions (not shipped)

- **Training-free speculative decoding** beyond the shipped PLD / n-gram path (for example methods in the spirit of [SPECTRA, ACL 2025](https://aclanthology.org/2025.acl-long.685/)) would be a separate research or prototype track, not a dependency of the HTTP server today.

## HTTP server

- **Inference timeout / cancel:** Long generations are cut off with HTTP 504 after `RBITNET_INFERENCE_TIMEOUT_SECS`. The server aborts the tokio `spawn_blocking` join handle **and** sets a cooperative cancel flag checked at **token boundaries** on the Llama decode/prefill path (and Qwen paths). Mid-generation abort should stop further tokens under that bound; Tokio still cannot hard-kill an in-flight matmul, so capacity planning still matters for worst-case wall time of one step.
- **Concurrency:** At most `RBITNET_MAX_CONCURRENT` generations at once; extra requests receive HTTP 503.
- **Auth:** When `RBITNET_API_KEY` is set, protect upstream with TLS and a reverse proxy for anything beyond localhost.