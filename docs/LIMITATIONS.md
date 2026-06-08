# Limitations

This page sets expectations for performance, formats, and architectures. For compatibility rules when **exporting** checkpoints, see [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md).

## Performance

- **CPU-first in production today:** multi-backend architecture exists (`cpu`, `cuda`, `rocm`, `vulkan`, `metal`) but non-CPU backends are currently MVP/parity stubs unless explicitly documented otherwise.
- **Llama weights:** **`RBITNET_LLAMA_WEIGHT_MODE`** defaults to **`auto`**. When all Llama matrices use supported GGML types, weights stay **quantized in the GGUF mmap** (row-wise GEMV), giving RAM closer to Ollama/llama.cpp on the same file. **`dense`** forces full `f32` materialization at load (legacy, very high RAM).
- **BitNet weights:** `general.architecture=bitnet` now routes to the native BitNet GGUF path for Llama-shaped Microsoft b1.58 exports. Projection weights may stay mmap-quantized when they use supported GGML types including ternary **TQ1_0** and **TQ2_0**. See [BITNET_NATIVE.md](BITNET_NATIVE.md).
- **No distributed inference:** One process loads one GGUF for real generation (stub/toy modes are separate smoke paths).
- **Not a vLLM / TRT-LLM–class server (yet):** fused GPU kernels (FlashAttention-style), **continuous batching** that fills GPU across heterogeneous requests, **API-style prompt caching** (reuse KV for common prefixes), **prefill/decode disaggregation**, and **CUDA graphs** are **not** the default baseline. **Llama-lineage** inference can use **opt-in paged KV slabs** (`RBITNET_LLAMA_PAGED_KV`, see [USAGE.md](USAGE.md)) as a step toward block-structured memory; **cross-request** pooling and full PagedAttention semantics remain roadmap ([INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md)). Smaller steps ship first: prefill **chunk** sizing via `RBITNET_PREFILL_CHUNK_TOKENS`, detailed **phase timings**, speculative **MVP** in the scheduler. See [STATUS_AND_ROADMAP.md — Advanced inference stack gaps](STATUS_AND_ROADMAP.md#advanced-inference-stack-gaps-vs-industry-serving).

## Prefix cache vs KV prompt caching

- **`RBITNET_PREFIX_CACHE`** stores **full responses** for requests where `(prompt, max_tokens, temperature)` matches exactly. It does **not** skip prefill by reusing **attention KV** for a shared token prefix (what hosted APIs call prompt caching). Implementing true prefix-KV reuse needs block-wise KV and routing/co-location policies across replicas.

## Multi-replica deployments

- Running several **rbitnet-server** instances behind a load balancer spreads requests randomly unless you add **sticky** or **prefix-aware** routing. Random spreading defeats hypothetical future **prefix KV** reuse on a single worker (same limitation discussed for production LLM gateways in vendor blogs). Document operational expectations when you scale out.

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

- The bundled generation path loads `**tokenizer.json`** via the Hugging Face `tokenizers` crate. If your workflow only produced `tokenizer.model`, convert or obtain a compatible `tokenizer.json` for Rbitnet (see [USAGE.md](USAGE.md)).

## Research directions (not shipped)

- **Training-free speculative decoding** beyond the current scheduler MVP (for example methods in the spirit of [SPECTRA, ACL 2025](https://aclanthology.org/2025.acl-long.685/)) would be a separate research or prototype track, not a dependency of the HTTP server today.

## HTTP server

- **Inference timeout:** Long generations are cut off with HTTP 504 after `RBITNET_INFERENCE_TIMEOUT_SECS`. The server **aborts** the tokio `spawn_blocking` join handle and requests cooperative cancellation in the engine; a **thread-pool worker may still run to completion** on some runs (Tokio cannot hard-kill an in-flight closure). Do not rely on 504 as a hard process-wide stop without capacity planning.
- **Concurrency:** At most `RBITNET_MAX_CONCURRENT` generations at once; extra requests receive HTTP 503.
- **Auth:** When `RBITNET_API_KEY` is set, protect upstream with TLS and a reverse proxy for anything beyond localhost.