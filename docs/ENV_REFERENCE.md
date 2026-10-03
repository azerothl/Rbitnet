# Environment variable reference

Single index for **`rbitnet-server`** / **`rbitnet serve`** and **`bitnet-core`** tuning. Defaults apply when a variable is unset unless noted.

## HTTP server (`bitnet-server`)

| Variable | Type | Default | Notes |
|----------|------|---------|--------|
| `RBITNET_BIND` | `host:port` | `127.0.0.1:8080` | Listening address. CLI `--bind` sets this if env unset. |
| `RBITNET_API_KEY` | string | (none) | If set, `/v1/*` requires `Authorization: Bearer` or `X-API-Key`. Health/metrics paths stay open. CLI `--api-key` sets this if env unset. |
| `RBITNET_MAX_BODY_BYTES` | usize | `1048576` | JSON body cap; HTTP **413** if exceeded. Invalid → startup error. |
| `RBITNET_MAX_PROMPT_CHARS` | usize | `256000` | UTF-8 character cap on constructed prompt; HTTP **400**. Must be ≥ 1. |
| `RBITNET_MAX_PROMPT_TOKENS` | u32 | (none) | Optional: cap on tokenizer-encoded prompt length (checked before inference); HTTP **400**. Must be ≥ 1 when set. |
| `RBITNET_MAX_TOKENS_CAP` | u32 | `8192` | Hard ceiling on client `max_tokens`; HTTP **400** above cap. Must be ≥ 1. |
| `RBITNET_MAX_CONCURRENT` | usize | `4` | Concurrent generations; HTTP **503** when saturated. Coerced to ≥ 1. |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | u64 | `600` | Wall-clock limit per completion; HTTP **504**. Coerced to ≥ 1 second. |
| `RBITNET_REQUIRE_MODEL_MATCH` | flag | off | If `1`/`true`/`yes`, OpenAI `model` must match server/registry id. |
| `RBITNET_CORS_ANY` | flag | off | If `1`, allow any origin on CORS (dev-style). |
| `RBITNET_MODEL_REGISTRY` | path | (none) | JSON registry for multi-model; see [USAGE.md](USAGE.md). |
| `RBITNET_ACTIVE_MODEL_ID` | string | (none) | Registry key when JSON has no `default`. |
| `RBITNET_ADMIN_TOKEN` | string | (none) | Enables `POST /v1/admin/unload` and `POST /v1/admin/reload` with matching header. |
| `RBITNET_IDLE_UNLOAD_SECS` | u64 | (none) | After idle, recycle child runners (**proxy** — recommended multi-model path) or swap engine for stub (single-process server). See [RUNNER_PROXY_SPEC.md](RUNNER_PROXY_SPEC.md). Metric: `rbitnet_model_unloads_total`. |
| `RBITNET_CHAT_BASE_URL` | URL | `http://127.0.0.1:8080/v1` | Default base URL for `rbitnet chat`; `/v1` is appended when omitted. |

## Runner proxy (`rbitnet-proxy`)

| Variable | Type | Default | Notes |
|----------|------|---------|--------|
| `RBITNET_PROXY_BIND` | `host:port` | `RBITNET_BIND` or `127.0.0.1:8080` | Parent proxy listen address. |
| `RBITNET_MODEL_REGISTRY` | path | required | JSON registry; each model id becomes one supervised worker in local mode. |
| `RBITNET_RUNNER_BIN` | path/name | sibling `rbitnet-runner`, then `PATH` | Worker executable spawned per model id. |
| `RBITNET_RUNNER_READY_TIMEOUT_SECS` | u64 | `60` | Time for child `GET /ready` to become 2xx. |
| `RBITNET_PROXY_REQUEST_TIMEOUT_SECS` | u64 | `600` | Upstream native child request timeout. |
| `RBITNET_PROXY_STICKY` | flag | off | If `1`/`true`/`yes`, bind sticky session ids to model ids and echo `X-Rbitnet-Session` / `X-Rbitnet-Sticky-Bucket`. Session from header, `rbitnet_session` cookie, or body `session`/`user`. |
| `RBITNET_PROXY_REPLICAS` | u32 | `1` | Planned replica count for sticky hash-bucket plumbing. Does not spawn extra children today (still one runner per model id); use with an external LB hashing `X-Rbitnet-Session`. |
| `RBITNET_INFERENCE_BACKEND` | string | `local` | Normal builds accept local native workers only. External delegation values require the dev-only Cargo feature `experimental-external-backends`. |
| `RBITNET_VLLM_BASE_URL` | URL | (none) | Ignored by default builds; only used when an experimental external backend is compiled in. |

## Core / model loading (`bitnet-core`)

| Variable | Default | Notes |
|----------|---------|--------|
| `RBITNET_MODEL` | (none) | Path to one `.gguf` file. |
| `RBITNET_TOKENIZER` | (auto beside GGUF) | `tokenizer.json` or `tokenizer.model`. |
| `RBITNET_CHAT_FORMAT` | auto template / `raw` fallback | Explicit `raw`, `llama3`, or `chatml`; when unset, render the discovered GGUF/tokenizer Jinja chat template. Thinking is disabled where the template supports it; GPT-OSS reasoning effort is low. |
| `RBITNET_CHAT_TEMPLATE` | (none) | Custom Jinja or simple placeholder template (`{messages}`, `{prompt}`, `{system}`, `{user}`, `{assistant}`); overrides format and discovered template. Invalid Jinja is an HTTP 400. |
| `RBITNET_STUB` | off | Synthetic completions; no weights. |
| `RBITNET_TOY` | off | Tiny in-process toy LM; no GGUF. |
| `RBITNET_BACKEND` | `auto` | Default **`auto`**: detect CUDA→ROCm→Metal→Vulkan→CPU. Explicit: `cpu`, `cuda`, `hybrid`, `rocm`, `vulkan` (aliases: `intel`, `oneapi`, `level-zero`), `metal`. See [USAGE.md](USAGE.md), [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md). |
| `RBITNET_CUDA_QUANT_LIB` | unset | Absolute path to `rbitnet_cuda_quant64.dll` / `librbitnet_cuda_quant.so` for Gate E device kernels (`native/cuda_quant`). |
| `RBITNET_CUDA_ATTENTION` | on for native CUDA/hybrid dense KV | `0`/`false`/`no` forces CPU attention. Fused all-head GQA/MQA, sliding window and sinks; resident F32 KV, maximum CUDA capacity 8192. Missing symbols/unsupported cache fall back to CPU. |
| `RBITNET_MAX_SEQ` | `8192` for native Qwen/GPT/MLA | Positive context capacity, capped by the model's metadata; CUDA attention above 8192 falls back to CPU. Does not change the existing Llama configuration. |
| `RBITNET_ARCHITECTURE` | (from GGUF) | Override `general.architecture`. |
| `RBITNET_MODEL_FAMILY` | `auto` | Architecture hint when no GGUF (stub/toy). |
| `RBITNET_PREFIX_CACHE` | off | Cache full duplicate completions (not KV). |
| `RBITNET_PREFIX_CACHE_MAX_ENTRIES` | `64` | Prefix response cache size. |
| `RBITNET_PREFIX_KV` | off | Dense/paged KV snapshot reuse for shared prompt prefixes (prefill skip). Prefer on for ≥2 concurrent interactive sessions (`rbitnet tune interactive`). Enables radix + LCP agent reuse. |
| `RBITNET_PREFIX_KV_MAX_ENTRIES` | `32` | LRU size for execution-time prefix KV snapshots. |
| `RBITNET_PREFIX_KV_RADIX_MAX` | `256` | Max radix leaf entries (LRU eviction). |
| `RBITNET_PREFIX_KV_MIN_TOKENS` | `8` | Minimum LCP tokens before previous-request prefix restore counts as a hit. |
| `RBITNET_SESSIONS` | off | Enable session store (also implied by `RBITNET_CONTINUOUS_BATCHING`). |
| `RBITNET_MODEL_SHA256` | (none) | Expected SHA-256 hex of the GGUF; verified after install/download when set. |
| `RBITNET_TRUSTED_MODELS_ONLY` | off | If `1`, refuse Hub downloads outside the curated catalog and require a known SHA-256 on curated install. |
| `RBITNET_CUDA_GRAPH` | off | Legacy scheduling diagnostics (`llama/cuda_graph.rs`); does not launch device graphs or increment native replay metrics. |
| `RBITNET_CUDA_RESIDENT` | auto | Fully resident dense Llama graph when every projection is on CUDA, KV is dense and prefix reuse is off; `0` restores per-projection execution. Unsupported models/DLLs retain their existing path. |
| `RBITNET_CUDA_RESIDENT_GRAPH` | on | Capture/replay the resident Llama token graph; `0` executes the same device kernels eagerly for ablation. |
| `RBITNET_CUDA_MOE` | auto | Resident routed FFN graph for GPT-OSS/MLA layers with all three expert matrices on CUDA; `0` restores the individual expert path. |
| `RBITNET_CUDA_QWEN_RECURRENT` | auto | Complete dense Qwen3.5 recurrent blocks on CUDA when their eight projections are resident/supported. Retains convolution and F32 GDN state; `0` uses the CPU recurrent path. Qwen MoE and unsupported head layouts retain their existing path. |
| `RBITNET_CUDA_QWEN_RECURRENT_GRAPH` | on | Capture/replay each supported recurrent block; `0` runs the same block kernels eagerly. |
| `RBITNET_CUDA_HEAD` | off | Experimental resident output RMSNorm/projection and GPU greedy reduction for Qwen/GPT-OSS/MLA: `1` enables it when supported/resident. Current Windows ablations show no speed gain, so it stays opt-in. Sampling, penalties and JSON keep full F32 logits and the common sampler. |
| `RBITNET_CPU_SIMD_QUANT` | on | Decode packed weights into AVX2/FMA registers with F16C; `0` restores the older block decoder. Activations remain F32. |
| `RBITNET_CPU_AVX512` | auto | Use 16-lane kernels when AVX512F/BW are available; `0` uses the 8-lane AVX2 path. |
| `RAYON_NUM_THREADS` | available CPUs | Also controls the dedicated quantized matvec pool; set before loading a runtime. |
| `RBITNET_MTP_K` | `1` | Multi-token burst width when `>1` (Atlas-style MTP scheduler hook). |
| `RBITNET_KV_POOL` | off | Enable process-wide shared physical KV pages end-to-end (`kv_pool.rs`). Implies paged slabs for Llama runtime; reclaim on `clear()` / sequence close. Prefer with `RBITNET_LLAMA_PAGED_KV=1`. Bench: `scripts/bench_paged_kv.sh`. |
| `RBITNET_KV_POOL_MAX_SEQS` | `8` | Max concurrent sequences / per-seq logical page budget divisor in `PagedKvPool`. |
| `RBITNET_KV_SIDECAR_URL` | (none) | Optional external KV sidecar base URL; see [KV_SIDECAR_SPEC.md](KV_SIDECAR_SPEC.md). |
| `RBITNET_KV_SIDECAR_TIMEOUT_SECS` | `30` | Sidecar HTTP timeout. |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | (server) | Same name used by server for HTTP timeout; core cancellation hooks align with server layer. |
| `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_LOAD_BYTES`, `RBITNET_MAX_VRAM_MB`, `RBITNET_BUDGET_MAX_SEQ` | (none) | Load guardrails; see [LIMITATIONS.md](LIMITATIONS.md). |
| `RBITNET_CONTINUOUS_BATCHING` | off | Enable Sarathi-style stall-free batching (`run_batch_waves`); decode-first + chunked prefill. |
| `RBITNET_FUSED_MULTI_SEQ` | off | **#46 stalled:** decode waves call `generate_decode_batch` (CPU). Default executors still sequential — **no e2e throughput claim**. See [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md). GPU fused = #22. |
| `RBITNET_ITERATION_TOKEN_BUDGET` | `2×chunk` | Token budget per stall-free iteration (see [USAGE.md](USAGE.md)). |
| `RBITNET_PREFILL_CHUNK_TOKENS` | `128` | Prefill chunk size for runtime loops and scheduler admission. |
| `RBITNET_SPECULATIVE`, `RBITNET_SPEC_DRAFT_RATIO_*` | varies | Speculative draft path; see [USAGE.md](USAGE.md). |
| `RBITNET_DRAFT_PATH` | `ngram` when speculative on else `target` | Speculative draft source: `ngram`/`pld` (prompt-lookup, no weights), `toy`, or `target`. |
| `RBITNET_DRAFT_MODEL` | (none) | Path reserved for a small GGUF draft model. Current builds recognize the path and fall back to the lightweight n-gram draft until separate draft-model verification is wired. |
| `RBITNET_STRUCTURED_OUTPUT` | `off` | Optional sampler mask: `json` / `tool` enables the ASCII/byte-token JSON FSM mask before sampling. Prefer request `response_format` (`json_object` / `json_schema`) when available. |
| `RBITNET_LLAMA_WEIGHT_MODE` | `auto` | Llama-lineage matrices: **`dense`** (legacy: full `tensor_to_f32` at load, high RAM), **`mmap_quant`** (quantized weights stay in the GGUF mmap; row-wise GEMV), **`auto`** (mmap when every weight tensor uses a supported GGML type for mmap GEMV; otherwise dense). |
| `RBITNET_QUANT_KERNEL` | `auto` | Quantized matvec backend: `auto`/CPU parallel, `scalar`, or `cuda` to use optional native symbols for F32, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K and MXFP4 with CPU fallback. Resident CUDA matrices use their device kernel directly. |
| `RBITNET_QUANT_PAR_MIN_ROWS` | `128` | Minimum output rows before the CPU quantized matvec path splits work across a reusable thread pool (`rayon`). |
| `RBITNET_BLAS` | off | If `1`/`true`/`yes`, loads **OpenBLAS** dynamically (`cblas_sgemv`) for Llama **CPU/hybrid** attention score GEMV and **dense** `f32` matvec helpers when the library is found on `PATH` / `LD_LIBRARY_PATH` / default search. See [USAGE.md](USAGE.md). |
| `RBITNET_SLIM_ATTENTION` | off | If `1`/`true`/`yes`/`on`, Llama CPU/hybrid decode uses SlimAttention 1D tiled online-softmax (`llama::slim_attention`) instead of contiguous scores→softmax→V. Default off. |
| `RBITNET_SLIM_ATTENTION_TILE` | `16` | Tile width (tokens) along the KV sequence when SlimAttention is on. Must be ≥ 1. |
| `RBITNET_LLAMA_MATMUL` | (unset) | Reserved hook: `ggml` requests a future native ggml bridge (still Rust kernels today). Meaningful only when `bitnet-core` is built with `--features experimental-ggml-kernels`; see [GOLDEN_TESTS.md](GOLDEN_TESTS.md). |
| `RBITNET_HYBRID_POLICY` | `layers` | Hybrid layer-selection policy: `layers` keeps explicit/early-layer behavior, `hotcold` selects deeper hot decode layers first, `auto` uses explicit `RBITNET_HYBRID_LAYERS` when present then hot/cold selection. |
| `RBITNET_HYBRID_LAYERS` | `auto` | With `RBITNET_BACKEND=hybrid`, comma/range list of Llama layers to offload (`0`, `0-3`, `0,2,4`). If unset, the loader selects early layers within `RBITNET_HYBRID_MAX_VRAM_MB`. |
| `RBITNET_HYBRID_MAX_VRAM_MB` | hybrid `512`; CUDA Llama `4096`, native Qwen/GPT/MLA `12288` | Soft weight placement budget. Quantized Llama auto/native Qwen/GPT/MLA count actual packed bytes; native models prefer output + attention/shared projections before routed experts. Qwen's extra small head projections count toward the same budget. KV, recurrent states, allocator and driver overhead are additional. |
| `RBITNET_HYBRID_MIN_ROWS` | `512` | Minimum matrix output rows for Llama hybrid upload; smaller matrices stay on CPU. |
| `RBITNET_HYBRID_OUTPUT` | CUDA on, hybrid off | Llama output head placement; `0`/`false`/`no` disables it. Automatic placement reserves its budget before selecting layers. |
| `RBITNET_HYBRID_LOG` | off | Reserved flag for verbose hybrid diagnostics; current builds always log the selected offload plan at runtime creation. |
| `RBITNET_LLAMA_PAGED_KV` | off | **`1`** enables paged KV storage for **Llama** GGUF (see [USAGE.md](USAGE.md)). |
| `RBITNET_KV_BACKEND` | `cpu` | KV backend hint. `gpu`/`cuda` marks paged KV as GPU-planned and keeps CPU fallback until native KV device storage is available. |
| `RBITNET_KV_QUANT` | `off` | Paged Llama KV format: `off`/`f32`, `q8`, or `q4`. Q8/Q4 use compact pages only (no F32 twin); decode-on-read for attention. Requires `RBITNET_LLAMA_PAGED_KV=1` (or pool). See [USAGE.md](USAGE.md) + `scripts/bench_kv_q8.sh`. |
| `RBITNET_PAGED_KV`, `RBITNET_PAGED_KV_*` | varies | Qwen35 attention scaffolding; Llama paged mode reads `PAGE_TOKENS` / `MAX_PAGES` via [`PagedKvCache`](../crates/bitnet-core/src/paged_kv.rs). |

## Tests only

| Variable | Purpose |
|----------|---------|
| `RBITNET_TEST_GGUF` | Optional path for `bitnet-core` integration smoke tests. |
| `RBITNET_SMOKE_MAX_TOKENS` | Only **`cargo run -p bitnet-core --example engine_smoke`**: caps **`max_tokens`** (default `8`) for long CPU runs on large GGUFs. |

For curl examples and Prometheus names, see [USAGE.md](USAGE.md).
