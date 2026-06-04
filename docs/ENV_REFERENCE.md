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
| `RBITNET_IDLE_UNLOAD_SECS` | u64 | (none) | After idle, swap engine for stub to free mmap. |
| `RBITNET_CHAT_BASE_URL` | URL | `http://127.0.0.1:8080/v1` | Default base URL for `rbitnet chat`; `/v1` is appended when omitted. |

## Runner proxy (`rbitnet-proxy`)

| Variable | Type | Default | Notes |
|----------|------|---------|--------|
| `RBITNET_PROXY_BIND` | `host:port` | `RBITNET_BIND` or `127.0.0.1:8080` | Parent proxy listen address. |
| `RBITNET_MODEL_REGISTRY` | path | required | JSON registry; each model id becomes one supervised worker in local mode. |
| `RBITNET_RUNNER_BIN` | path/name | sibling `rbitnet-runner`, then `PATH` | Worker executable spawned per model id. |
| `RBITNET_RUNNER_READY_TIMEOUT_SECS` | u64 | `60` | Time for child `GET /ready` to become 2xx. |
| `RBITNET_PROXY_REQUEST_TIMEOUT_SECS` | u64 | `600` | Upstream native child request timeout. |
| `RBITNET_INFERENCE_BACKEND` | string | `local` | Normal builds accept local native workers only. External delegation values require the dev-only Cargo feature `experimental-external-backends`. |
| `RBITNET_VLLM_BASE_URL` | URL | (none) | Ignored by default builds; only used when an experimental external backend is compiled in. |

## Core / model loading (`bitnet-core`)

| Variable | Default | Notes |
|----------|---------|--------|
| `RBITNET_MODEL` | (none) | Path to one `.gguf` file. |
| `RBITNET_TOKENIZER` | (auto beside GGUF) | `tokenizer.json` or `tokenizer.model`. |
| `RBITNET_CHAT_FORMAT` | `raw` | Server prompt rendering for chat messages: `raw`, `llama3`, or `chatml`; if unset and no custom template exists, `tokenizer_config.json` `chat_template` is used for common Llama 3 / ChatML templates. |
| `RBITNET_CHAT_TEMPLATE` | (none) | Simple custom prompt template; supports `{messages}`, `{prompt}`, `{system}`, `{user}`, `{assistant}` and overrides `RBITNET_CHAT_FORMAT` / tokenizer config discovery. |
| `RBITNET_STUB` | off | Synthetic completions; no weights. |
| `RBITNET_TOY` | off | Tiny in-process toy LM; no GGUF. |
| `RBITNET_BACKEND` | `cpu` | e.g. `cpu`, `cuda`, `hybrid` — see [USAGE.md](USAGE.md). |
| `RBITNET_ARCHITECTURE` | (from GGUF) | Override `general.architecture`. |
| `RBITNET_MODEL_FAMILY` | `auto` | Architecture hint when no GGUF (stub/toy). |
| `RBITNET_PREFIX_CACHE` | off | Cache full duplicate completions (not KV). |
| `RBITNET_PREFIX_CACHE_MAX_ENTRIES` | `64` | Prefix response cache size. |
| `RBITNET_PREFIX_KV` | off | Dense KV snapshot reuse for shared prompt prefixes (prefill skip). |
| `RBITNET_PREFIX_KV_MAX_ENTRIES` | `32` | LRU size for execution-time prefix KV snapshots. |
| `RBITNET_CUDA_GRAPH` | off | Enable CUDA graph decode metrics path (`llama/cuda_graph.rs`). |
| `RBITNET_MTP_K` | `1` | Multi-token burst width when `>1` (Atlas-style MTP scheduler hook). |
| `RBITNET_KV_POOL_MAX_SEQS` | `8` | Max concurrent sequences in `PagedKvPool`. |
| `RBITNET_KV_SIDECAR_URL` | (none) | Optional external KV sidecar base URL; see [KV_SIDECAR_SPEC.md](KV_SIDECAR_SPEC.md). |
| `RBITNET_KV_SIDECAR_TIMEOUT_SECS` | `30` | Sidecar HTTP timeout. |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | (server) | Same name used by server for HTTP timeout; core cancellation hooks align with server layer. |
| `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_LOAD_BYTES`, `RBITNET_MAX_VRAM_MB`, `RBITNET_BUDGET_MAX_SEQ` | (none) | Load guardrails; see [LIMITATIONS.md](LIMITATIONS.md). |
| `RBITNET_CONTINUOUS_BATCHING`, `RBITNET_SPECULATIVE`, `RBITNET_SPEC_DRAFT_RATIO_*`, `RBITNET_PREFILL_CHUNK_TOKENS` | varies | Scheduler, chunked prefill hooks, and speculative draft path; see [USAGE.md](USAGE.md). |
| `RBITNET_DRAFT_PATH` | `target` | Speculative draft source when `RBITNET_SPECULATIVE=1`: `target`, `ngram`, or `toy`. |
| `RBITNET_DRAFT_MODEL` | (none) | Path reserved for a small GGUF draft model. Current builds recognize the path and fall back to the lightweight n-gram draft until separate draft-model verification is wired. |
| `RBITNET_STRUCTURED_OUTPUT` | `off` | Optional sampler mask: `json` / `tool` enables the ASCII/byte-token JSON FSM mask before sampling. |
| `RBITNET_LLAMA_WEIGHT_MODE` | `auto` | Llama-lineage matrices: **`dense`** (legacy: full `tensor_to_f32` at load, high RAM), **`mmap_quant`** (quantized weights stay in the GGUF mmap; row-wise GEMV), **`auto`** (mmap when every weight tensor uses a supported GGML type for mmap GEMV; otherwise dense). |
| `RBITNET_QUANT_KERNEL` | `auto` | Quantized matvec backend: `auto`/CPU parallel, `scalar`, or `cuda` to use optional native `rbitnet_cuda_quant*` symbols for `Q4_K`, `Q6_K`, `Q4_0`, `Q8_0` with CPU fallback. |
| `RBITNET_QUANT_PAR_MIN_ROWS` | `128` | Minimum output rows before the CPU quantized matvec path splits work across a reusable thread pool (`rayon`). |
| `RBITNET_BLAS` | off | If `1`/`true`/`yes`, loads **OpenBLAS** dynamically (`cblas_sgemv`) for Llama **CPU/hybrid** attention score GEMV and **dense** `f32` matvec helpers when the library is found on `PATH` / `LD_LIBRARY_PATH` / default search. See [USAGE.md](USAGE.md). |
| `RBITNET_LLAMA_MATMUL` | (unset) | Reserved hook: `ggml` requests a future native ggml bridge (still Rust kernels today). Meaningful only when `bitnet-core` is built with `--features experimental-ggml-kernels`; see [GOLDEN_TESTS.md](GOLDEN_TESTS.md). |
| `RBITNET_HYBRID_POLICY` | `layers` | Hybrid layer-selection policy: `layers` keeps explicit/early-layer behavior, `hotcold` selects deeper hot decode layers first, `auto` uses explicit `RBITNET_HYBRID_LAYERS` when present then hot/cold selection. |
| `RBITNET_HYBRID_LAYERS` | `auto` | With `RBITNET_BACKEND=hybrid`, comma/range list of Llama layers to offload (`0`, `0-3`, `0,2,4`). If unset, the loader selects early layers within `RBITNET_HYBRID_MAX_VRAM_MB`. |
| `RBITNET_HYBRID_MAX_VRAM_MB` | `512` | Approximate f32 weight upload budget for automatic Llama hybrid offload. This is a soft planning budget, not a hard CUDA allocator limit. |
| `RBITNET_HYBRID_MIN_ROWS` | `512` | Minimum matrix output rows for Llama hybrid upload; smaller matrices stay on CPU. |
| `RBITNET_HYBRID_OUTPUT` | off | If enabled, also attempts to offload the Llama output head. This can use substantial VRAM. |
| `RBITNET_HYBRID_LOG` | off | Reserved flag for verbose hybrid diagnostics; current builds always log the selected offload plan at runtime creation. |
| `RBITNET_LLAMA_PAGED_KV` | off | **`1`** enables paged KV storage for **Llama** GGUF (see [USAGE.md](USAGE.md)). |
| `RBITNET_KV_BACKEND` | `cpu` | KV backend hint. `gpu`/`cuda` marks paged KV as GPU-planned and keeps CPU fallback until native KV device storage is available. |
| `RBITNET_KV_QUANT` | `off` | Paged Llama KV format: `off`/`f32`, `q8`, or `q4`. Quantized pages decode K/V heads on demand during attention. |
| `RBITNET_PAGED_KV`, `RBITNET_PAGED_KV_*` | varies | Qwen35 attention scaffolding; Llama paged mode reads `PAGE_TOKENS` / `MAX_PAGES` via [`PagedKvCache`](../crates/bitnet-core/src/paged_kv.rs). |

## Tests only

| Variable | Purpose |
|----------|---------|
| `RBITNET_TEST_GGUF` | Optional path for `bitnet-core` integration smoke tests. |
| `RBITNET_SMOKE_MAX_TOKENS` | Only **`cargo run -p bitnet-core --example engine_smoke`**: caps **`max_tokens`** (default `8`) for long CPU runs on large GGUFs. |

For curl examples and Prometheus names, see [USAGE.md](USAGE.md).
