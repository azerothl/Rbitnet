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
| `RBITNET_ADMIN_TOKEN` | string | (none) | Enables `POST /v1/admin/unload` with matching header. |
| `RBITNET_IDLE_UNLOAD_SECS` | u64 | (none) | After idle, swap engine for stub to free mmap. |

## Core / model loading (`bitnet-core`)

| Variable | Default | Notes |
|----------|---------|--------|
| `RBITNET_MODEL` | (none) | Path to one `.gguf` file. |
| `RBITNET_TOKENIZER` | (auto beside GGUF) | `tokenizer.json` or `tokenizer.model`. |
| `RBITNET_CHAT_FORMAT` | `raw` | Server prompt rendering for chat messages: `raw`, `llama3`, or `chatml`. |
| `RBITNET_CHAT_TEMPLATE` | (none) | Simple custom prompt template; supports `{messages}`, `{prompt}`, `{system}`, `{user}`, `{assistant}` and overrides `RBITNET_CHAT_FORMAT`. |
| `RBITNET_STUB` | off | Synthetic completions; no weights. |
| `RBITNET_TOY` | off | Tiny in-process toy LM; no GGUF. |
| `RBITNET_BACKEND` | `cpu` | e.g. `cpu`, `cuda` — see [USAGE.md](USAGE.md). |
| `RBITNET_ARCHITECTURE` | (from GGUF) | Override `general.architecture`. |
| `RBITNET_MODEL_FAMILY` | `auto` | Architecture hint when no GGUF (stub/toy). |
| `RBITNET_PREFIX_CACHE` | off | Cache full duplicate completions (not KV). |
| `RBITNET_PREFIX_CACHE_MAX_ENTRIES` | `64` | Prefix response cache size. |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | (server) | Same name used by server for HTTP timeout; core cancellation hooks align with server layer. |
| `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_LOAD_BYTES`, `RBITNET_MAX_VRAM_MB`, `RBITNET_BUDGET_MAX_SEQ` | (none) | Load guardrails; see [LIMITATIONS.md](LIMITATIONS.md). |
| `RBITNET_CONTINUOUS_BATCHING`, `RBITNET_SPECULATIVE`, `RBITNET_SPEC_DRAFT_RATIO_*`, `RBITNET_PREFILL_CHUNK_TOKENS` | varies | Scheduler / speculative MVP; see [USAGE.md](USAGE.md). |
| `RBITNET_LLAMA_WEIGHT_MODE` | `auto` | Llama-lineage matrices: **`dense`** (legacy: full `tensor_to_f32` at load, high RAM), **`mmap_quant`** (quantized weights stay in the GGUF mmap; row-wise GEMV), **`auto`** (mmap when every weight tensor uses a supported GGML type for mmap GEMV; otherwise dense). |
| `RBITNET_LLAMA_PAGED_KV` | off | **`1`** enables paged KV storage for **Llama** GGUF (see [USAGE.md](USAGE.md)). |
| `RBITNET_PAGED_KV`, `RBITNET_PAGED_KV_*` | varies | Qwen35 attention scaffolding; Llama paged mode reads `PAGE_TOKENS` / `MAX_PAGES` via [`PagedKvCache`](../crates/bitnet-core/src/paged_kv.rs). |

## Tests only

| Variable | Purpose |
|----------|---------|
| `RBITNET_TEST_GGUF` | Optional path for `bitnet-core` integration smoke tests. |
| `RBITNET_SMOKE_MAX_TOKENS` | Only **`cargo run -p bitnet-core --example engine_smoke`**: caps **`max_tokens`** (default `8`) for long CPU runs on large GGUFs. |

For curl examples and Prometheus names, see [USAGE.md](USAGE.md).
