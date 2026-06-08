# Rbitnet — implementation status and roadmap

This document complements `[PLAN_PRODUCTION.md](PLAN_PRODUCTION.md)`: it tracks **what is implemented**, **what is partial or missing**, and **suggested next steps**. It should be updated when major features land or scope changes.

---

## Summary


| Area                                                           | Status                                                        |
| -------------------------------------------------------------- | ------------------------------------------------------------- |
| HTTP server (OpenAI-shaped API, limits, auth, health, metrics) | **Done**                                                      |
| CI (build, test, clippy, audit)                                | **Done**                                                      |
| Release binaries (GitHub Actions on tag `v`*)                  | **Done** — see `[RELEASE.md](RELEASE.md)`                     |
| Core inference (GGUF, Llama forward, tokenizer, stub/toy)      | **Done** — Llama **`auto`** mmap-quant GEMV when supported (`RBITNET_LLAMA_WEIGHT_MODE`); legacy **`dense`** full `f32` load; optional **sliding-window** + **Q/K RMSNorm** when tensors/metadata are present |
| Llama **golden** parity (llama.cpp reference)                 | **Optional** — `docs/GOLDEN_TESTS.md`, `cargo test -p bitnet-core optional_golden_*`, workflow `.github/workflows/golden-optional.yml` |
| Optional train/export docs + Python recipe + `rbitnet train`     | **Done** — see [`training/README.md`](../training/README.md), [`TRAINING_AND_COMPATIBILITY.md`](TRAINING_AND_COMPATIBILITY.md) |
| Performance baselines (published numbers)                      | **Frozen procedure + rows** — see `[BENCHMARKS.md](BENCHMARKS.md)` |
| **Perf vs llama.cpp (CPU)**                                    | **Measured gap** — compare with `llama-bench` vs `scripts/compare_llamacpp_rbitnet.*`; parity is **not** claimed until frozen rows show it (optional BLAS / future ggml bridge narrow the gap) |
| Profiling report (hot paths, prioritized follow-ups)           | **Checklist + archived snapshots** — see `[PROFILING.md](PROFILING.md)`, `[profiling/](profiling/README.md)` |
| Production-grade GPU kernels (FlashAttention-class, fused GEMM/MoE) | **Not in scope today** — see [Advanced inference stack gaps](#advanced-inference-stack-gaps-vs-industry-serving) |
| KV memory (PagedAttention-style), aggressive cache scheduling | **Not implemented** — dense per-request KV; see same section |
| Full serving pipeline (continuous batching, chunked prefill, prefix cache, graphs, speculative decoding) | **Not implemented** — single-stream completions per worker slot |
| Hugging Face–centric “automatic” tokenizer + model pairing      | **Partial** — `models install`, manifests; no embedded Transformers auto-config |
| “Prod ready” exit criteria (all of PLAN)                       | **Not claimed** — several doc-only / measurement items remain |


---

## By phase (vs `PLAN_PRODUCTION.md`)

### Phase 0 — Scoping


| Item                                                           | Status                                                     |
| -------------------------------------------------------------- | ---------------------------------------------------------- |
| Prod scope (local / trusted network)                           | Documented in `PLAN_PRODUCTION.md`                         |
| Reference GGUF roles (stub, toy, optional `RBITNET_TEST_GGUF`) | Documented; CI uses stub/toy                               |
| Indicative SLO table                                           | Documented; measure using **[BENCHMARKS.md](BENCHMARKS.md)** frozen baseline procedure per release |


### Phase 1 — Reliability and limits


| Item                                    | Status                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| JSON body size cap                      | **Done** — `RBITNET_MAX_BODY_BYTES`, HTTP 413                                               |
| Prompt length / `max_tokens` caps       | **Done** — `RBITNET_MAX_PROMPT_CHARS`, `RBITNET_MAX_TOKENS_CAP`, HTTP 400                   |
| Per-request inference timeout           | **Done** — `RBITNET_INFERENCE_TIMEOUT_SECS`, HTTP 504                                       |
| OOM / mmap / missing file               | **Typed errors** in core; no panics on normal API paths                                     |
| Concurrency limit                       | **Done** — `RBITNET_MAX_CONCURRENT`, HTTP 503 when saturated                                |
| Sequential + parallel integration tests | **Done** — `bitnet-server` tests (including parallel stub)                                  |
| Effective context vs model max          | **Partial** — optional `RBITNET_MAX_PROMPT_TOKENS` (tokenizer length); full prompt+decode vs architecture `max_seq` remains engine-side |


**Gap:** HTTP **504** aborts the blocking task handle and sends cooperative cancel into the engine; the underlying pool thread **may still finish** the closure on some schedulers ([LIMITATIONS.md](LIMITATIONS.md)).

### Phase 2 — Performance and resources


| Item                                           | Status                                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Reproducible benchmarks doc                    | `[BENCHMARKS.md](BENCHMARKS.md)` — **frozen baseline procedure** + historical measurement rows |
| Criterion kernel benches                       | `cargo bench -p bitnet-core`                                                             |
| Profiling write-up                             | `[PROFILING.md](PROFILING.md)` + `[profiling/](profiling/README.md)` dated snapshots      |
| CPU optimizations (SIMD, threads, allocations) | **Ongoing** / backend abstraction + scheduler scaffolding landed                          |
| RAM ceiling per model                          | **Partial** — **Peak RAM** table in `[BENCHMARKS.md](BENCHMARKS.md)`; fill when publishing baselines |


### Phase 3 — Security and exposure


| Item                   | Status                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Env paths without `..` | **Done** — `validate_no_parent_components` in core                                                                         |
| API key                | **Done** — `RBITNET_API_KEY`, `Authorization` / `X-API-Key` on API routes; `/health`, `/ready`, `/metrics` unauthenticated |
| CORS + bind warning    | **Done** — `RBITNET_CORS_ANY`, log warning on `0.0.0.0` / `[::]`                                                           |
| `cargo audit` in CI    | **Done** — `.github/workflows/ci.yml`                                                                                      |
| CLI `--api-key` / `--bind` | **Done** — `rbitnet serve` and `rbitnet-server` (applied only when env unset); see `[USAGE.md](USAGE.md)` |


### Phase 4 — Observability


| Item                      | Status                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------- |
| Prometheus-style metrics  | **Done** — `GET /metrics`                                                             |
| Structured tracing        | **Done** — `x-request-id` on outer layer; span fields `method`, `path`, `request_id`  |
| Liveness / readiness      | **Done** — `GET /health`, `GET /ready` (`Engine::is_ready`)                           |
| Startup summary of limits | **Done** — logged in `rbitnet-server` main                                            |
| Config validation         | **Done** — `[ENV_REFERENCE.md](ENV_REFERENCE.md)` + startup validation for zero/absurd limits |


### Phase 5 — Quality and compatibility


| Item                                                       | Status                                                            |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| GGML types documented                                      | `[BITNET_SPEC.md](BITNET_SPEC.md)` + `types.rs`                   |
| Tensor name aliases                                        | **Done** — `tensor_first_of` (e.g. `lm_head`, `attn_out`)         |
| Regression / golden tests                                  | Kernel goldens in CI; optional Llama greedy-token golden via `RBITNET_GOLDEN_JSON` + `docs/GOLDEN_TESTS.md`; optional GGUF smoke via `RBITNET_TEST_GGUF` |
| Second exporter (e.g. llama.cpp vs BitNet) automated tests | **Partial** — optional `RBITNET_TEST_GGUF` mmap + optional engine load when tokenizer beside GGUF; **greedy first-token** optional test when golden JSON is provided |
| Release process + semver                                   | `[RELEASE.md](RELEASE.md)`                                        |
| Prebuilt binaries on tag                                   | **Done** — `.github/workflows/release.yml`                        |


### Phase 6 — Documentation


| Item                                      | Status                                               |
| ----------------------------------------- | ---------------------------------------------------- |
| USAGE, TRAINING_AND_COMPATIBILITY, README, `training/` | **Updated** — optional fine-tune recipe + CLI helpers |
| Limitations                               | `[LIMITATIONS.md](LIMITATIONS.md)`                   |
| Deployment (systemd, nginx, Docker)       | `[DEPLOYMENT.md](DEPLOYMENT.md)` + root `Dockerfile` |


---

## Advanced inference stack gaps vs industry serving

Rbitnet today targets **correct GGUF execution**, a **small HTTP surface**, and **incremental CUDA helpers** (for example experimental `qwen35moe`). It is **not** yet comparable to stacks built around highly fused kernels and batched schedulers (vLLM, TensorRT-LLM, Triton servers, etc.). The following items are **deliberately tracked** as roadmap / differentiation axes—not promises on a fixed timeline.

### GPU kernels (attention / GEMM / MoE)

| Topic | Industry stack | Rbitnet today |
|-------|----------------|---------------|
| Attention | **FlashAttention**-style fused softmax–matmul, warp-specialized paths | Mostly **dequant on host** + discrete GEMV/GEMM calls; **no** FA3-class fused attention kernel for general GGUF layouts |
| Dense linear / MoE | **CUTLASS**, cuBLASLt epilogues, vendor MoE routes (**TRT-LLM**, etc.) | Partial CUDA paths (e.g. logits GEMV); MoE and blocks still carry **non-fused**, **per-op** work on CPU-held payloads |
| Fusion | Operator fusion, persistent kernels | **Unfused** chains; few opportunities for kernel fusion until layouts and memory residency move wholesale to device |

**Implication:** throughput and latency will remain below specialized servers until fused kernels (or a delegation layer to one) land for the supported architectures.

### KV memory and cache management

| Topic | Industry stack | Rbitnet today |
|-------|----------------|---------------|
| **PagedAttention** (vLLM) | Fixed-size **pages** of KV, block tables, high batch utilisation | **Dense** per-layer KV buffers sized for `max_seq` (architecture-dependent) |
| Memory / batching | VRAM–throughput trade-offs, **preemption**-friendly blocks | **One completion** per slot; no cross-request KV block pool |
| “Aggressive” cache | Eviction, prefix sharing at block level | **No** first-class paged or block-mapped KV API |

**Implication:** long context and high concurrency are **memory- and CPU-copy–heavy** relative to a paged design; this is a major future lever for multi-tenant or long-context serving.

### Serving pipeline (orchestration)

| Feature | Status (Jun 2026) | Notes |
|---------|-------------------|-------|
| **Live token streaming** | **Shipped (MVP)** | SSE token deltas via [`stream.rs`](../crates/bitnet-core/src/stream.rs) and `live_stream_chat_completion` in `bitnet-server` (not post-generation chunking). |
| **Prefix KV (tensorial)** | **MVP (dense)** | `RBITNET_PREFIX_KV` — dense KV snapshots + partial prefill; **not** with `RBITNET_LLAMA_PAGED_KV` yet. Radix block index in [`prefix_kv.rs`](../crates/bitnet-core/src/prefix_kv.rs). |
| **Paged KV pool** | **API + Engine hook** | `PagedKvPool`, `RBITNET_KV_POOL=1` on [`Engine`](../crates/bitnet-core/src/inference.rs); shared physical pages across sequences. |
| **Continuous batching** | **Hook → waves** | `RBITNET_CONTINUOUS_BATCHING` — interleaved decode waves when sessions enabled; fused multi-seq matmul still pending. |
| **Chunked prefill** | **Partial** | `RBITNET_PREFILL_CHUNK_TOKENS` slices the prompt loop (one forward step per token). |
| **Full-response prefix cache** | **Optional** | `RBITNET_PREFIX_CACHE` — duplicate **completions**, distinct from prefix KV. |
| **CUDA graphs** | **Metrics + capture hook** | `RBITNET_CUDA_GRAPH` — [`cuda_graph.rs`](../crates/bitnet-core/src/llama/cuda_graph.rs); device capture on stable decode shapes (CUDA). |
| **KV sidecar** | **HTTP MVP** | `RBITNET_KV_SIDECAR_URL` — PUT/GET JSON block tables ([`kv_sidecar.rs`](../crates/bitnet-core/src/kv_sidecar.rs), [KV_SIDECAR_SPEC.md](KV_SIDECAR_SPEC.md)). |
| **Speculative decoding** | **MVP** | `RBITNET_SPECULATIVE`, `RBITNET_DRAFT_MODEL` (secondary GGUF when configured). |

Production-grade throughput still needs fused forwards and GPU-resident KV — see [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) *Done / Next*.

**Frontier-aligned backlog (multi-token prefill, block KV / PagedAttention, continuous batching, prefix-KV + L7 routing):** tracked conceptually against industry stacks (vLLM, TensorRT-LLM, gateway routing patterns); see discussion in [USAGE.md](USAGE.md), [PROFILING.md](PROFILING.md), and [LIMITATIONS.md](LIMITATIONS.md).

### Tokenizer and Hugging Face ecosystem

| Today | Gap / direction |
|-------|-----------------|
| **`rbitnet models install`** downloads GGUF + tokenizer files and writes **`rbitnet.manifest.json`** | Does **not** auto-resolve every gated repo graph or run **Transformers** `AutoTokenizer` / `AutoConfig` in-process |
| **`tokenizer.json`** / **`tokenizer.model`** paths via env or beside GGUF | No single “paste Hub id → guaranteed coherent tokenizer + chat template + special tokens” path without user following docs |
| **Direction** | Tighter integration: optional Hub id → resolve tokenizer siblings + validate against GGUF metadata; document chat templates per family; optional thin wrapper that mirrors HF recommended files |

---

## Next todos (suggested order)

1. **Measurements** — Refresh the **frozen baseline** row in `[BENCHMARKS.md](BENCHMARKS.md)` each release (SHA, p50/p95, optional RSS).
2. **Profiling** — Add a dated file under `[profiling/](profiling/README.md)` when kernels or scheduler change materially.
3. **Timeouts** — Optional: stronger isolation after HTTP 504 (trade-offs in [LIMITATIONS.md](LIMITATIONS.md)); default remains cooperative cancel + documented pool behaviour.
4. **Tokenizer / HF** — Deeper Hub alignment remains under [Advanced inference stack gaps](#advanced-inference-stack-gaps-vs-industry-serving); extend tensor aliases as real exports appear.
5. **Security** — For internet-facing edges: TLS + rate limits at the proxy ([DEPLOYMENT.md](DEPLOYMENT.md)); in-process rate limiting remains out of scope.
6. **Inference epic** — See [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) for PagedAttention-class backlog.

---

## Related docs


| Doc                                        | Role                                    |
| ------------------------------------------ | --------------------------------------- |
| `[PLAN_PRODUCTION.md](PLAN_PRODUCTION.md)` | Target exit criteria and phased roadmap |
| `[USAGE.md](USAGE.md)`                     | Runtime env vars and curl examples      |
| `[RELEASE.md](RELEASE.md)`                 | Tags, semver, GitHub release binaries   |
| `[LIMITATIONS.md](LIMITATIONS.md)`         | Known constraints                       |
| `[BENCHMARKS.md](BENCHMARKS.md)`           | Where to record performance numbers     |
| `[PROFILING.md](PROFILING.md)`             | How to profile CPU hot paths            |
| `[ENV_REFERENCE.md](ENV_REFERENCE.md)`   | Consolidated `RBITNET_*` index          |
| `[INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md)` | Long-term PagedAttention-class backlog |
| [`CHANGELOG.md`](../CHANGELOG.md)          | Keep a Changelog–style release notes    |
| [`training/README.md`](../training/README.md) | Optional Python LoRA/SFT + `rbitnet train` |
