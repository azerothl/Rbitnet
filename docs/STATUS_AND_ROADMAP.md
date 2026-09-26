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
| Production-grade GPU kernels (FlashAttention-class, fused GEMM/MoE) | **Deferred** — FA2/FA3 = GPU_NATIVE research only, **not** near-term default ([GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md)) |
| KV memory (PagedAttention-style), aggressive cache scheduling | **E2E opt-in** — dense default; `RBITNET_LLAMA_PAGED_KV=1` + `RBITNET_KV_POOL=1` shared phys pages on Llama runtime; reclaim + `/metrics` gauges; see [Research-backed priorities](#research-backed-priorities-2026-09) |
| Full serving pipeline (continuous batching, chunked prefill, prefix cache, graphs, speculative decoding) | **Hooks / MVP** — env-flagged; fused multi-seq + stall-free schedule still open — [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) |
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
| **Prefix KV (tensorial)** | **MVP + radix LRU** | `RBITNET_PREFIX_KV` — dense/paged snaps + LCP agent reuse + radix LRU; `/metrics` `rbitnet_core_prefix_hit`. |
| **Paged KV pool** | **E2E opt-in** | `RBITNET_KV_POOL=1` backs Llama runtime KV with `SharedPhysKvStore`; multi-seq `PagedKvPool`; free-list reclaim; gauges `rbitnet_core_kv_pool_*`. Bench: [`scripts/bench_paged_kv.sh`](../scripts/bench_paged_kv.sh). |
| **Continuous batching** | **Hook → waves** | `RBITNET_CONTINUOUS_BATCHING` — interleaved decode waves when sessions enabled; fused multi-seq matmul still pending. |
| **Chunked prefill** | **Partial** | `RBITNET_PREFILL_CHUNK_TOKENS` slices the prompt loop (one forward step per token). |
| **Full-response prefix cache** | **Optional** | `RBITNET_PREFIX_CACHE` — duplicate **completions**, distinct from prefix KV. |
| **CUDA graphs** | **Metrics + capture hook** | `RBITNET_CUDA_GRAPH` — [`cuda_graph.rs`](../crates/bitnet-core/src/llama/cuda_graph.rs); device capture on stable decode shapes (CUDA). |
| **KV sidecar** | **HTTP MVP** | `RBITNET_KV_SIDECAR_URL` — PUT/GET JSON block tables ([`kv_sidecar.rs`](../crates/bitnet-core/src/kv_sidecar.rs), [KV_SIDECAR_SPEC.md](KV_SIDECAR_SPEC.md)). |
| **Speculative decoding** | **MVP + PLD** | `RBITNET_SPECULATIVE` + default `RBITNET_DRAFT_PATH=ngram` (prompt-lookup); verify/accept; `rbitnet_core_draft_accept`. |

Production-grade throughput still needs fused forwards and GPU-resident KV — see [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) *Done / Next*.

**Frontier-aligned backlog (multi-token prefill, block KV / PagedAttention, continuous batching, prefix-KV + L7 routing):** tracked conceptually against industry stacks (vLLM, TensorRT-LLM, gateway routing patterns); see discussion in [USAGE.md](USAGE.md), [PROFILING.md](PROFILING.md), and [LIMITATIONS.md](LIMITATIONS.md). Prioritized experiments and skip list: [Research-backed priorities](#research-backed-priorities-2026-09).

### Tokenizer and Hugging Face ecosystem

| Today | Gap / direction |
|-------|-----------------|
| **`rbitnet models install`** downloads GGUF + tokenizer files and writes **`rbitnet.manifest.json`** | Does **not** auto-resolve every gated repo graph or run **Transformers** `AutoTokenizer` / `AutoConfig` in-process |
| **`tokenizer.json`** / **`tokenizer.model`** paths via env or beside GGUF | No single “paste Hub id → guaranteed coherent tokenizer + chat template + special tokens” path without user following docs |
| **Direction** | Tighter integration: optional Hub id → resolve tokenizer siblings + validate against GGUF metadata; document chat templates per family; optional thin wrapper that mirrors HF recommended files |

---

## Research-backed priorities (2026-09)

Folded from product research (engine landscape + arXiv ideas for a **native-first GGUF/BitNet** server). Criteria: fit to existing `bitnet-core` hooks > local impact (tok/s, RSS, concurrency ≥4) > cost > mature open code. **Native-first CPU remains the default path**; FlashAttention-2 / FA3 stay **deferred GPU_NATIVE research**, not a near-term default.

Phased detail and experiment gates live in [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) (*Research backlog*). Product-facing quick wins (metrics UX, runner-proxy, provenance) also align with [FUTURE_DIFFERENTIATION.md](FUTURE_DIFFERENTIATION.md).

### Quick wins

| Item | Why / cite | Action |
|------|------------|--------|
| **Metrics surface** — TTFT, decode tok/s, `prefix_hit`, `draft_accept` on `/metrics` + `/ui` | Akasha scrape + akasha-os Models UX parity | Extend [AKASHA_METRICS.md](AKASHA_METRICS.md) series; status line in UI |
| **Model provenance** — SHA256 in catalog/recipes; optional `RBITNET_TRUSTED_MODELS_ONLY` | Ecosystem security (GGUF trust) | Harden `rbitnet models install` / recipes |
| **Structured output** stabilize | SGLang-style FSM already env-flagged | `RBITNET_STRUCTURED_OUTPUT` + JSON-schema tests |
| **Speculative verify + metrics** | [2211.17192](https://arxiv.org/abs/2211.17192) | Solidify accept path; expose `draft_accept` |
| **Prompt-lookup / n-gram draft** (no second GGUF) | PLD / [2304.04487](https://arxiv.org/abs/2304.04487); akasha-os already does PLD | Wire into existing speculative scheduler |
| **BitNet recipe / null-loss criteria** | [2402.17764](https://arxiv.org/abs/2402.17764), [2504.12285](https://arxiv.org/abs/2504.12285) | Document `bitnet-b158` pack + golden gate |
| **Runner-proxy multi-model path** | Ollama-like ops; [RUNNER_PROXY_SPEC.md](RUNNER_PROXY_SPEC.md) | Document as recommended multi-model default |
| **Bench gate vs llama.cpp** | Credible Akasha SLO | One frozen line in [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) per release |
| **Load-failure → retry** | akasha-os P16 pattern | State machine without process restart |

### Medium

| Item | Why / cite | Action |
|------|------------|--------|
| **Paged KV E2E** | PagedAttention [2309.06180](https://arxiv.org/abs/2309.06180) | **Shipped opt-in** — pool + paged attention path; measure RSS/fragmentation @ concurrency 1/4/8 via `scripts/bench_paged_kv.sh` |
| **Radix prefix (agent prompts)** | SGLang / RadixAttention [2312.07104](https://arxiv.org/abs/2312.07104) | **Shipped opt-in** — LRU radix + LCP reuse; `rbitnet_core_prefix_hit`; unit gate ≥70% after warm-up |
| **Chunked prefill + stall-free schedule** | Sarathi-Serve [2403.02310](https://arxiv.org/abs/2403.02310) (Orca iteration-level batching) | Token budget / iteration on `RBITNET_CONTINUOUS_BATCHING` for ≥2–4 Akasha sessions |
| **Continuous batching fused waves** | vLLM-class serving | Complete Phase B: single forward for N seq |
| **CPU tiled attention + KV Q8** | SlimAttention [2407.07304](https://arxiv.org/abs/2407.07304) | Prototype CPU path; measure decode latency + RSS (`RBITNET_KV_QUANT=q8`) |
| **KV asymmetry (after Q8)** | KIVI [2402.02750](https://arxiv.org/abs/2402.02750) | K per-channel / V per-token; golden/PPL before prod |
| **BitNet ternary kernels (Rust SIMD)** | bitnet.cpp [2502.11880](https://arxiv.org/abs/2502.11880), [2410.16144](https://arxiv.org/abs/2410.16144) | **Shipped microbench** — I2_S pack + TL2-LUT in `kernels.rs`; `scripts/bench_bitnet_kernels.sh`; row in BENCHMARKS_RESULTS |
| **Tune profiles** | Product UX | `interactive` / `batch` / `bitnet-cpu` via `rbitnet tune` |

### Strategic

| Item | Why / cite | Position |
|------|------------|----------|
| **GPU_NATIVE** — device KV, fused attention/GEMV | Industry FA / FlashInfer-class | After frozen CPU benches; see [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) |
| **BitNet packed GPU kernels** | 2B4T report [2504.12285](https://arxiv.org/abs/2504.12285) | Research after CPU ternary parity |
| **Lookahead decoding** | [2402.02057](https://arxiv.org/abs/2402.02057) | Research (tree attention / FLOPs); PLD first |
| **Prefill/decode disaggregation** | DistServe [2401.09670](https://arxiv.org/abs/2401.09670) | After batching + paged KV; multi-GPU cluster — not desktop default |
| **Tensor parallel / multi-GPU** | vLLM / TRT-LLM | After single-GPU correct |
| **semver Engine crate** for embed | Alternative to HTTP for Akasha | Without breaking NATIVE_FIRST |
| **Metal / optional MLX** | Apple path | Optional; stub today |

### Explicitly deferred (do not chase near-term)

| Topic | Cite | Reason |
|-------|------|--------|
| **FlashAttention-2 / FA3 as default path** | [2205.14135](https://arxiv.org/abs/2205.14135), [2307.08691](https://arxiv.org/abs/2307.08691) | GPU-first; keep as GPU_NATIVE research only — **native-first CPU stays default** |
| **INT-FlashAttention / TurboAttention** | [2409.16997](https://arxiv.org/abs/2409.16997), [2412.08585](https://arxiv.org/abs/2412.08585) | After GPU_NATIVE epic |
| **DistServe disagg as product default** | [2401.09670](https://arxiv.org/abs/2401.09670) | Cluster KV transfer — out of local/desktop target |
| **DeepSeek-V3 MoE / MLA native** | [2412.19437](https://arxiv.org/abs/2412.19437) | Pure MoE GGUF still refused; too invasive |
| **Medusa / EAGLE / EAGLE-2** | [2401.10774](https://arxiv.org/abs/2401.10774), [2401.15077](https://arxiv.org/abs/2401.15077), [2406.16858](https://arxiv.org/abs/2406.16858) | Draft heads / fine-tune — incompatible with download-and-serve GGUF |
| **AWQ / GPTQ as primary format** | ecosystem | Conflicts with GGUF native-first; Akasha may route vLLM elsewhere |
| **T-MAC as second LUT stack** | [2407.00088](https://arxiv.org/abs/2407.00088) | Cross-read only; do not port in parallel with bitnet.cpp patterns |
| **FFI llama.cpp / second engine in TCB** | — | Violates [NATIVE_FIRST.md](NATIVE_FIRST.md) |
| **Novel AutoTokenizer research** | — | Keep recipe sidecars + checksums |

---

## Next todos (suggested order)

1. **Measurements** — Refresh the **frozen baseline** row in `[BENCHMARKS.md](BENCHMARKS.md)` each release (SHA, p50/p95, optional RSS); publish vs-llama.cpp line in [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md).
2. **Quick wins (metrics + provenance + PLD)** — See [Research-backed priorities](#research-backed-priorities-2026-09); prefer these before large scheduler merges.
3. **Profiling** — Add a dated file under `[profiling/](profiling/README.md)` when kernels or scheduler change materially.
4. **Timeouts** — Optional: stronger isolation after HTTP 504 (trade-offs in [LIMITATIONS.md](LIMITATIONS.md)); default remains cooperative cancel + documented pool behaviour.
5. **Tokenizer / HF** — Deeper Hub alignment remains under [Advanced inference stack gaps](#advanced-inference-stack-gaps-vs-industry-serving); extend tensor aliases as real exports appear.
6. **Security** — For internet-facing edges: TLS + rate limits at the proxy ([DEPLOYMENT.md](DEPLOYMENT.md)); in-process rate limiting remains out of scope.
7. **Inference epic (medium)** — Paged KV E2E, radix agent hit-rate, Sarathi-style chunked prefill — [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md).

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
| `[INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md)` | Serving epic + research backlog (arXiv-aligned) |
| `[GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md)` | Native CUDA path; FA2/FA3 deferred |
| `[FUTURE_DIFFERENTIATION.md](FUTURE_DIFFERENTIATION.md)` | Product differentiation axes |
| [`CHANGELOG.md`](../CHANGELOG.md)          | Keep a Changelog–style release notes    |
| [`training/README.md`](../training/README.md) | Optional Python LoRA/SFT + `rbitnet train` |
