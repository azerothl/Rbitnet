# Inference stack v2 — backlog (epic)

This document splits the **long-term** items from [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) (*Advanced inference stack gaps* + *Research-backed priorities*) into phases. It is **not** part of the short “prod-ready” checklist in [PLAN_PRODUCTION.md](PLAN_PRODUCTION.md); ship cadence is independent.

**Default path:** native-first **CPU** GGUF/BitNet. FlashAttention-2 / FA3 are **deferred** under [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — research only, not the near-term default.

## Goals

- Raise throughput and memory efficiency toward industry stacks (vLLM-class patterns) without breaking the existing **single-process** OpenAI API.
- Prefer ideas that hook into existing `bitnet-core` scaffolding (paged KV, prefix radix, continuous batching, speculative MVP) over porting Python serving stacks.

## Phase summary (Done / Next)

| Phase | Done (hooks / MVP) | Next (production) |
|-------|-------------------|-------------------|
| **A — KV and memory** | `KvStorage` / `PagedSeqKv` (`RBITNET_LLAMA_PAGED_KV`); **`RBITNET_KV_POOL=1` E2E** shared phys + reclaim + pool gauges | GPU-resident pages; then fused multi-seq |
| **B — Scheduling** | `run_batch` / `run_batch_waves`; `PrefillDecodeQueue`; `RBITNET_CONTINUOUS_BATCHING` interleaved decode; chunked prefill env; **measured** via `rbitnet_core_scheduler_decode_waves_total` + `rbitnet tune throughput` | Fused multi-seq matmul; **stall-free chunked prefill** ([2403.02310](https://arxiv.org/abs/2403.02310)) |
| **C — Cache semantics** | Dense/paged prefix KV (`RBITNET_PREFIX_KV`); radix LRU + LCP agent reuse; **`rbitnet_core_prefix_hit`** on `/metrics` | L7 sticky routing notes |
| **C — Streaming** | Live SSE (`StreamEvent`, `complete_streaming`) | GGUF load tests under sustained concurrency |
| **D — GPU decode** | `CudaDecodeGraph` metrics + capture hook; `RBITNET_KV_BACKEND=gpu` planning bit; cuBLASLt M=1 policy env | Full graph replay; fused norm+quant — **after** CPU benches; FA2/FA3 not default |
| **E — Speculative / CPU attention / BitNet** | Speculative MVP (`RBITNET_SPECULATIVE`); KV quant env; ternary CPU path | PLD/n-gram ([2211.17192](https://arxiv.org/abs/2211.17192)); SlimAttention+KV Q8 ([2407.07304](https://arxiv.org/abs/2407.07304)); Rust SIMD vs bitnet.cpp ([2502.11880](https://arxiv.org/abs/2502.11880)) |

## Phase A — KV and memory

1. **Block-structured KV** (step toward PagedAttention [2309.06180](https://arxiv.org/abs/2309.06180)): fixed token pages, block tables per sequence, predictable memory for long context.
2. **Eviction / pooling** across requests only after block KV exists (today KV is dense per request unless pool enabled).
3. **KV quantization ladder:** measure `RBITNET_KV_QUANT=q8` (RSS + golden) before asymmetric 2-bit (KIVI [2402.02750](https://arxiv.org/abs/2402.02750)).

## Phase B — Scheduling

1. **Continuous batching** (optional mode): pack compatible requests into forward waves.
2. **Chunked prefill** coordinated with decode (beyond fixed `RBITNET_PREFILL_CHUNK_TOKENS` slice loops) — Sarathi-Serve stall-free schedule [2403.02310](https://arxiv.org/abs/2403.02310); conceptual base Orca (OSDI’22).
3. Token **budget per iteration** so one large prefill cannot freeze other sessions’ decode (priority once ≥2–4 concurrent Akasha sessions).

## Phase C — Cache semantics

1. **Prefix KV reuse** for shared prompt prefixes (distinct from `RBITNET_PREFIX_CACHE` full-response cache).
2. **RadixAttention-style** tree + LRU ([2312.07104](https://arxiv.org/abs/2312.07104)) for repeated system/tool prompts; expose `prefix_hit` on `/metrics`.
3. **L7 / sticky routing** notes for multi-replica setups — align with [LIMITATIONS.md](LIMITATIONS.md).

## Phase D — Kernel fusion (GPU)

1. Fused attention / GEMV paths for supported GGUF layouts where residency allows — **after** CPU baseline freeze.
2. **FlashAttention-2 / FA3** ([2205.14135](https://arxiv.org/abs/2205.14135), [2307.08691](https://arxiv.org/abs/2307.08691)): IO-aware *ideas* useful; **implementation stays deferred GPU_NATIVE research**, not the default serving path. See [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md).
3. Compare against external high-performance backends as benchmarks only; in-tree delegation remains dev-only behind `experimental-external-backends`.

## Phase E — Speculative decode, CPU attention, BitNet kernels

1. **Speculative MVP → production metrics:** verify/accept lossless frame [2211.17192](https://arxiv.org/abs/2211.17192); prefer **prompt-lookup / n-gram** drafts over a second GGUF on CPU.
2. **Lookahead** [2402.02057](https://arxiv.org/abs/2402.02057): research only (tree attention); do not block PLD.
3. **CPU attention:** SlimAttention-style 1D tiling + KV INT8 [2407.07304](https://arxiv.org/abs/2407.07304) — realistic without CUDA; FA 2D-tiling is a poor CPU fit.
4. **BitNet ternary:** inventory Rbitnet kernels vs I2_S/TL2 ([2502.11880](https://arxiv.org/abs/2502.11880), [2410.16144](https://arxiv.org/abs/2410.16144)); **reimplement** LUT/MAD patterns in Rust SIMD — no bitnet.cpp FFI ([NATIVE_FIRST.md](NATIVE_FIRST.md)).
5. **BitNet b1.58 product contract:** recipes + null-loss criteria ([2402.17764](https://arxiv.org/abs/2402.17764), [2504.12285](https://arxiv.org/abs/2504.12285)); packed GPU kernels = later research.

## Next experiments (concrete gates)

| # | Experiment | Gate |
|---|------------|------|
| 1 | **Paged KV E2E** on TinyLlama Q4 + BitNet 2B @ concurrency 1/4/8 vs contiguous | No golden regression; report RSS, fragmentation, decode tok/s — run `scripts/bench_paged_kv.sh` |
| 2 | **Radix prefix agent** — 50 Akasha-like reqs (same system+tools) | `prefix_hit` ≥70% after warm-up — unit test `agent_style_prefix_hit_rate_after_warmup`; series `rbitnet_core_prefix_hit` |
| 3 | **PLD / n-gram speculative** in existing scheduler | `draft_accept` + TTFT/decode — `prompt_lookup_draft` + verify/accept; series `rbitnet_core_draft_accept` |
| 4 | **KV Q8 then KIVI-style asymmetry** | Bench RSS + PPL/golden; Q8 default decision before 2-bit |
| 5 | **BitNet matmul microbench** vs bitnet.cpp I2_S/TL2 patterns (same numerics, Rust SIMD/LUT) | One line in [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) via `scripts/bench_bitnet_kernels.sh` |

## Explicitly deferred (serving epic)

Do **not** schedule as near-term default work (full rationale in [STATUS_AND_ROADMAP.md — Research-backed priorities](STATUS_AND_ROADMAP.md#research-backed-priorities-2026-09)):

| Topic | arXiv / note |
|-------|----------------|
| DistServe prefill/decode disagg as desktop default | [2401.09670](https://arxiv.org/abs/2401.09670) |
| DeepSeek-V3 MoE / MLA native load | [2412.19437](https://arxiv.org/abs/2412.19437) |
| FA2/FA3 / INT-Flash / TurboAttention as default | [2205.14135](https://arxiv.org/abs/2205.14135), [2307.08691](https://arxiv.org/abs/2307.08691), [2409.16997](https://arxiv.org/abs/2409.16997), [2412.08585](https://arxiv.org/abs/2412.08585) |
| Medusa / EAGLE / EAGLE-2 | [2401.10774](https://arxiv.org/abs/2401.10774), [2401.15077](https://arxiv.org/abs/2401.15077), [2406.16858](https://arxiv.org/abs/2406.16858) |
| AWQ / GPTQ primary format | contradicts GGUF native-first |
| Parallel T-MAC LUT stack | [2407.00088](https://arxiv.org/abs/2407.00088) — cross-read only |

## Related specs

- [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) — status tables + research priority phases.
- [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — native CUDA; FA deferred.
- [RUNNER_PROXY_SPEC.md](RUNNER_PROXY_SPEC.md) — multi-process isolation (Ollama-style).
- [PROFILING.md](PROFILING.md) — measure before large scheduler changes.
- [DELEGATION.md](DELEGATION.md) — dev-only notes for external backend experiments.
- [AKASHA_METRICS.md](AKASHA_METRICS.md) — correlating Rbitnet `/metrics` with Akasha daemon (3876).
- [FUTURE_DIFFERENTIATION.md](FUTURE_DIFFERENTIATION.md) — product axes (metrics UX, energy profiles, provenance).

## Implementation notes (in-repo)

| Phase | Delivered hooks |
|-------|-------------------|
| A.1 | [`llama/kv_storage.rs`](../crates/bitnet-core/src/llama/kv_storage.rs) — `KvStorage` / `PagedSeqKv`; enable with **`RBITNET_LLAMA_PAGED_KV`**. |
| A.2 | `PagedSeqKv::pool_stats()` — allocation vs reuse counters after `clear()`. |
| A.2b | `SharedPhysKvStore` + **`PagedKvPool`** — multi-sequence pool; **`RBITNET_KV_POOL=1`** wires Llama runtime KV to shared phys (E2E); free-list reclaim on `clear`/close; metrics `rbitnet_core_kv_pool_*`; bench [`scripts/bench_paged_kv.sh`](../scripts/bench_paged_kv.sh). |
| B.1 | [`scheduler.rs`](../crates/bitnet-core/src/scheduler.rs) — `run_batch` / `run_batch_waves` when `RBITNET_CONTINUOUS_BATCHING`. |
| B.2 | `PrefillDecodeQueue` + [`inference_session.rs`](../crates/bitnet-core/src/inference_session.rs). |
| C.1 | [`prefix_kv.rs`](../crates/bitnet-core/src/prefix_kv.rs) + [`prefix_kv_exec.rs`](../crates/bitnet-core/src/prefix_kv_exec.rs) — dense + paged snapshots (`RBITNET_PREFIX_KV`). |
| C.1b | Token streaming SSE — [`stream.rs`](../crates/bitnet-core/src/stream.rs), live path in `bitnet-server`. |
| C.2 | [`kv_sidecar.rs`](../crates/bitnet-core/src/kv_sidecar.rs) — HTTP PUT/GET; [KV_SIDECAR_SPEC.md](KV_SIDECAR_SPEC.md). |
| D.0 | [`cuda_graph.rs`](../crates/bitnet-core/src/llama/cuda_graph.rs) — decode graph + capture hook. |
| D.1 | [`llama/fusion.rs`](../crates/bitnet-core/src/llama/fusion.rs) — fusion counters (Sprint 5). |
| D | [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — native GPU work; [DELEGATION.md](DELEGATION.md) is dev-only. |
| E | Speculative MVP + BitNet CPU paths — see STATUS research section for next gates. |
