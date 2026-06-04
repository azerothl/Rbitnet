# Inference stack v2 — backlog (epic)

This document splits the **long-term** items from [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) (*Advanced inference stack gaps*) into phases. It is **not** part of the short “prod-ready” checklist in [PLAN_PRODUCTION.md](PLAN_PRODUCTION.md); ship cadence is independent.

## Goals

- Raise throughput and memory efficiency toward industry stacks (vLLM-class patterns) without breaking the existing **single-process** OpenAI API.

## Phase summary (Done / Next)

| Phase | Done (hooks / MVP) | Next (production) |
|-------|-------------------|-------------------|
| **A — KV and memory** | `KvStorage` / `PagedSeqKv` (`RBITNET_LLAMA_PAGED_KV`); `PagedKvPool` + `RBITNET_KV_POOL` on `Engine`; shared physical page allocator | Fused paged attention; GPU-resident pages end-to-end |
| **B — Scheduling** | `run_batch` / `run_batch_waves`; `PrefillDecodeQueue`; `RBITNET_CONTINUOUS_BATCHING` interleaved decode; chunked prefill env | Single forward wave for N sequences (batched matmul) |
| **C — Cache semantics** | Dense prefix KV (`RBITNET_PREFIX_KV`); paged snapshots; radix `longest_token_prefix`; HTTP sidecar PUT/GET | Sidecar warm path in multi-replica deploys; L7 sticky hardening |
| **C — Streaming** | Live SSE (`StreamEvent`, `complete_streaming`) | GGUF load tests under sustained concurrency |
| **D — GPU decode** | `CudaDecodeGraph` metrics + capture hook; `RBITNET_KV_BACKEND=gpu` planning bit; cuBLASLt M=1 policy env | Full graph replay on device; fused norm+quant decode kernels |

## Phase A — KV and memory

1. **Block-structured KV** (step toward PagedAttention): fixed token pages, block tables per sequence, predictable VRAM usage for long context.
2. **Eviction / pooling** across requests only after block KV exists (today KV is dense per request unless pool enabled).

## Phase B — Scheduling

1. **Continuous batching** (optional mode): pack compatible requests into forward waves.
2. **Chunked prefill** coordinated with decode (beyond fixed `RBITNET_PREFILL_CHUNK_TOKENS` slice loops).

## Phase C — Cache semantics

1. **Prefix KV reuse** for shared prompt prefixes (distinct from `RBITNET_PREFIX_CACHE` full-response cache).
2. **L7 / sticky routing** notes for multi-replica setups — align with [LIMITATIONS.md](LIMITATIONS.md).

## Phase D — Kernel fusion (GPU)

1. Fused attention / GEMV paths for supported GGUF layouts where residency allows.
2. Compare against external high-performance backends as benchmarks only; in-tree delegation remains dev-only behind `experimental-external-backends`.

## Related specs

- [RUNNER_PROXY_SPEC.md](RUNNER_PROXY_SPEC.md) — multi-process isolation (Ollama-style).
- [PROFILING.md](PROFILING.md) — measure before large scheduler changes.
- [DELEGATION.md](DELEGATION.md) — dev-only notes for external backend experiments.
- [AKASHA_METRICS.md](AKASHA_METRICS.md) — correlating Rbitnet `/metrics` with Akasha daemon (3876).

## Implementation notes (in-repo)

| Phase | Delivered hooks |
|-------|-------------------|
| A.1 | [`llama/kv_storage.rs`](../crates/bitnet-core/src/llama/kv_storage.rs) — `KvStorage` / `PagedSeqKv`; enable with **`RBITNET_LLAMA_PAGED_KV`**. |
| A.2 | `PagedSeqKv::pool_stats()` — allocation vs reuse counters after `clear()`. |
| A.2b | `SharedPhysKvAllocator` + **`PagedKvPool`** — multi-sequence pool; **`RBITNET_KV_POOL=1`** on `Engine`. |
| B.1 | [`scheduler.rs`](../crates/bitnet-core/src/scheduler.rs) — `run_batch` / `run_batch_waves` when `RBITNET_CONTINUOUS_BATCHING`. |
| B.2 | `PrefillDecodeQueue` + [`inference_session.rs`](../crates/bitnet-core/src/inference_session.rs). |
| C.1 | [`prefix_kv.rs`](../crates/bitnet-core/src/prefix_kv.rs) + [`prefix_kv_exec.rs`](../crates/bitnet-core/src/prefix_kv_exec.rs) — dense + paged snapshots (`RBITNET_PREFIX_KV`). |
| C.1b | Token streaming SSE — [`stream.rs`](../crates/bitnet-core/src/stream.rs), live path in `bitnet-server`. |
| C.2 | [`kv_sidecar.rs`](../crates/bitnet-core/src/kv_sidecar.rs) — HTTP PUT/GET; [KV_SIDECAR_SPEC.md](KV_SIDECAR_SPEC.md). |
| D.0 | [`cuda_graph.rs`](../crates/bitnet-core/src/llama/cuda_graph.rs) — decode graph + capture hook. |
| D.1 | [`llama/fusion.rs`](../crates/bitnet-core/src/llama/fusion.rs) — fusion counters (Sprint 5). |
| D | [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — native GPU work; [DELEGATION.md](DELEGATION.md) is dev-only. |
