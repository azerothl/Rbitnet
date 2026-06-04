# Inference stack v2 — backlog (epic)

This document splits the **long-term** items from [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) (*Advanced inference stack gaps*) into phases. It is **not** part of the short “prod-ready” checklist in [PLAN_PRODUCTION.md](PLAN_PRODUCTION.md); ship cadence is independent.

## Goals

- Raise throughput and memory efficiency toward industry stacks (vLLM-class patterns) without breaking the existing **single-process** OpenAI API.

## Phase A — KV and memory

1. **Block-structured KV** (step toward PagedAttention): fixed token pages, block tables per sequence, predictable VRAM usage for long context.
2. **Eviction / pooling** across requests only after block KV exists (today KV is dense per request).

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

## Implementation notes (in-repo)

| Phase | Delivered hooks |
|-------|-------------------|
| A.1 | [`llama/kv_storage.rs`](../crates/bitnet-core/src/llama/kv_storage.rs) — `KvStorage` / `PagedSeqKv`; enable with **`RBITNET_LLAMA_PAGED_KV`**. |
| A.2 | `PagedSeqKv::pool_stats()` — allocation vs reuse counters after `clear()`. |
| B.1 | [`scheduler.rs`](../crates/bitnet-core/src/scheduler.rs) — `run_batch` logs when `RBITNET_CONTINUOUS_BATCHING` packs multiple requests (still sequential forward until batched matmul lands). |
| B.2 | `PrefillDecodeQueue` placeholder struct (same module). |
| C.1 | [`prefix_kv.rs`](../crates/bitnet-core/src/prefix_kv.rs) + [`prefix_kv_exec.rs`](../crates/bitnet-core/src/prefix_kv_exec.rs) — radix scaffold + dense KV snapshot reuse (`RBITNET_PREFIX_KV`). |
| C.1b | Token streaming SSE — [`stream.rs`](../crates/bitnet-core/src/stream.rs), live path in `bitnet-server`. |
| A.2b | [`PagedKvPool`](../crates/bitnet-core/src/llama/kv_storage.rs) — multi-sequence paged KV API. |
| D.0 | [`cuda_graph.rs`](../crates/bitnet-core/src/llama/cuda_graph.rs) — decode graph scaffold + metrics. |
| C.2 | [DEPLOYMENT.md](DEPLOYMENT.md) — replica / hash routing sketch. |
| D | [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — native GPU work; [DELEGATION.md](DELEGATION.md) is dev-only. |
