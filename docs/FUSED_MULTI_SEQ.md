# Fused multi-seq CPU forward (#46) — stall decision

**Status:** CUDA resident Llama has an experimental end-to-end HTTP path; CPU,
BitNet, and Qwen remain sequential. This document retains the original CPU
stall decision and records the narrower #96 progress below.

**Issue:** [#46](https://github.com/azerothl/Rbitnet/issues/46) (slice of epic [#24](https://github.com/azerothl/Rbitnet/issues/24)).
**Spike PR:** [#67](https://github.com/azerothl/Rbitnet/pull/67) — `dense_matvec_multi_seq`, `ModelExecutor::generate_decode_batch`, opt-in `RBITNET_FUSED_MULTI_SEQ=1`.

## Acceptance for #46

| Criterion | Outcome |
|-----------|---------|
| Gain mesuré **ou** explication stallée à concurrency ≥4 | **Stall explained** (this doc) + kernel microbench numbers |
| Tests scheduler + golden non régressés | Scheduler fused-hook test remains; default path unchanged |

## What exists

1. **Scheduler** — with `RBITNET_CONTINUOUS_BATCHING=1` and `RBITNET_FUSED_MULTI_SEQ=1`, admitted prompt chunks call `generate_prefill_batch` and decode waves call `generate_decode_batch` once (`scheduler.rs`).
2. **Kernel** — `bitnet_core::fused_batch::dense_matvec_multi_seq` is a real weight-stationary batch matvec (N activation rows share one pass over `W`).
3. **Executors** — with CUDA resident Llama + `RBITNET_FUSED_MULTI_SEQ=1`, `LlamaExecutor` runs Native `llama_batch` shared prompt-token and decode waves (`scheduler_fused.rs`). CPU-only / BitNet / Qwen retain logical prefill admission and sequential generation.

## Why e2e concurrency ≥4 does not speed up (stall)

At concurrency 4–8, CPU and non-Llama executors can admit a decode wave of N
sequences, but each sequence still runs an independent native forward.
Enabling `RBITNET_FUSED_MULTI_SEQ` only changes the **scheduler call shape**
(one batch API call), not the FLOPs schedule inside their layers.

Wiring every layer (attention, MLP, residual, KV) to a batched activation tensor is a large invasive change to `llama/runtime.rs` (and BitNet/Qwen mirrors). That work is **deferred**:

- CPU vertical: optional later PR that implements true multi-seq decode inside `LlamaRuntime` using `dense_matvec_multi_seq` for dense GEMV sites.
- GPU fused waves: epic [#22](https://github.com/azerothl/Rbitnet/issues/22) (not this issue).

Until that lands, operators should treat `RBITNET_FUSED_MULTI_SEQ` as an **experimental hook** (CI / scheduler contract), **not** a throughput feature.

## Kernel microbench (measured building block)

```bash
# Criterion (release recommended)
cargo bench -p bitnet-core --bench kernels -- fused_multi_seq

# Or helper (prints fused vs sequential wall time @ batch 4/8)
./scripts/bench_fused_multi_seq.sh
```

Expected shape (illustrative; host-dependent): fused kernel faster than N independent matvecs on the same `W` once `batch ≥ 4` and `n,k` are large enough that weight traffic dominates. **This does not imply** HTTP concurrency ≥4 tok/s improvement today.

## Operator checklist

| Goal | What to set | Expectation |
|------|-------------|-------------|
| Stall-free schedule only | `RBITNET_CONTINUOUS_BATCHING=1` | Decode-first waves; no fused claim |
| Exercise fused hook in CI/tests | `+ RBITNET_FUSED_MULTI_SEQ=1` | Batch API called; CPU/non-Llama executors remain sequential |
| CUDA Llama HTTP batch | `RBITNET_CONTINUOUS_BATCHING=1` + `RBITNET_FUSED_MULTI_SEQ=1` | HTTP requests are coalesced briefly; Llama prompt-token and decode rows share GPU projections |

## #96 CUDA HTTP progress

For a CUDA-resident Llama with F32 KV, non-streaming `/v1/chat/completions`
requests are coalesced for a short server-side window and delivered as one
`Engine::complete_batch_detailed` call. The existing Sarathi scheduler then
uses `SchedulerFusedLlama` and `rbitnet_cuda_llama_batch_step` for multi-row
decode waves. This supersedes the older, separate
`RBITNET_CUDA_CONTINUOUS=1` worker for this configuration: leave that legacy
flag unset (or `0`).

Measure concurrency 1/4/8 with:

```bash
RBITNET_MODEL=/path/model.gguf RBITNET_TOKENIZER=/path/tokenizer.json \
./scripts/bench_http_sarathi_fused.sh
```

Publish the script's wall time, requested aggregate tok/s, and deltas for
`rbitnet_core_gpu_llama_batch_rows_total` and
`rbitnet_core_gpu_llama_batch_waves_total`. Do not publish a throughput claim
unless the rows delta exceeds the waves delta for the concurrent runs.

Remaining #96 gaps: streaming-request coalescing, CUDA graphs, adaptive
admission/backpressure, and validation on
representative GPU/model matrices. Qwen, GPT, MoE, and CPU paths are explicitly
outside this Llama-first slice.

## Docs sync

- [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md) — #46 marked stalled/closed with this decision
- [LIMITATIONS.md](LIMITATIONS.md) / [ENV_REFERENCE.md](ENV_REFERENCE.md) / [USAGE.md](USAGE.md)
- [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) — kernel row + e2e “non mesuré / stalled”
