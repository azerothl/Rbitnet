# Fused multi-seq CPU forward (#46) — stall decision

**Status:** **Stalled (documented)** for end-to-end concurrency gain. Kernel building block shipped; Llama/BitNet/Qwen full forwards remain per-seq.

**Issue:** [#46](https://github.com/azerothl/Rbitnet/issues/46) (slice of epic [#24](https://github.com/azerothl/Rbitnet/issues/24)).
**Spike PR:** [#67](https://github.com/azerothl/Rbitnet/pull/67) — `dense_matvec_multi_seq`, `ModelExecutor::generate_decode_batch`, opt-in `RBITNET_FUSED_MULTI_SEQ=1`.

## Acceptance for #46

| Criterion | Outcome |
|-----------|---------|
| Gain mesuré **ou** explication stallée à concurrency ≥4 | **Stall explained** (this doc) + kernel microbench numbers |
| Tests scheduler + golden non régressés | Scheduler fused-hook test remains; default path unchanged |

## What exists

1. **Scheduler** — with `RBITNET_CONTINUOUS_BATCHING=1` and `RBITNET_FUSED_MULTI_SEQ=1`, decode waves call `generate_decode_batch` once (`scheduler.rs` / `decode_wave_fused`).
2. **Kernel** — `bitnet_core::fused_batch::dense_matvec_multi_seq` is a real weight-stationary batch matvec (N activation rows share one pass over `W`).
3. **Executors** — default `generate_decode_batch` falls back to sequential `generate_with_timings` per item (`model/executor.rs`). Llama / BitNet / Qwen do **not** yet run a shared multi-seq forward.

## Why e2e concurrency ≥4 does not speed up (stall)

At concurrency 4–8, Sarathi can admit a decode wave of N sequences, but each sequence still runs an independent native forward. Enabling `RBITNET_FUSED_MULTI_SEQ` only changes the **scheduler call shape** (one batch API call), not the FLOPs schedule inside Llama layers.

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
| Exercise fused hook in CI/tests | `+ RBITNET_FUSED_MULTI_SEQ=1` | Batch API called; still sequential executors |
| Real multi-seq throughput | Wait for Llama batched forward **or** #22 GPU | Not shipped |

## Docs sync

- [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md) — #46 marked stalled/closed with this decision
- [LIMITATIONS.md](LIMITATIONS.md) / [ENV_REFERENCE.md](ENV_REFERENCE.md) / [USAGE.md](USAGE.md)
- [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) — kernel row + e2e “non mesuré / stalled”
