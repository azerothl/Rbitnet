# Decision: hybrid per-expert MoE placement — measured no-go; retain bounded opt-in policies

**Status:** closed — no general per-expert CPU/GPU policy shipped  
**Issue:** [#86](https://github.com/azerothl/Rbitnet/issues/86)  
**Date:** 2026-10-08  
**Evidence:** [whole-FFN placement](benchmarks/2026-10-03-moe-placement/README.md), [asynchronous cache](benchmarks/2026-10-04-async-expert-cache/README.md), [expert-policy ablation](benchmarks/2026-10-04-expert-policies/README.md), [cache-stack contract](PERFORMANCE_CACHE_STACK.md)

## Verdict

Do **not** add an always-on per-selected-expert CPU/GPU splitter.  The
implemented opt-in controls are the bounded alternatives:

- `RBITNET_MOE_EXECUTION=cache` keeps fixed/cache GPU execution (the default).
- `RBITNET_MOE_EXECUTION=cpu` computes the selected routed FFN on SIMD CPU
  without allocating unused GPU expert banks.
- `RBITNET_MOE_EXECUTION=adaptive` calibrates CPU, GPU, and actual missing-group
  transfer cost per layer, then chooses the complete selected routed FFN.
- `RBITNET_MOE_ASYNC=1` is a separate, bounded `cache`-only experiment with
  pinned H2D slots and CUDA-event ownership; `RBITNET_MOE_PREFETCH=previous-pass`
  remains opt-in.

These policies preserve router IDs, selected-expert order, coefficients, and
the existing ordered aggregation.  They also avoid nested CPU scheduling:
the CPU fallback uses the established quantized SIMD path and the async GPU
copy path has its own bounded slots rather than creating Rayon work for each
transfer.

## Cost model and observability retained

`adaptive` owns a cost estimator for each model layer, so observations from
different shapes or quantizations are not combined.  It calibrates CPU routed
FFN wall time, GPU work excluding measured admission time, and PCIe cost per
actual transferred byte.  It re-probes both paths every 128 decisions; an
unfit selected set uses CPU.  Its Prometheus metrics expose CPU/GPU decisions,
GPU and fallback FFN latencies, bytes and duration of uploads, capacity
refusals, selected-expert waits, DMA/staging time, and async pool/pinned-RAM
state.  The async report separately records observed copy/compute overlap.

`cache` and `cpu` are the fixed ablation policies.  They make comparisons
reproducible without allowing an adaptive estimator to hide a regression.

## Why this is a no-go

The same-revision GPT-OSS-20B and GLM-4.7-Flash placement sweep does not show
a robust win for moving only some selected experts:

| Case | Adaptive / cache decode ratio |
|---|---:|
| GPT-OSS, 0 MiB expert cache | 0.963 |
| GPT-OSS, 8192 MiB expert cache | 0.666 |
| GLM, 0 MiB expert cache | 0.988 |
| GLM, 8192 MiB expert cache | 0.746 |

The explicit CPU path is much slower in that sweep (GPT-OSS 13.90 versus
81.22 tok/s at 0 MiB; GLM 10.87 versus 21.51 tok/s).  The asynchronous cache
is similarly model- and budget-dependent: it improves the measured GLM
8192-MiB case, but regresses GPT-OSS.  Its Nsight capture found only 10.90%
copy/compute overlap for that capture, not a general throughput result.

A per-expert splitter adds synchronizations, partial-result aggregation, and
contention for the same CPU/GPU resources.  No published same-revision
measurement establishes that its extra overhead is bounded by the best
single-path policy.  Claiming such a win from hit rate, theoretical PCIe
bandwidth, or one overlap trace would not meet the issue's validation bar.

## Validation and follow-up boundary

The retained policies were validated with real GGUF GPT-OSS/GLM logits,
generations, HTTP/SSE, seeded sampling, stops, cancellation/reconnect, and
serialized concurrent requests.  The cited reports retain raw timings,
RAM/VRAM, transfer counters, warm-up cycles, and output checks.  This closes
the requested product decision: the measured hybrid mechanism is available
only as bounded whole-FFN/admission experiments, while the unproven
per-expert splitter is deliberately not shipped.

Reopen only with a frozen, same-build GPT-OSS+GLM experiment that implements
ordered per-expert aggregation and shows:

1. measured CPU, GPU, wait, and PCIe-throughput telemetry by shape/format;
2. controlled CPU-copy-GPU overlap without Rayon oversubscription;
3. numerical/oracle and HTTP/SSE parity; and
4. hybrid latency no worse than the best fixed policy by a published,
   predeclared margin across warm/cold and multiple budgets.
