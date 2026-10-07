# Decision: MoE dynamic expert VRAM cache — shipped opt-in, not default

**Status:** **closed for P1 MVP; default remains off**  
**Issue:** [#83](https://github.com/azerothl/Rbitnet/issues/83)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-03-cache-foundation/README.md](benchmarks/2026-10-03-cache-foundation/README.md), [benchmarks/2026-10-03-moe-placement/README.md](benchmarks/2026-10-03-moe-placement/README.md), [benchmarks/2026-10-04-expert-arena-fresh/README.md](benchmarks/2026-10-04-expert-arena-fresh/README.md), related PRs #99/#109/#110/#114/#121/#128

## Verdict

The **dynamic expert VRAM cache** (pool, on-demand load, leases, LRU/LFU/Least-Stale, mapped weights, metrics, optional async) is **shipped and opt-in**. Keep **placement fixe / cache désactivé** as the default: published ablations show cases where a small or mid-size cache is **slower** than fixed banks (including GLM). Do not promote to default until a frozen budget sweep shows a net decode win on both GPT-OSS and GLM.

## What shipped (acceptance slice)

| Criterion | Status |
|-----------|--------|
| Slot pool + (model, layer, expert) table; gate/up/down groups | Done |
| Separate expert budget vs shared weights / KV / activations | Done (managed CUDA ceiling) |
| On-demand load + explicit CPU fallback | Done |
| Lease protection / unload without stale graph pointers | Done (arena + leases) |
| Hits/misses/bytes/evictions metrics | Done |
| Oracle / HTTP-SSE suites on opt-in budgets | Published; cache remains experimental |

## Remaining product gaps (not blocking this closure)

| Gap | Tracking |
|-----|----------|
| Compaction / predictor-trained residency | Research / follow-up |
| Mixed per-expert CPU+GPU (#86) | Still open |
| Default-on when always faster | Reopen when ablation proves it |
| Four-engine parity | #98 |

## Reopen criteria

Reopen to flip the **default** only after a same-revision GPT-OSS+GLM sweep shows median decode and TTFT wins (or clear capacity wins with acceptable slowdown) across cold/hot and ≥2 prompt families, with HTTP/SSE parity.
