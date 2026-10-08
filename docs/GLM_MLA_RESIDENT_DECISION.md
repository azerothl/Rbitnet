# Decision: GLM MLA CUDA resident backbone — shipped opt-in

**Status:** **closed with measured MLA/split/prefix path; partial expert placement remains**  
**Issue:** [#89](https://github.com/azerothl/Rbitnet/issues/89)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-03-mla-full/README.md](benchmarks/2026-10-03-mla-full/README.md), PRs [#108](https://github.com/azerothl/Rbitnet/pull/108), [#137](https://github.com/azerothl/Rbitnet/pull/137), [#154](https://github.com/azerothl/Rbitnet/pull/154)

## Verdict

**Compressed MLA attention**, split tile attention, device-resident backbone (router, shared FFN, head), and **prefix KV snapshots** for GLM under fixed or cached expert placement are **shipped and opt-in**. The model is **not** fully resident under typical caps: some routed layers still CPU-fallback. Keep MLA modes experimental; do not default-on until a same-revision sweep shows consistent wins over baseline placement.

## What shipped (acceptance slice)

| Criterion | Status |
|-----------|--------|
| MLA KV + attention on CUDA with exact outputs vs baseline | Done; ablation in `mla-full` |
| Split-KV composition for long contexts | Done; large prefill/decode improvements in corpus |
| Prefix snapshots (immutable KV + final-token recompute) | Done; warm-prefix medians in README |
| HTTP/SSE / streaming parity on optimized modes | Done for published 72+24 cases |
| Ordered CUDA normalization (exact order) | Done (#137); local decode/prefill gains |

## Remaining gaps (not blocking #89 closure)

| Gap | Reason |
|-----|--------|
| All routed layers GPU-resident under 16 GiB | Cap forces partial placement + CPU FFN fallback |
| Dynamic cache as default | Hit rate alone does not prove benefit (`mla-full` cache table) |
| Block prefill on MLA | Shipped separately under [#95](CUDA_BLOCK_PREFILL_DECISION.md) (#154) |
| Multi-sequence / continuous batching | [#96](https://github.com/azerothl/Rbitnet/issues/96) |

## Reopen criteria

Reopen for **default-on MLA** or **full GPU routing** only after a frozen GLM-4.7-Flash sweep at a declared device cap shows median decode and long-prefill wins (or acceptable trade-offs) vs baseline, with byte-identical responses on the published HTTP/SSE corpus and no increase in CPU fallback rate beyond documented limits.
