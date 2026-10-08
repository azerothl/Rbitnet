# Decision: GPT-OSS CUDA resident pipeline — shipped opt-in

**Status:** **closed with measured resident path; default remains partial/serial where full cannot load**  
**Issue:** [#88](https://github.com/azerothl/Rbitnet/issues/88)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-03-gpt-full/README.md](benchmarks/2026-10-03-gpt-full/README.md), PRs [#106](https://github.com/azerothl/Rbitnet/pull/106), [#109](https://github.com/azerothl/Rbitnet/pull/109), [#111](https://github.com/azerothl/Rbitnet/pull/111), [#124](https://github.com/azerothl/Rbitnet/pull/124), [#128](https://github.com/azerothl/Rbitnet/pull/128), [#150](https://github.com/azerothl/Rbitnet/pull/150)

## Verdict

The **GPT-OSS CUDA resident** path (attention, router, norms, Q/K/V, RoPE, GQA, sinks/windows, split-KV, fixed expert banks on device) is **shipped and opt-in** via `RBITNET_CUDA_GPT_FULL=1`. Keep the legacy partial pipeline as fallback when full resident cannot be created or when dynamic expert cache is enabled. Do not claim reference-engine parity or default-on until published cross-engine ablations meet the issue gates.

## What shipped (acceptance slice)

| Criterion | Status |
|-----------|--------|
| End-to-end token on GPU with fixed resident banks | Done (`RBITNET_CUDA_GPT_FULL`, optional graphs) |
| Ordered router / OAI FFN / sinks / alternating windows | Done; CPU SIMD reference for tie-breaking |
| Split-KV attention on long prompts | Done; measured prefill/decode gains in `gpt-full` ablation |
| HTTP/SSE parity vs partial path on same revision | Done for published corpus (greedy, seed, penalties, stops) |
| Explicit refusal when full path unavailable | Done (`RBITNET_REQUIRE_GPT_FULL=1`) |

## Remaining gaps (not blocking #88 closure)

| Gap | Tracking |
|-----|----------|
| Dynamic MoE VRAM cache + full resident | Incompatible; cache keeps partial path |
| Reference-engine tok/s parity (Ollama / llama.cpp) | Documented gap in benchmark README |
| Segmented/cache partial placement resident parity | Block prefill for non-fixed segments → [#95](CUDA_BLOCK_PREFILL_DECISION.md) |
| Default-on full resident | Reopen when VRAM + correctness gates pass on representative loads |

## Reopen criteria

Reopen to flip **default** or to require full resident under cache only after a frozen GPT-OSS sweep shows net decode/TTFT wins (or clear transfer wins with acceptable slowdown) vs partial, with HTTP/SSE parity and no router-order regressions on the published real-model fixtures.
