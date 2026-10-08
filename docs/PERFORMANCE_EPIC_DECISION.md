# Decision: performance epic #98 — coordinated caches and CUDA verticals closed (bounded)

**Status:** **ready to close after [#96](https://github.com/azerothl/Rbitnet/issues/96) lands via [PR #167](https://github.com/azerothl/Rbitnet/pull/167)**  
**Issue:** [#98](https://github.com/azerothl/Rbitnet/issues/98)  
**Date:** 2026-10-08  
**Evidence:** child decision docs below, [PERFORMANCE_CACHE_STACK.md](PERFORMANCE_CACHE_STACK.md), [benchmarks/](benchmarks/) corpus from PRs #99–#137 and follow-ups #154–#166

## Verdict

Close **#98** as a **coordination epic**, not as “parity with Ollama/llama.cpp on every model and path.” Each P1–P3 child shipped **opt-in** verticals with published ablations, explicit defaults unchanged, and decision memos. Residual serving work (**[#168](https://github.com/azerothl/Rbitnet/issues/168)–[#170](https://github.com/azerothl/Rbitnet/issues/170)**) is **post-#96 follow-up**, not unchecked items on the original #98 checklist.

## Child tickets (#83–#97)

| Issue | Title (short) | Closure style | Decision / evidence |
|------:|---------------|---------------|---------------------|
| [#83](https://github.com/azerothl/Rbitnet/issues/83) | MoE dynamic expert VRAM cache | Shipped opt-in; default off | [MOE_EXPERT_CACHE_DECISION.md](MOE_EXPERT_CACHE_DECISION.md) |
| [#84](https://github.com/azerothl/Rbitnet/issues/84) | Async expert prefetch | Shipped baseline; trained Mira no-go | [MOE_PREFETCH_DECISION.md](MOE_PREFETCH_DECISION.md) |
| [#85](https://github.com/azerothl/Rbitnet/issues/85) | LRU/LFU/Least-Stale | Shipped; LRU default | [EXPERT_POLICY_DECISION.md](EXPERT_POLICY_DECISION.md) |
| [#86](https://github.com/azerothl/Rbitnet/issues/86) | Hybrid CPU/GPU MoE | Whole-FFN policies only; no per-expert splitter | [MOE_HYBRID_DECISION.md](MOE_HYBRID_DECISION.md) |
| [#87](https://github.com/azerothl/Rbitnet/issues/87) | Qwen3.5 CUDA full resident | Shipped opt-in pipeline | [benchmarks/2026-10-03-qwen-full/README.md](benchmarks/2026-10-03-qwen-full/README.md), PR [#101](https://github.com/azerothl/Rbitnet/pull/101) |
| [#88](https://github.com/azerothl/Rbitnet/issues/88) | GPT-OSS CUDA resident | Shipped; partial/segmented when banks do not fit | [GPT_OSS_RESIDENT_DECISION.md](GPT_OSS_RESIDENT_DECISION.md) |
| [#89](https://github.com/azerothl/Rbitnet/issues/89) | GLM MLA CUDA resident | Shipped; partial expert placement remains | [GLM_MLA_RESIDENT_DECISION.md](GLM_MLA_RESIDENT_DECISION.md) |
| [#90](https://github.com/azerothl/Rbitnet/issues/90) | Llama prefix KV + CUDA graph | Shipped opt-in snapshots | [PERFORMANCE_CACHE_STACK.md](PERFORMANCE_CACHE_STACK.md), PR [#99](https://github.com/azerothl/Rbitnet/pull/99) / [#103](https://github.com/azerothl/Rbitnet/pull/103) |
| [#91](https://github.com/azerothl/Rbitnet/issues/91) | Qwen prefix + GDN/conv | Shipped checkpoint-exact reuse | [PERFORMANCE_CACHE_STACK.md](PERFORMANCE_CACHE_STACK.md), PR [#101](https://github.com/azerothl/Rbitnet/pull/101) |
| [#92](https://github.com/azerothl/Rbitnet/issues/92) | CUDA paged KV | Llama F32 paged opt-in | [PAGED_KV_DECISION.md](PAGED_KV_DECISION.md) |
| [#93](https://github.com/azerothl/Rbitnet/issues/93) | CUDA KV F16/Q8 + KIVI | F16/Q8 shipped; KIVI no-go | [KV_CUDA_DECISION.md](KV_CUDA_DECISION.md), [KIVI_DECISION.md](KIVI_DECISION.md) |
| [#94](https://github.com/azerothl/Rbitnet/issues/94) | Session tiers VRAM/RAM/SSD | Llama/Qwen bounded tiers only | [SESSION_TIERS_DECISION.md](SESSION_TIERS_DECISION.md), PR [#166](https://github.com/azerothl/Rbitnet/pull/166) |
| [#95](https://github.com/azerothl/Rbitnet/issues/95) | CUDA block prefill | Per-arch verticals shipped opt-in | [CUDA_BLOCK_PREFILL_DECISION.md](CUDA_BLOCK_PREFILL_DECISION.md) |
| [#96](https://github.com/azerothl/Rbitnet/issues/96) | Fused multi-seq serving | **Bounded CUDA Llama Sarathi** (merge #167) | [FUSED_MULTI_SEQ_DECISION.md](FUSED_MULTI_SEQ_DECISION.md) (PR #167) |
| [#97](https://github.com/azerothl/Rbitnet/issues/97) | Speculative decode | Llama PLD opt-in; default no-go | [SPECULATIVE_DECISION.md](SPECULATIVE_DECISION.md) |

## Explicitly not claimed by closing #98

| Topic | Tracking |
|-------|----------|
| General CPU/GPU parity vs Ollama/llama.cpp on all four reference GGUF | Frozen rows in [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md); gaps documented per model |
| Fused multi-seq / live SSE mux / cross-arch batching beyond Llama CUDA | [#168](https://github.com/azerothl/Rbitnet/issues/168)–[#170](https://github.com/azerothl/Rbitnet/issues/170) |
| FlashAttention-class default path, packed BitNet GPU, ROCm/Vulkan/Metal GGUF | [#22](https://github.com/azerothl/Rbitnet/issues/22) |
| Qwen/GPT/MLA **device** paged KV beyond Llama F32 paging | Follow-ups under [#92](PAGED_KV_DECISION.md) / [#93](KV_CUDA_DECISION.md) “out of closure” tables |
| Stub/MVP inventory and non-CPU backend honesty | [#24](https://github.com/azerothl/Rbitnet/issues/24) |

## Common validation (#98 issue body)

| Criterion | Honest status at epic close |
|-----------|----------------------------|
| Four-model Rbitnet/Ollama/llama.cpp comparison | Published ([parity-round2](benchmarks/2026-10-03-parity-round2/README.md)); **no general parity claim** |
| Per-optimization ablations on same revision | Extensive under `docs/benchmarks/2026-10-*`; negatives published |
| Short/long prompts, 1/4/8 concurrency where supported | Llama fused probe in [HTTP_SARATHI_FUSED.md](benchmarks/2026-10-08-cuda-fused-scheduler/HTTP_SARATHI_FUSED.md) (#167); other paths document serial execution |
| Quality / long-context gates for lossy experiments | KIVI and default-on promotions **no-go** per decision docs |
| Limits, flags, metrics updated with real execution | [ENV_REFERENCE.md](ENV_REFERENCE.md), [PERFORMANCE_CACHE_STACK.md](PERFORMANCE_CACHE_STACK.md), `/metrics` series |

## Reopen criteria

Reopen **#98** only to add a **new coordinated tranche** (new baseline after a major serving/kernel generation), not for individual follow-ups already ticketed as #22, #24, or #168–#170.
