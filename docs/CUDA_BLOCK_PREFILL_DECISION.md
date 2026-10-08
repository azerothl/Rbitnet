# Decision: CUDA block prefill verticals — shipped opt-in

**Status:** **closed for declared architecture verticals; corpus and serial MoE remnants deferred**  
**Issue:** [#95](https://github.com/azerothl/Rbitnet/issues/95)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-03-qwen-block-prefill/README.md](benchmarks/2026-10-03-qwen-block-prefill/README.md), [benchmarks/2026-10-04-gpt-block/README.md](benchmarks/2026-10-04-gpt-block/README.md), [benchmarks/2026-10-08-gpt-segmented-block-prefill/README.md](benchmarks/2026-10-08-gpt-segmented-block-prefill/README.md), [benchmarks/2026-10-08-mla-block-prefill/README.md](benchmarks/2026-10-08-mla-block-prefill/README.md), PRs [#99](https://github.com/azerothl/Rbitnet/pull/99), [#104](https://github.com/azerothl/Rbitnet/pull/104), [#105](https://github.com/azerothl/Rbitnet/pull/105), [#111](https://github.com/azerothl/Rbitnet/pull/111), [#154](https://github.com/azerothl/Rbitnet/pull/154), [#156](https://github.com/azerothl/Rbitnet/pull/156)

## Verdict

**CUDA block prefill** is **shipped and opt-in** for:

| Vertical | Env (representative) | Notes |
|----------|-------------------|--------|
| **Llama** dense resident | `RBITNET_CUDA_PREFILL=1` (+ optional `RBITNET_CUDA_PREFILL_TF32X3=1`) | SIMT/Tensor Core projections, causal attention, KV update per block |
| **Qwen3.5 dense** full GPU | `RBITNET_CUDA_QWEN_PREFILL=1` | GDN state retained per block; checkpoints at block boundaries |
| **GPT-OSS fixed banks** | `RBITNET_CUDA_GPT_PREFILL=1` | Ordered projections + grouped expert FFN in block |
| **GPT-OSS segmented/cache/partial** | `RBITNET_CUDA_GPT_PREFILL=1` on segmented resident path | Matmul attention/router per block; **host MoE admission unchanged** (#156) |
| **GLM MLA fixed banks** | `RBITNET_CUDA_MLA_PREFILL=1` | Causal MLA block forward (#154) |

All remain **experimental opt-in**; serial token loops stay the default when flags are off or workspaces refuse allocation.

## What shipped (acceptance slice)

| Criterion | Status |
|-----------|--------|
| Multi-token forward vs serial: logits/KV/recurrent state | Done per-arch native tests + published GPT/Qwen ablations |
| Block sizes 1, non-multiples, context boundaries | Covered in GPT block production validation |
| Prefix checkpoints interact correctly with blocks | Done (GPT/Qwen prefix rows; Llama stack) |
| HTTP/SSE on paths with published ablations | Done for GPT fixed-bank production corpus |
| Proof of matmul sharing (not wrapped serial loop) | Counters / Nsight grids in GPT block profile |

## Remaining gaps (follow-up; not blocking #95 close)

| Gap | Reason |
|-----|--------|
| **Published HTTP/SSE tok/s corpus** for segmented GPT + MLA block modes | README notes + unit/oracle proof; full network ablation not yet frozen |
| **Serial segments** when dynamic/cache/partial MoE admission stays on host | By design for segmented GPT; prefill wins partial |
| Default-on block prefill | Requires per-family default promotion gates |
| Speculative multi-logit consumers beyond Llama PLD | [#97](SPECULATIVE_DECISION.md) scope |

## Reopen criteria

Reopen for **default-on** block prefill per architecture only after a same-revision HTTP/SSE ablation (`scripts/benchmark_cache_stack.py` or successor) shows median long-prefill and TTFT wins without >5% decode regression on two prompt families, with parity on greedy/seed/penalties/stops.
