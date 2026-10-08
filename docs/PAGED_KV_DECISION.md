# Decision: Llama F32 paged device KV — shipped opt-in

**Status:** **closed for Llama native paged F32; other architectures deferred**  
**Issue:** [#92](https://github.com/azerothl/Rbitnet/issues/92)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-04-paged-kv/README.md](benchmarks/2026-10-04-paged-kv/README.md), PR [#155](https://github.com/azerothl/Rbitnet/pull/155), related encoded formats in [KV_CUDA_DECISION.md](KV_CUDA_DECISION.md)

## Verdict

Ship **Llama F32 paged device KV** (`RBITNET_CUDA_KV_PAGE_LIMIT` with physical 32-token pages) as **opt-in**, integrated with dense/split attention, block prefill, speculative verify, CUDA graphs, and prefix snapshots (COW on shared pages). Keep **dense F32** as default. Close **#92** for this Llama vertical; track Qwen/GPT/MLA paging, CPU/SSD tiers, and true multi-sequence serving under [#96](https://github.com/azerothl/Rbitnet/issues/96) and follow-ups.

## What shipped (acceptance slice)

| Criterion | Status |
|-----------|--------|
| Paged pool + persistent tables per context | Done |
| Bit-identical logits vs dense (fixtures + live protocol) | Done per README |
| Snapshot share/COW and cross-context refusal guards | Done |
| Prefix replay with paged + block prefill | Done (`paged-prefix` row) |
| Measured VRAM reduction on long corpus | Done (~48 MiB vs ~128 MiB category in ablation) |

## Explicitly out of this closure

| Topic | Reason |
|-------|--------|
| Default-on paging | Measured decode regression vs dense on same load (~8% in ablation) |
| Qwen / GPT / MLA device paging | Not validated as paged KV in #92 scope |
| F16/Q8 paged formats | [#93](KV_CUDA_DECISION.md) Llama-only |
| Continuous batching / multi-seq executor | [#96](https://github.com/azerothl/Rbitnet/issues/96) |

## Reopen criteria

Reopen to promote paging to **default** only if a frozen Llama suite shows net VRAM win **and** decode within 10% of dense F32 on the published long-prompt corpus, with HTTP/SSE parity unchanged.
