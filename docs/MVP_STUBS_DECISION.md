# Decision: stubs / MVP audit (#24) — inventory complete; residual work ticketed

**Status:** **ready to close when docs reflect #96 boundary (PR #167) and this mapping**  
**Issue:** [#24](https://github.com/azerothl/Rbitnet/issues/24)  
**Date:** 2026-10-08  
**Evidence:** [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md), [LIMITATIONS.md](LIMITATIONS.md)

## Verdict

**#24** is a **process** ticket: every path that docs, HTTP, or CLI **present as supported** must be real execution (or explicit test hooks `STUB`/`TOY`). Everything else must appear as **non-supported** in [LIMITATIONS.md](LIMITATIONS.md) or an open issue. That bar is met after the #83–#97 tranche and the bounded **#96** Llama CUDA fused vertical; remaining gaps are **not** hidden stubs.

**#22** remains the umbrella for **additional GPU backends and native kernel depth** (ROCm quant, Vulkan/Metal GGUF, FA-class research). **#24 does not duplicate #22** — non-CPU probes that refuse GGUF load are documented limitations, not silent MVP.

## Checklist mapping (issue #24 body)

| #24 item | Outcome |
|----------|---------|
| Exhaustive audit | [STUBS_AND_MVP_AUDIT.md](STUBS_AND_MVP_AUDIT.md) (dated; code/doc grep recurred on major merges) |
| Non-CPU backends | CUDA/hybrid **measured** on supported arch; ROCm/Vulkan/Metal **explicit load failure or prototype** → [#22](https://github.com/azerothl/Rbitnet/issues/22) |
| Prefix-KV (tensorial) | **Shipped opt-in** — not to be confused with `RBITNET_PREFIX_CACHE` |
| Fused continuous batching | **Bounded:** CUDA Llama Sarathi (`RBITNET_FUSED_MULTI_SEQ` + resident path) per #96/#167; Qwen/GPT/MoE → [#170](https://github.com/azerothl/Rbitnet/issues/170); ablations → [#168](https://github.com/azerothl/Rbitnet/issues/168); live SSE mux → [#169](https://github.com/azerothl/Rbitnet/issues/169) |
| Speculative beyond scheduler MVP | **Llama PLD + verify** opt-in ([#97](SPECULATIVE_DECISION.md)); no default win |
| Roadmap loaders (`gptoss`, `deepseek2`, …) | Real native paths where tensors validate; `glm4moe` Llama-shaped only — [LIMITATIONS.md](LIMITATIONS.md) architecture table |
| `tokenizer.model` without manual conversion | **Supported** — [benchmarks/2026-10-04-context-tokenizer/README.md](benchmarks/2026-10-04-context-tokenizer/README.md) |
| Tests + LIMITATIONS/STATUS per conversion | CI goldens/e2e for shipped paths; negative ablations published |

## Residual rows (not stubs; tracked or non-supported)

| Audit area | Resolution |
|------------|------------|
| ROCm / Vulkan / Metal / Intel GGUF | Non-supported for inference; [#22](https://github.com/azerothl/Rbitnet/issues/22) |
| Qwen3.5 MoE GGUF | Unvalidated — load/HTTP error or CPU-only until proven |
| BitNet `bitnet_cuda_matvec_mvp` | CPU execution; no validated TQ CUDA |
| Structured JSON / tools generation | HTTP **501** before SSE — [STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md) |
| Vision beyond LLaVA mmproj | **501** without projector; [#143](https://github.com/azerothl/Rbitnet/issues/143) |
| Spark-X2.5 CUDA / 1M ctx | CPU MVP; [#142](https://github.com/azerothl/Rbitnet/issues/142) |
| Process-wide VRAM gauge | Not shipped; managed CUDA category gauges only |
| GPT/GLM session transport, proxy-session persistence | Out of product — [SESSION_TIERS_DECISION.md](SESSION_TIERS_DECISION.md) |

## Reopen criteria

Reopen **#24** if a **new** user-facing path ships without audit row, test, or LIMITATIONS entry, or if a previously documented limitation starts accepting traffic without measured execution.
