# Decision: close the bounded Llama/admission slice of #170

**Status:** accepted; closes [#170](https://github.com/azerothl/Rbitnet/issues/170)  
**Date:** 2026-10-08  
**Evidence:** [adaptive-admission implementation](https://github.com/azerothl/Rbitnet/pull/176), [Llama live-mux contract](CONTINUOUS_LLAMA.md), [fused Llama decision](FUSED_MULTI_SEQ_DECISION.md)

## Verdict

Ship the Llama-only adaptive-admission capability as an explicit experimental
opt-in:

- `RBITNET_CUDA_CONTINUOUS_ADMISSION=adaptive` reserves active decode rows and
  the next prefill partition before accepting another request-local KV owner;
- it applies only to the CUDA-resident Llama continuous worker, including its
  live-SSE-mux variant;
- FIFO remains the default policy;
- the implementation has no throughput claim until its CUDA benchmark gate is
  published.

This is a bounded close, not a claim that #170's full multi-architecture scope
is implemented. The Llama fused decode vertical and adaptive admission now have
defined behavior and an opt-in boundary. The remaining architecture work is
tracked independently below.

## What is not shipped

No Qwen, GPT-OSS, GLM, or MoE executor has a shared multi-owner CUDA forward in
this release:

- Qwen GDN/recurrent state does not yet have fused prefill/decode with
  request-local state isolation;
- GPT-OSS / GLM MoE do not yet pack per-token expert routing for a shared
  decode wave;
- CUDA graphs are not enabled for multi-sequence waves;
- the adaptive policy has not yet passed a published CUDA benchmark gate.

Operators must not interpret `RBITNET_FUSED_MULTI_SEQ=1` or the Llama adaptive
flag as support for these architectures.

## Deferred follow-up decision

The deferred Qwen, MoE, adaptive-admission, and CUDA-graph work is resolved by
the explicit no-go and reopen criteria in
[the follow-up decision](FUSED_MULTI_FOLLOWUP_DECISIONS.md), which closes
[#178](https://github.com/azerothl/Rbitnet/issues/178),
[#179](https://github.com/azerothl/Rbitnet/issues/179), and
[#180](https://github.com/azerothl/Rbitnet/issues/180). No architecture is
considered shipped without its listed correctness oracle and benchmark evidence.

## Reopen threshold

Reopen #170 only if the documented Llama adaptive-admission boundary regresses,
or if a follow-up cannot be evaluated independently of this closed slice.
