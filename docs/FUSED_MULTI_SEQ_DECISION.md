# Decision: ship the bounded CUDA Llama Sarathi fused vertical

**Status:** accepted; closes [#96](https://github.com/azerothl/Rbitnet/issues/96)  
**Date:** 2026-10-08  
**Evidence:** [synchronized HTTP 1/4/8 probe](benchmarks/2026-10-08-cuda-fused-scheduler/HTTP_SARATHI_FUSED.md), [scheduler implementation note](benchmarks/2026-10-08-cuda-fused-scheduler/README.md), [operator contract](FUSED_MULTI_SEQ.md)

## Verdict

Ship the following narrow vertical for compatible OpenAI-chat Llama requests:

- `RBITNET_CONTINUOUS_BATCHING=1` and `RBITNET_FUSED_MULTI_SEQ=1` admit
  compatible HTTP requests into Sarathi prefill/decode waves;
- CUDA-resident Llama uses one native multi-row projection path while retaining
  request-local KV state, positions, sampling, and terminal outputs;
- native F32 paged KV is required in a constrained-VRAM deployment. Operators
  must set a suitable `RBITNET_CUDA_KV_PAGE_LIMIT`; dense model-maximum KV can
  refuse a multi-request wave on a 16 GiB GPU;
- `RBITNET_CUDA_CONTINUOUS` is a distinct legacy worker and remains unset (or
  `0`) for this path;
- OpenAI-compatible SSE requests can enter the same rendezvous; the current
  bridge buffers a completed shared result before emitting content deltas.

The measured 4- and 8-request HTTP probes prove fused work was reached: GPU
batch rows exceed GPU batch waves. This supersedes the older statement that the
Llama GPU path was only a scheduler hook.

## Product boundary

This decision deliberately ships one observable, opt-in architecture/path:
CUDA Llama with F32 native KV, bounded rows, and the OpenAI-compatible chat
transport. It does not claim that all native executors, all transport shapes,
or all KV configurations share a forward.

CPU Llama/BitNet and Qwen retain their existing sequential execution. GPT and
MoE paths do not enter this vertical. Unsupported combinations must remain
explicit rather than silently claiming fused throughput.

## Follow-ups (explicit, non-blocking for #96)

Broader acceptance items from the original issue body are deferred to tracked
follow-ups rather than implied by this Llama shipment:

1. [#168](https://github.com/azerothl/Rbitnet/issues/168) — fused-on/off 1/4/8
   ablation, TTFT/ITL dispersion, KV page-limit capacity sweep.
2. [#169](https://github.com/azerothl/Rbitnet/issues/169) — live per-token SSE
   multiplexing and heterogeneous cancel/admit waves.
3. [#170](https://github.com/azerothl/Rbitnet/issues/170) — Qwen/GPT/MoE fused
   multi-seq and adaptive admission / CUDA graphs.

This matches the bounded close style used for #94 and #86: ship the measured
vertical, keep residual work ticketed.

## Reopen threshold

Reopen #96 only if the shipped Llama Sarathi fused path regresses correctness
or loses the measured multi-row fused work under the documented flags.
