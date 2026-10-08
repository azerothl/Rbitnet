# Decision: ship the bounded CUDA Llama Sarathi fused vertical; keep #96 open

**Status:** accepted as an opt-in Llama CUDA product slice; [#96](https://github.com/azerothl/Rbitnet/issues/96) remains open  
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
- OpenAI-compatible SSE requests can enter the same rendezvous, but their
  bridge buffers a completed shared result before emitting its content delta.

The measured 4- and 8-request HTTP probes prove the intended fused work was
actually reached: GPU batch rows exceed GPU batch waves. This supersedes the
older statement that the Llama GPU path was only a scheduler hook.

## Product boundary

This decision deliberately ships one observable, opt-in architecture/path:
CUDA Llama with F32 native KV, bounded rows, and the OpenAI-compatible chat
transport. It does not claim that all native executors, all transport shapes,
or all KV configurations share a forward.

CPU Llama/BitNet and Qwen retain their existing sequential execution. GPT and
MoE paths do not enter this vertical. Unsupported combinations must remain
explicit rather than silently claiming fused throughput.

## Why #96 cannot close yet

Unlike #94 and #86, the issue's stated acceptance is broader than this product
slice. Its required evidence and correctness scope have not been reduced by
the issue itself. In particular, the published HTTP probe is a single,
fused-on routing proof, not the required benchmark protocol:

1. It has no same-revision fused-off/native-separate ablation, warm-up
   repetitions, or TTFT/inter-token/total-latency dispersion.
2. It does not validate heterogeneous contexts, arrivals while decoding,
   cancellation, and completion/stop/sampling parity through a shared live
   wave.
3. The SSE bridge is buffered: it is not live per-token multiplexing, so it
   cannot establish streaming TTFT or inter-token behavior.
4. Qwen, GPT, and MoE have no fused multi-sequence forward, including the
   requested per-token expert routing/packing treatment.
5. Admission is fixed and bounded; adaptive admission/backpressure is not
   implemented or measured. CUDA graphs are likewise not part of this
   multi-sequence acceptance.

The HTTP probe also found a concrete operational constraint: a 16 GiB GPU
with the dense model-maximum KV allocation rejects a synchronized four-request
wave. Bounded paged F32 KV made the measured 4/8 waves succeed, but the
capacity policy has not yet been swept across prompt lengths and GPU memory
budgets.

Therefore the truthful PR linkage is **`Progress on #96`**, not `Closes #96`.
Closing the issue would represent the unmeasured ablation, transport, and
architecture acceptance as completed.

## Reopen and completion threshold

Keep #96 open until a frozen revision publishes the issue's requested
fused-on/fused-off 1/4/8 matrix with raw repetitions, warm-up, aggregate and
per-request decode throughput, client TTFT, inter-token latency dispersion,
total latency, RAM/VRAM, and transfer counters. It must also demonstrate
heterogeneous admission/cancellation correctness and define whether live SSE
token multiplexing is delivered or moved into a separately accepted issue.

Qwen/GPT/MoE fused forwards, adaptive admission/backpressure, and CUDA graphs
are explicit follow-ups. They may become separately scoped decisions, but
they cannot be implied by this Llama-only shipment.
