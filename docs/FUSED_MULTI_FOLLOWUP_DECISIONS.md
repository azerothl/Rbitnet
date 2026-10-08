# Decision: do not ship unmeasured multi-owner CUDA follow-ups

**Status:** accepted; closes [#178](https://github.com/azerothl/Rbitnet/issues/178),
[#179](https://github.com/azerothl/Rbitnet/issues/179), and
[#180](https://github.com/azerothl/Rbitnet/issues/180)

**Date:** 2026-10-08

**Scope:** the deferred, architecture-specific work from the bounded #170
decision

## Verdict

Do not represent the existing Llama-only CUDA worker as an implementation of
Qwen recurrent decode, GPT-OSS/GLM MoE waves, or CUDA-graph replay. No
minimal opt-in slice is shipped for those architectures in this decision.

The Llama adaptive-admission policy remains experimental and opt-in, as
documented in [the #170 decision](FUSED_MULTI_ARCH_DECISION.md). It reserves
decode and prefill capacity before accepting another Llama owner; it is not an
admission result for Qwen or MoE, and it has not cleared a throughput or
tail-latency gate. CUDA graphs remain disabled for shared waves.

Closing these deferred tickets records a deliberate no-go rather than implying
that the parent vertical was completed. Future implementation work must reopen
the relevant decision with evidence; it must not silently extend the Llama
flags to another executor.

## #178: Qwen GDN/recurrent owners

No Qwen fused multi-owner decode/prefill is enabled. A Qwen implementation
cannot reuse the Llama worker by changing an architecture predicate: recurrent
or GDN state must be independently owned, initialized, advanced, cancelled,
and released for every request. The required first slice is therefore larger
than a safe configuration flag.

Reopen #178 only with all of the following:

1. an explicit Qwen eligibility boundary and opt-in;
2. request-local recurrent-state lifecycle tests covering staggered admission,
   cancellation, replacement, and shutdown;
3. a serial Qwen oracle proving tokens and finish reasons for every owner; and
4. an architecture-specific CUDA benchmark showing shared-wave counters plus
   TTFT, ITL, aggregate throughput, and memory/capacity results.

Until then, Qwen remains on its existing sequential path.

## #179: GPT-OSS and GLM MoE owners

No GPT-OSS or GLM MoE shared CUDA wave is enabled. Concatenating request rows
is insufficient: the executor needs a per-token expert-routing representation,
stable packing/scatter behavior, owner-local token positions, and a correctness
oracle that catches routing and ordering errors. Treating this as a generic
Llama projection batch would be an unsound claim.

Reopen #179 only with:

1. a documented per-token routing/packing contract, including capacity and
   overflow behavior;
2. an opt-in architecture-specific executor;
3. serial-oracle tests for mixed owners, expert skew, cancellation, and
   terminal ordering; and
4. benchmarks for both balanced and skewed routing, reporting expert packing,
   latency dispersion, throughput, and memory pressure.

Until then, MoE execution stays request-local and sequential.

## #180: adaptive admission and CUDA-graph gate

The repository has a reproducible fused HTTP harness at
[`scripts/benchmark_cuda_fused_http.py`](../scripts/benchmark_cuda_fused_http.py)
and a published Llama fused-on/off baseline in
[the #168 benchmark note](benchmarks/2026-10-08-cuda-fused-http-ablation/README.md).
That baseline demonstrates Llama shared-wave telemetry, but it does not
measure adaptive versus FIFO admission, nor graph replay. In particular, its
buffered bridge cannot observe streaming ITL dispersion. It is evidence of a
testable harness, not an acceptance result for adaptive admission.

The explicit acceptance gate for any reopened #180 is:

1. run FIFO and `RBITNET_CUDA_CONTINUOUS_ADMISSION=adaptive` under the same
   revision, model hashes, page limits, request mix, and 1/4/8 concurrency;
2. report per-owner TTFT and ITL p50/p95/max, aggregate completion rate,
   admission/refusal counts, queue delay, shared-wave counters, and managed/KV
   peak memory;
3. require no correctness or isolation regression, no capacity refusal
   regression at the documented page-limit sweep, and no material tail-latency
   regression for the admitted owners;
4. compare graph-off and graph-on only after that adaptive result passes, with
   graph capture/replay misses and memory included in the report; and
5. keep both policies and graphs opt-in until the architecture-specific
   benchmark is published.

No numeric speedup threshold is set here because the current evidence does not
measure this policy. Inventing one would turn a decision document into an
unsupported performance claim.

## Product boundary and reopen rule

The shipped product boundary is still CUDA-resident Llama with the documented
experimental admission setting. This decision closes the deferred follow-up
issues as no-go decisions, not as feature completion. Reopen only the affected
issue when its listed evidence can be supplied; no additional tracking ticket
is needed merely to preserve this boundary.
