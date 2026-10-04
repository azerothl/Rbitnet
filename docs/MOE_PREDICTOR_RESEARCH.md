# Expert prediction after the cache-policy ablation

Research checked on 4 October 2026. This note supports #84 and #85; it does not
implement a trained predictor or resolve the retained GLM Least-Stale divergence.

## What the papers establish

[Mira v1](https://arxiv.org/html/2609.38090v1), sections IV-E and V–VI, trains
per-layer predictors using hidden activations, router logits and selection history
to stage experts two layers ahead. Its evaluation mainly uses Mixtral-8x7B,
with DeepSeek-V2-Lite-Chat also included, on 8/24/48 GB GPUs. Its runtime also
changes expert storage to a custom INT8 format. The reported gains cannot be
assigned to prefetch alone: on the 48 GB setup, adding prediction to Mira yields
1.07× throughput and 1.85× TTFT improvement; on the 24 GB setup, these factors
are 1.32× and 1.42×. These are paper results, not predictions for Rbitnet.

[MoE-Infinity v3](https://arxiv.org/html/2401.14361v3) uses activation traces to
guide expert replacement and prefetching for personal-machine inference. It
supports studying a trace-based baseline before adding a trained predictor.
Our previous-token baseline does not reproduce that complete system.

## Observed constraints in this repository

The [warmed GPT-OSS CUDA capture](benchmarks/2026-10-04-async-expert-cache/production/trace-analysis.json)
uses the previous-pass predictor and a 512 MiB expert pool. For the dedicated
weight-copy stream it records 126,666,374,400 uploaded bytes, 4,989,725,427 ns
of copy duration and 543,795,810 ns overlapping compute on another stream:
10.898% of copy time. The other-stream compute union is 818,379,693 ns.
This covers the captured request, including prefill; it is not a decode-only
throughput measurement. CUDA tracing perturbs timing.

The same capture records no warm CUDA allocation/free calls. Consequently,
removing allocation calls alone cannot explain a steady-state speedup there.
This observation does not exclude an arena benefit during model loading.

The [policy measurements](benchmarks/2026-10-04-expert-policies/README.md)
also retain slower LFU/Least-Stale cases. More hits or a different policy name
are insufficient adoption criteria. The initial Least-Stale output fault remains
an independent correctness blocker.

## Decision and next experiment

Keep previous-pass prefetch and alternative replacement policies opt-in. Do not
translate Mira's INT8 representation into an unmeasured requantification of the
existing Q4_K/Q6_K/MXFP4 GGUF files.

For prediction, first compare causal one-/two-layer trace baselines at identical
physical cache and pinned-buffer budgets, recording useful, late and unused
copies as well as actual wait time. Split training/validation by request and
prompt family; repeated benchmark cycles must not leak into the holdout.
Existing ID-only route logs lack Mira's hidden-state/router-logit features, so
they cannot train or validate its MixtureHead as described.

A trained-feature experiment requires a separately bounded trace collector,
model/tokenizer identity, measured collection/training cost and held-out
coverage. GPU prediction, staging pressure and false-positive transfers must
then be included in quiet HTTP/SSE measurements. The router's real choice and
GGUF expert bytes remain authoritative. Adopt only after numerical/output,
ownership and end-to-end measurements pass; predictor training is still pending.
