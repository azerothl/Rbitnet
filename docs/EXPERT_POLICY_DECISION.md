# Decision: MoE expert cache policies — LRU default

**Status:** **closed with measured default**  
**Issue:** [#85](https://github.com/azerothl/Rbitnet/issues/85)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-04-expert-policies/README.md](benchmarks/2026-10-04-expert-policies/README.md), [MOE_PREDICTOR_RESEARCH.md](MOE_PREDICTOR_RESEARCH.md), GLM diagnostics [2026-10-04-glm-cache-diagnostics](benchmarks/2026-10-04-glm-cache-diagnostics/README.md)

## Verdict

Keep **LRU** as the default expert-cache replacement policy. Ship **LFU** and **Least-Stale** as opt-in alternatives. Do **not** promote Least-Stale (or LFU) to default.

## What the measurements showed

- Identical router IDs and GGUF expert bytes across policies on the published captures; policy choice must not rewrite routing.
- Hit-rate alone is insufficient: some LFU / Least-Stale cells transfer more or decode slower at the same budget (see GPT-OSS 8192 MiB tables in the expert-policies report).
- An early **Least-Stale GLM** synchronous run at 8 GiB produced a divergent narrative vs LRU. Later full passes and sanitizer diagnostics pass, but the **root cause of that first divergence remains unknown**. The negative capture stays under `negative/` and is not treated as a correctness proof after a later green run.

## Default choice

| Policy | Role |
|--------|------|
| **LRU** | Default — stable outputs in the published suites; competitive decode where cache helps |
| **LFU** | Opt-in ablation / research |
| **Least-Stale** | Opt-in experimental — not default until the retained GLM divergence is explained or reproduced as a non-issue under a frozen harness |

## Reopen criteria

Reopen only if (1) the Least-Stale GLM divergence is root-caused with a failing deterministic fixture, or (2) a frozen A/B on GPT-OSS + GLM shows a sustained decode/TTFT win for another policy **with** identical HTTP/SSE outputs and no unexplained mismatches.
