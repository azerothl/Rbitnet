# Decision: MoE expert prefetch — previous-pass opt-in; Mira no-go

**Status:** **closed with measured baseline + research no-go for trained Mira**  
**Issue:** [#84](https://github.com/azerothl/Rbitnet/issues/84)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-04-async-expert-cache/README.md](benchmarks/2026-10-04-async-expert-cache/README.md), [MOE_PREDICTOR_RESEARCH.md](MOE_PREDICTOR_RESEARCH.md), [research/2026-10-03-moe-prefetch-plan.md](research/2026-10-03-moe-prefetch-plan.md)

## Verdict

Ship **bounded async expert copies** (`RBITNET_MOE_ASYNC`) with **previous-pass** prefetch as **opt-in**. Keep the **synchronous** path as default. Do **not** adopt Mira-style trained predictors or Mira’s INT8 expert storage without a new held-out experiment that beats previous-pass on our GGUF bytes and HTTP/SSE suites.

## What shipped

| Piece | Where |
|-------|--------|
| Pinned H2D slots + CUDA events | Expert cache async path (#114) |
| READY/PENDING sharing one physical budget | Measured separately from demand loads |
| Previous-pass predictor (router IDs unchanged) | Opt-in; misses never alter routing |
| Overlap / transfer diagnostics | ~10.9% copy/compute overlap on one Nsight capture (not a general tok/s claim) |

## Mira / trained predictors — no-go for now

`MOE_PREDICTOR_RESEARCH.md` records that Mira’s reported gains mix storage format changes with prediction, need hidden-state/router-logit features our ID-only traces lack, and must not requantize Q4_K/Q6_K/MXFP4 GGUF experts to an unmeasured INT8 layout. Keep previous-pass; refuse default-on when representative workloads slow down.

## Explicitly deferred

| Topic | Follow-up |
|-------|-----------|
| Causal one-/two-layer trace baselines at equal budget | New experiment issue if pursued |
| Trained MixtureHead + held-out coverage | After feature collector exists |
| Declaring prefetch a general decode win | Needs frozen GPT-OSS+GLM ablation with net tok/s |

## Reopen criteria

Reopen if a held-out trace predictor (or Mira port) shows sustained HTTP/SSE tok/s or TTFT gains at equal physical cache + pinned RAM, with identical router IDs and outputs, on both GPT-OSS and GLM.
