# Decision: CUDA KV F16/Q8 — shipped; KIVI deferred

**Status:** **closed with measured F16/Q8 path + KIVI no-go**  
**Issue:** [#93](https://github.com/azerothl/Rbitnet/issues/93)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-04-kv-formats/README.md](benchmarks/2026-10-04-kv-formats/README.md), [KIVI_DECISION.md](KIVI_DECISION.md), TF32+encoded guard (#115 / related)

## Verdict

Ship **`RBITNET_CUDA_KV_FORMAT=f16|q8`** for **Llama** dense/paged device KV as **opt-in**. Keep **F32** as the default native KV. Do **not** implement asymmetric **KIVI** in this ticket — already **no-go** in [KIVI_DECISION.md](KIVI_DECISION.md); reopen only if those RSS/PPL/tok/s gates pass.

## What shipped

| Piece | Where |
|-------|--------|
| F16 / Q8 device KV encode/decode + attention | Llama CUDA resident path |
| Dense and paged format keys in snapshots | Format included in cache identity |
| Measured capacity/speed trade-offs | `2026-10-04-kv-formats` (Q8 decode can regress ~35%; not default) |
| TF32 incompatibility guard with encoded KV | Reject mutable TF32 when encoded KV active |

## Explicitly out of this closure

| Topic | Reason |
|-------|--------|
| KIVI asymmetric 2-bit | Layout ≠ uniform row pack; gates in `KIVI_DECISION.md` unmet |
| Default-on Q8/F16 | Measured decode regressions on representative loads |
| Qwen / GPT / MLA device KV formats | Separate architecture work (#92 follow-ups) |

## Reopen criteria

Reopen only for (1) a KIVI spike that meets `KIVI_DECISION.md` gates, or (2) promoting F16/Q8 to default after a frozen ablation shows net VRAM win **without** >10% decode regression on the published Llama suite.
