# Decision: speculative decoding — done opt-in, no-go as default

**Status:** **closed with documented no-go for default activation**  
**Issue:** [#97](https://github.com/azerothl/Rbitnet/issues/97)  
**Date:** 2026-10-08  
**Evidence:** [benchmarks/2026-10-03-speculative-llama/README.md](benchmarks/2026-10-03-speculative-llama/README.md), PR [#100](https://github.com/azerothl/Rbitnet/pull/100), Qwen draft experiment PR [#123](https://github.com/azerothl/Rbitnet/pull/123) (closed draft; measured slowdown)

## Verdict

Ship **Llama PLD** (prompt-lookup draft + CUDA multi-position verify + rollback + cost guard) as **opt-in** (`RBITNET_SPECULATIVE` / `RBITNET_SPECULATIVE_PLD`). Do **not** enable speculative decoding by default.

Acceptance for #97 allows either a measured net win **or** a documented no-go. The published ablations show a net win only on a short repetitive Greek sequence (~19% tok/s at PLD 15); free-form narrative and code prompts do not improve, and early ungated graphs **regressed** narrative latency. Keep the adaptive cost guard; leave the flag off.

## What shipped

| Piece | Where |
|-------|--------|
| Multi-position CUDA verify + truncate/rollback | `llama` resident path, PR #100 |
| Target-coupled accept/correct sampling | Sampler + distribution tests |
| Cost guard | Drops further proposals when block verify exceeds observed serial cost |
| HTTP/SSE parity | Greedy / seed / penalties; disconnect-resume; stops |
| Metrics | draft/verify timings and accept counters |

## Explicitly out of product default

| Topic | Reason |
|-------|--------|
| Default-on PLD | No net win on representative prompts |
| Small Llama draft GGUF that beats the target | Not demonstrated; verify/head GEMM dominates on 1B-class targets |
| Qwen draft GGUF as default | PR #123: exact outputs, measured **slower** than target alone |
| Medusa / EAGLE / Lookahead trees | Already deferred ([LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md)) |

## Reopen criteria

Reopen or spawn a follow-up only if a measured workload shows **sustained** end-to-end tok/s gain after draft+verify cost on at least two prompt families (short repetitive **and** free-form ≥128 tokens), with HTTP/SSE parity and no acceptance of draft tokens before target verify.
