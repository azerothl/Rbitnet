# Decision: Lookahead Decoding — wontfix for now

**Status:** **wontfix for now** (deferred research)  
**Issue:** [#44](https://github.com/azerothl/Rbitnet/issues/44)  
**Paper:** [Lookahead Decoding (2402.02057)](https://arxiv.org/abs/2402.02057)  
**Depends on:** PLD / n-gram speculative shipped (#18)  
**Date:** 2026-10-02

## Verdict

Do **not** implement Jacobi / tree-attention Lookahead in the near term. Prompt-lookup decoding (PLD) / n-gram draft already covers the “training-free speculative, no second GGUF” niche that matters for local CPU GGUF/BitNet serving. Reopen only if the criteria below are met.

Medusa / EAGLE / EAGLE-2 remain separately excluded (draft heads / fine-tune; see [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md#explicitly-deferred-do-not-chase-near-term)).

## What shipped instead (PLD)

| Piece | Where |
|-------|--------|
| Speculative verify/accept frame | `bitnet-core` scheduler (`RBITNET_SPECULATIVE`) |
| Default draft path | `RBITNET_DRAFT_PATH=ngram` → `prompt_lookup_draft` |
| Accept metric | `rbitnet_core_draft_accept` on `/metrics` |

PLD reuses tokens already in the prompt (suffix n-gram continuation). It hooks the existing draft → verify → accept path without a draft model, without tree attention, and without changing the Sarathi / continuous-batching wave schedule (#21).

## Fit with the current scheduler

Lookahead Decoding predicts future tokens via **Jacobi-style parallel drafting** and typically needs **multi-branch / tree attention** (or equivalent FLOPs-heavy verify) to keep acceptance lossless.

That sits poorly on today’s stack:

1. **Draft API is linear.** The scheduler draft path returns a single string continuation (`ngram` / optional secondary draft), then verifies a **prefix** accept. There is no tree of candidate branches, no branch-merge KV, and no attention mask for parallel Jacobi positions.
2. **CPU-first cost model.** Lookahead increases FLOPs per step to raise parallelism; on desktop CPU (the native-first default) that often hurts wall-clock latency unless acceptance is high and kernels are fused. PLD’s cost is near-zero CPU when the n-gram hits, and zero when it misses beyond a cheap string lookup.
3. **Batching interaction.** Stall-free Sarathi waves (#21) already schedule decode-first + chunked prefill. Tree drafts would need per-request branch budgets that fight `RBITNET_ITERATION_TOKEN_BUDGET` and multi-seq packing. Fused multi-seq forward is still open; Lookahead would deepen that gap.
4. **KV / prefix reuse.** Prefix KV + radix (#17) assume a single token spine. Branching drafts need either speculative KV forks or recompute — neither exists in `PagedSeqKv` / prefix snapshots today.

A limited “n-gram Jacobi” spike could theoretically reuse `prompt_lookup_draft` ideas, but it would still need a new verify path and metrics semantics; it is not a small env-flag on the PLD path.

## Risks vs PLD

| Dimension | PLD / n-gram (shipped) | Lookahead (Jacobi / tree) |
|-----------|------------------------|---------------------------|
| Extra model weights | None | None (attractive) but needs custom forward |
| Implementation surface | String draft + existing verify | Tree attention / multi-position logits + KV forks |
| CPU local gain | Real on repetitive / templated prompts; free miss path | Uncertain; often more FLOPs/step |
| Scheduler fit | Matches linear draft/verify | Requires new branch scheduling |
| Golden / lossless story | Prefix accept against target logits | Same goal, harder to validate under batching |
| Product priority | Aligns with Akasha / download-and-serve GGUF | Research; blocks no prod exit criteria |

**Bottom line:** Lookahead’s upside overlaps PLD’s (no draft GGUF) while its downside (complexity, FLOPs, scheduler/KV shape) is much larger. Prefer measuring and tuning PLD `draft_accept` + fused multi-seq before any Jacobi prototype.

## Criteria to reopen

Reopen #44 (or a successor) only when **all** of the following hold:

1. **PLD ceiling documented** — Frozen bench shows PLD `draft_accept` saturated on target workloads (agent/system+tools + interactive) and tok/s still below an agreed Akasha SLO vs llama.cpp / peer servers.
2. **Scheduler spine ready** — Fused multi-seq decode waves exist (or a clear single-seq research harness) so tree draft cost can be measured without confounding batching gaps.
3. **Scoped spike plan** — Written design for ≤N Jacobi positions, KV strategy (fork vs recompute), accept metric series, and a kill criterion (e.g. no ≥X% decode tok/s gain on CPU TinyLlama/BitNet within Y engineer-days).
4. **No Medusa/EAGLE creep** — Spike stays training-free; no draft-head fine-tune.

Until then, treat Lookahead as **deferred research** in the serving epic, not a backlog implementation item.

## Acceptance (this note)

- [x] Decision written in docs (wontfix for now + reopen triggers)
- [x] Linked from roadmap / inference stack / differentiation docs
- [x] No Jacobi / tree-attention implementation in this change

## Related

- [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) — Phase E speculative backlog
- [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) — research priorities
- [FUTURE_DIFFERENTIATION.md](FUTURE_DIFFERENTIATION.md) — product axes (PLD metrics, not Lookahead)
- [LIMITATIONS.md](LIMITATIONS.md) — training-free speculative beyond PLD is separate research
