# Inference optima research — capacity & decode (2026-10-06)

Issue: [#144](https://github.com/azerothl/Rbitnet/issues/144)  
Native-first: llama.cpp / vLLM / Ollama / Strata are **oracle / idea sources only**, not runtimes ([NATIVE_FIRST.md](../NATIVE_FIRST.md)).

This note classifies levers for **(A) larger models** and **(B) more tok/s**. It does **not** implement kernels. Where work already has an issue, the verdict is **suivre #N**.

## Method

- Reuse measured baselines (PR #82 / RTX 4080 SUPER evidence, Qwen3.8-27B CPU smoke) plus existing go/no-go docs ([KIVI_DECISION.md](../KIVI_DECISION.md), [LOOKAHEAD_DECISION.md](../LOOKAHEAD_DECISION.md), [MOE_PREDICTOR_RESEARCH.md](../MOE_PREDICTOR_RESEARCH.md)).
- One fresh local ablation: TinyLlama Q4_K_M CPU, baseline vs `RBITNET_SPECULATIVE=1` + `RBITNET_DRAFT_PATH=ngram` (see [ablation folder](../benchmarks/2026-10-06-inference-optima-ablation/)).

## A — Capacity (bigger models)

| Piste | Verdict | Notes |
| --- | --- | --- |
| MoE expert VRAM cache HOT/STAGE + async prefetch | **suivre #83–#86** | Already scoped; keep measuring PCIe vs CPU expert cost |
| Hybrid CPU/GPU expert placement by real cost | **suivre #86** | Do not assume GPU always wins |
| Session context hierarchy VRAM/RAM/SSD | **suivre #94** | Distinct from weight streaming |
| KV F16/Q8 on CUDA, then KIVI | **suivre #93** / **no-go KIVI** until [KIVI_DECISION.md](../KIVI_DECISION.md) gates | Hold Q8 default |
| Remaining I-quant IQ4_NL | **go** (small) | Wire mmap GEMV like other I-quants; smoke on mixed GSQ-RCO if present |
| Dense layer-wise offload recipe (27B on 16 GiB) | **go** (measure-first) | Hypothesis: with `RBITNET_MAX_SEQ≤2k` + hybrid layers, Qwen3.8-27B CUDA hybrid completes `/v1` without host OOM; gate = HTTP 200 + RSS/VRAM log on 4080 SUPER |
| Context compression / real SWA families | **suivre #142** (Spark ISWA shipped CPU) | Generic “compress context” without arch support = no-go |

## B — Decode / TTFT (tok/s)

| Piste | Verdict | Notes |
| --- | --- | --- |
| Speculative that actually wins (PLD, draft GGUF, MTP) | **suivre #97** + evidence below | Lookahead remains [wontfix](../LOOKAHEAD_DECISION.md) |
| Prefill Tensor Core / tiled attention / FA2-class | **suivre #95**; FA2/FA3 **deferred** | Not desktop default ([INFERENCE_STACK_V2.md](../INFERENCE_STACK_V2.md)) |
| Resident decode graphs / less CPU↔GPU sync | **suivre #88 #89 #96** | GPT-OSS / GLM / multi-seq |
| True multi-seq GPU batching | **suivre #96** | Hook exists; execution still serial |
| BitNet packed GPU / LUT | **no-go near-term** | After CPU parity chase; research-only for now |
| Adaptive nucleus sampling | **suivre (shipped opt-in)** | Do not re-open; cite existing benches only |

## Fresh ablation — PLD n-gram on TinyLlama CPU

**Hypothesis (falsifiable):** `RBITNET_SPECULATIVE=1` + `RBITNET_DRAFT_PATH=ngram` improves decode wall time by **≥1.3×** vs baseline on TinyLlama-1.1B-Chat Q4_K_M, greedy, `max_tokens=32`, same machine, warm run.

**Setup:** cloud agent VM, `RBITNET_BACKEND=cpu`, `RBITNET_MAX_SEQ=512`, prompt via `engine_smoke` (“Say OK in one word.” → model verbose), release build.

| Run | Mode | decode_ms | itl_us | completion_tokens |
| --- | --- | ---: | ---: | ---: |
| base-1 (cold) | off | 1485 | 46406 | 32 |
| base-2 (warm) | off | 1466 | 45812 | 32 |
| spec-1 | ngram | 1426 | 44562 | 32 |
| spec-2 | ngram | 1400 | 43750 | 32 |

Warm decode: **1466 → ~1413 ms** ≈ **1.04×** (far below 1.3×). Text quality unchanged (same greedy continuation shape).

**Verdict:** **no-go** for treating PLD n-gram as a default “win” on short TinyLlama CPU prompts. Keep #97 open for draft-GGUF / higher-accept workloads; do not flip defaults. Raw lines: [results.txt](../benchmarks/2026-10-06-inference-optima-ablation/results.txt).

## Priority recommendations (implementation order)

1. Continue #88 / #89 / #95 / #96 on 4080 SUPER — highest leverage vs measured gaps vs llama.cpp.
2. MoE #83–#86 for GLM/GPT-OSS capacity on 16 GiB.
3. Small **go**: IQ4_NL decode parity with other I-quants.
4. Measure-first **go**: dense 27B hybrid recipe on 4080 (docs + script, not speculative kernels).
5. Do **not** chase Lookahead / FA2 default / BitNet packed GPU until the above move the needle.

## Mapping to code

| Area | Code |
| --- | --- |
| Speculative / PLD | `scheduler.rs`, `RBITNET_SPECULATIVE`, `RBITNET_DRAFT_PATH` |
| MoE expert cache | `native/expert_cache.rs`, MoE issues #83–#86 |
| KV quant | `llama/kv_storage.rs`, #93 |
| Prefill CUDA | #95 / `native/cuda_quant` |
| I-quants | `ggml/iq.rs`, #138 remainder IQ4_NL |

## Native-first reminder

Any “2× from paper X” claim requires a **local** ablation on the same GGUF revision before a merge. External engines stay oracles.
