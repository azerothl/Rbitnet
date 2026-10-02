# KIVI 2-bit KV — go / no-go (after Q8)

Spike decision for [#39](https://github.com/azerothl/Rbitnet/issues/39). Context: uniform **KV Q8** compact pages already ship (`RBITNET_KV_QUANT=q8`, no F32 twin; unit drift gate in `kv_storage_paged`). Uniform **Q4** packs K and V the same way (`encode_quant_row` / nibble pack). **KIVI** ([2402.02750](https://arxiv.org/abs/2402.02750)) is *asymmetric*: **K per-channel**, **V per-token**, plus a recent FP window — not a drop-in on the current uniform row pack.

## Decision (spike)

| Verdict | Scope |
|---------|--------|
| **No-go for production merge of KIVI 2-bit in this PR** | Full asymmetric store + attention decode + golden/PPL is larger than a scoped spike. |
| **Hold Q8 as default KV quant** | Keep `RBITNET_KV_QUANT=q8` as the measured compact path; Q4 remains experimental uniform pack. |
| **Go for a follow-up only if gates pass** | Implement KIVI behind a flag after live RSS + PPL criteria below. |

SlimAttention 1D tiling lands as a **CPU path** (`llama::slim_attention`, `RBITNET_SLIM_ATTENTION=1` decode opt-in) with a tiled-vs-baseline drift gate — independent of KIVI.

## Why not implement KIVI here

1. **Layout mismatch:** today’s `KvQuantFormat::{Q8,Q4}` stores one scale (+ packed ints) **per token row** for both K and V. KIVI needs channel-wise scales on K and token-wise on V, plus mixed FP/quant pages for the recent window.
2. **Attention path:** decode-on-read (`fill_*_head_values` / `attention_scores_cpu`) assumes uniform row decode. Asymmetric dequant changes score and V-combine kernels.
3. **Acceptance (#39):** issue asks for documented go/no-go vs staying on Q8, with PPL / golden / RSS — not a half-wired format.

## Go criteria (follow-up PR)

Run against the same TinyLlama / BitNet fixtures used for Q8 (`scripts/bench_kv_q8.sh` pattern + optional golden):

| Gate | Threshold | Notes |
|------|-----------|--------|
| **RSS** | KIVI resident_bytes ≤ **0.55×** Q8 at equal `max_seq` / page config | Approximate 2-bit vs 8-bit; allow overhead for FP window + dual scale tables. |
| **PPL / golden** | ΔPPL ≤ **0.15** vs Q8 (or golden abs error within existing Q8 gate family) on a fixed prompt set | Fail closed: if PPL regresses beyond gate, keep Q8 default. |
| **Attention drift** | Synthetic max relative score/output drift ≤ **5%** vs F32 (same spirit as Q8 unit gate) | Unit test without GGUF. |
| **Flag** | Opt-in only (`RBITNET_KV_QUANT=kivi` or similar); default remains `off` / `q8` profiles | No silent default flip. |

## No-go criteria (stay on Q8)

- Live RSS savings **&lt; 20%** vs Q8 after packing overhead, **or**
- PPL / golden regression beyond the table above, **or**
- Decode tok/s drops **&gt; 10%** vs Q8 on the frozen CPU bench row.

## Pointers

- Uniform Q8/Q4: [`crates/bitnet-core/src/llama/kv_storage.rs`](../crates/bitnet-core/src/llama/kv_storage.rs)
- SlimAttention proto: [`crates/bitnet-core/src/llama/slim_attention.rs`](../crates/bitnet-core/src/llama/slim_attention.rs)
- Stack backlog: [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md)
- Paper: [KIVI](https://arxiv.org/abs/2402.02750) · reference impl [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)
