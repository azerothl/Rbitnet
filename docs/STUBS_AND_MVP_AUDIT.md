# Stubs / MVP audit (issue #24)

Inventory of paths that are stubs, intentional smoke modes, shipped MVPs, or still open.
Last reviewed: **2026-09-30** against `main` (post #17–#21: radix, PLD, BitNet kernels, KV Q8, Sarathi).

## Intentional smoke paths (keep; not “fake support”)

| Path | Status | Notes |
|------|--------|-------|
| `RBITNET_STUB=1` | **Smoke only** | Synthetic HTTP completions; CI / API overhead. Documented as non-model. |
| `RBITNET_TOY=1` | **Smoke only** | Tiny in-process F32 LM; no GGUF. |
| Idle unload → stub engine | **Ops feature** | `RBITNET_IDLE_UNLOAD_SECS`; swaps to stub after idle. |

## Backends

| Backend | Status | Action |
|---------|--------|--------|
| `cpu` | **Production path** | Default native-first. |
| `cuda` / `rocm` / `vulkan` / `metal` | **Parity stubs / MVP** | See #22 / [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md). Not claimed as full GPU inference. |
| `hybrid` | **Partial** | CPU fallback documented; not a full offload product yet. |

## Serving / KV / speculative (native CPU stack)

| Feature | Issue #24 claim (2026-09-30) | Actual status |
|---------|------------------------------|---------------|
| Prefix **response** cache (`RBITNET_PREFIX_CACHE`) | Full responses only | **Correct** — distinct from KV reuse. |
| Prefix **KV** (`RBITNET_PREFIX_KV`) | “Not real KV reuse” | **Outdated claim** — dense/paged snaps + radix LRU + LCP agent reuse + `rbitnet_core_prefix_hit` shipped (#17). |
| Continuous batching fused multi-seq | Still stub | **Stall-free Sarathi schedule shipped** (#21); **fused multi-seq matmul still open**. |
| Speculative decoding | Scheduler MVP only | **PLD / n-gram draft + verify/accept shipped** (#18); further research (SPECTRA-class) open. |
| Paged KV / pool | — | **E2E opt-in shipped** (#16 era). |
| KV Q8 | — | **Compact CPU pages shipped** (#20). |
| BitNet ternary kernels | — | **I2_S / TL2 microbench shipped** (#19). |
| Roadmap loaders (`glm4moe`, `gptoss`, `deepseek2`) | Stub executors | **Clear startup error** via `roadmap_unsupported` when not Llama-shaped; Llama-shaped tensors run for real. Full MoE/MLA = #25. |
| `tokenizer.model` | Manual conversion required | **SentencePiece path shipped** (`prompt_tokenizer.rs`); `tokenizer.json` preferred. |

## Docs sync done with this audit

- [LIMITATIONS.md](LIMITATIONS.md) — distinguish `PREFIX_CACHE` vs `PREFIX_KV`; stop implying KV reuse is absent.
- [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) — serving pipeline row reflects stall-free + PLD shipped; fused multi-seq still open.
- [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) — Phase E Done includes PLD + KV Q8 + BitNet kernel microbench.

## Remaining #24 work (backlog)

1. **Fused multi-seq forward** (CPU first, then GPU) — largest remaining serving gap.
2. **GPU backends** — owned by #22.
3. **Non-Llama / MoE / MLA loaders** — owned by #25.
4. Optional: SlimAttention 1D tiling after KV Q8; MTP research. Lookahead Decoding is **wontfix for now** — [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md).
5. Keep stub/toy clearly labeled forever (do not remove — CI depends on them).

## Exit criteria mapping

| Criterion | State |
|-----------|-------|
| Exhaustive stub audit | **This document** |
| No path *announced as supported* is a silent stub | **Improved** via LIMITATIONS/STATUS sync; GPU still explicitly stub |
| Unsupported listed as unsupported | **Yes** for MoE/MLA/GPU |
