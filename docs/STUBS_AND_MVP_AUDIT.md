# Stubs / MVP audit (epic #24)

Inventory of paths that are stubs, intentional smoke modes, shipped MVPs, or still open.
Last reviewed: **2026-10-02** against `main` (post #16–#21 + #46 fused multi-seq stall decision + #44 Lookahead decision).

**Scope of this spike:** refresh the audit and map remaining work to child issues. This does **not** eliminate stubs end-to-end; epic [#24](https://github.com/azerothl/Rbitnet/issues/24) stays open until those issues land (or are explicitly deferred).

## Epic #24 checklist (status map)

| Theme | Status | Owner issue |
|-------|--------|-------------|
| Continuous batching **fused** multi-seq | **Stalled / closed** | [#46](https://github.com/azerothl/Rbitnet/issues/46) — kernel + scheduler hook shipped; e2e concurrency gain stalled → [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md); GPU fused = #22 |
| Speculative beyond scheduler MVP | **Partial** | PLD / n-gram shipped (#18); Lookahead **wontfix for now** → [#44](https://github.com/azerothl/Rbitnet/issues/44) / [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md) |
| Roadmap loaders (`glm4moe`, `gptoss`, `deepseek2`) | **Partial** | Clear `roadmap_unsupported` when non-Llama; full MoE/MLA → [#25](https://github.com/azerothl/Rbitnet/issues/25) |
| GPU backends | **Open** | [#22](https://github.com/azerothl/Rbitnet/issues/22) |
| SlimAttention / KIVI | **Partial** | SlimAttention proto + KIVI no-go → [#39](https://github.com/azerothl/Rbitnet/issues/39) closed |
| `tokenizer.model` without manual conversion | **Shipped** | SentencePiece path in `prompt_tokenizer.rs`; prefer `tokenizer.json` |
| Each conversion updates LIMITATIONS / STATUS | **Ongoing** | Required on each child issue close |

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
| `cuda` / `rocm` / `vulkan` / `metal` | **Parity stubs / MVP** | [#22](https://github.com/azerothl/Rbitnet/issues/22) / [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md). Not claimed as full GPU inference. |
| `hybrid` | **Partial** | CPU fallback documented; not a full offload product yet (#22). |

## Serving / KV / speculative (native CPU stack)

| Feature | Status | Notes |
|---------|--------|-------|
| Prefix **response** cache (`RBITNET_PREFIX_CACHE`) | **Shipped (MVP)** | Full responses only — distinct from KV reuse. |
| Prefix **KV** (`RBITNET_PREFIX_KV`) | **Shipped** | Dense/paged snaps + radix LRU + LCP agent reuse + `rbitnet_core_prefix_hit` (#17). **Present** — do not claim absent. |
| Continuous batching schedule | **Shipped (MVP)** | Stall-free Sarathi (#21); decode-first + chunked prefill. |
| Continuous batching fused multi-seq | **Stalled** | #46: kernel + hook shipped; e2e sequential executors — [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md); GPU fused = #22. |
| Speculative decoding | **Shipped (PLD)** | PLD / n-gram draft + verify/accept (#18); further research open. |
| Paged KV / pool | **Shipped** | E2E opt-in (#16 era). |
| KV Q8 | **Shipped** | Compact CPU pages (#20). |
| SlimAttention 1D tiling / KIVI | **Open** | [#39](https://github.com/azerothl/Rbitnet/issues/39) — after KV Q8. |
| BitNet ternary kernels | **Shipped (microbench)** | I2_S / TL2 (#19). |
| Roadmap loaders | **Clear refuse / Llama-shaped only** | Full MoE/MLA = [#25](https://github.com/azerothl/Rbitnet/issues/25). |
| `tokenizer.model` | **Shipped** | SentencePiece path; `tokenizer.json` preferred. |

## Remaining work (issue map)

- [x] **[#46](https://github.com/azerothl/Rbitnet/issues/46)** — Fused multi-seq: spike + **stall decision** ([FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md)); true Llama batched forward deferred; GPU via #22.
- [ ] **[#22](https://github.com/azerothl/Rbitnet/issues/22)** — GPU backends (CUDA / ROCm / Vulkan / Metal) + hybrid residency.
- [ ] **[#25](https://github.com/azerothl/Rbitnet/issues/25)** — Non-Llama / MoE / MLA loaders beyond Llama-shaped refuse path.
- [x] **[#39](https://github.com/azerothl/Rbitnet/issues/39)** — SlimAttention tiled CPU attention (+ KIVI no-go) after KV Q8.
- [x] **[#44](https://github.com/azerothl/Rbitnet/issues/44)** — Lookahead Decoding **wontfix for now** — [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md).
- Keep stub/toy clearly labeled forever (do not remove — CI depends on them).

## Docs sync

- [LIMITATIONS.md](LIMITATIONS.md) — distinguishes `PREFIX_CACHE` vs `PREFIX_KV`; KV reuse is **documented as present** (opt-in); fused multi-seq **stalled** → [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md).
- [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) — serving pipeline reflects stall-free + PLD shipped; fused multi-seq stalled; Lookahead wontfix.
- [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) — Phase E Done includes PLD + KV Q8 + BitNet kernel microbench.

## Exit criteria (epic #24)

| Criterion | State |
|-----------|-------|
| Exhaustive stub audit | **This document** (refreshed 2026-10-02) |
| No path *announced as supported* is a silent stub | **Improved** — GPU / MoE explicitly open via #22 / #25; fused multi-seq explicitly stalled |
| Unsupported listed as unsupported | **Yes** for MoE/MLA/GPU / fused e2e gain |
