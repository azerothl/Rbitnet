# Stubs / MVP audit (epic #24)

Inventory of paths that are stubs, intentional smoke modes, shipped MVPs, or still open.
Last reviewed: **2026-10-02** against `main` (post #51–#71: fused multi-seq stall #46, SlimAttention decode wire, Lookahead #44, GPU/MoE spikes #68/#69).

**Scope of this refresh:** close satisfied checklist items in the audit map, wire remaining **non-#22/#25** leftover (SlimAttention decode), and leave epic [#24](https://github.com/azerothl/Rbitnet/issues/24) open only while GPU (#22) and MoE/MLA (#25) remain.

## Epic #24 checklist (status map)

| Theme | Status | Owner issue |
|-------|--------|-------------|
| Audit: exhaustive stubs / MVP / silent fake parity | **This document** | [#24](https://github.com/azerothl/Rbitnet/issues/24) |
| Prefix-KV real (shared-prefix reuse) | **Shipped** | [#17](https://github.com/azerothl/Rbitnet/issues/17) — not response-cache-only |
| Continuous batching **fused** multi-seq | **Stalled / closed** | [#46](https://github.com/azerothl/Rbitnet/issues/46) — kernel + scheduler hook shipped; e2e concurrency gain stalled → [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md); GPU fused = #22 |
| Speculative beyond scheduler MVP | **Partial / deferred** | PLD / n-gram shipped (#18); Lookahead **wontfix for now** → [#44](https://github.com/azerothl/Rbitnet/issues/44) / [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md) |
| Roadmap loaders (`glm4moe`, `gptoss`, `deepseek2`) | **Clear refuse; MoE open** | Llama-shaped only + `roadmap_unsupported`; full MoE/MLA → [#25](https://github.com/azerothl/Rbitnet/issues/25) |
| GPU backends | **Open** | [#22](https://github.com/azerothl/Rbitnet/issues/22) — CUDA residency spike landed; ROCm/Vulkan/Metal remain parity stubs |
| SlimAttention / KIVI | **Shipped proto + decode opt-in** | [#39](https://github.com/azerothl/Rbitnet/issues/39) closed — 1D tile + drift gate; `RBITNET_SLIM_ATTENTION=1` wired into Llama CPU/hybrid decode; KIVI no-go |
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
| `cuda` | **Partial (spike)** | Device-resident GEMV hook + acceptance gates A–D ([GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md)); not full token path yet → [#22](https://github.com/azerothl/Rbitnet/issues/22). |
| `rocm` / `vulkan` / `metal` | **Parity stubs** | Library probe may set `is_native_accelerated`; **matvec still CPU**. Not claimed as full GPU inference → #22. |
| `hybrid` | **Partial** | Placement budgets + CPU fallback documented; not a full offload product yet (#22). |

## Serving / KV / speculative (native CPU stack)

| Feature | Status | Notes |
|---------|--------|-------|
| Prefix **response** cache (`RBITNET_PREFIX_CACHE`) | **Shipped (MVP)** | Full responses only — distinct from KV reuse. |
| Prefix **KV** (`RBITNET_PREFIX_KV`) | **Shipped** | Dense/paged snaps + radix LRU + LCP agent reuse + `rbitnet_core_prefix_hit` (#17). **Present** — do not claim absent. |
| Continuous batching schedule | **Shipped (MVP)** | Stall-free Sarathi (#21); decode-first + chunked prefill. |
| Continuous batching fused multi-seq | **Stalled** | #46: kernel + hook shipped; e2e sequential executors — [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md); GPU fused = #22. |
| Speculative decoding | **Shipped (PLD)** | PLD / n-gram draft + verify/accept (#18); Lookahead wontfix (#44). |
| Paged KV / pool | **Shipped** | E2E opt-in (#16 era). |
| KV Q8 | **Shipped** | Compact CPU pages (#20). |
| SlimAttention 1D tiling / KIVI | **Shipped (opt-in decode)** | Proto + KIVI no-go (#39); decode uses tiled path when `RBITNET_SLIM_ATTENTION=1`. |
| BitNet ternary kernels | **Shipped (microbench)** | I2_S / TL2 (#19). |
| Roadmap loaders | **Clear refuse / Llama-shaped only** | Full MoE/MLA = [#25](https://github.com/azerothl/Rbitnet/issues/25). |
| `tokenizer.model` | **Shipped** | SentencePiece path; `tokenizer.json` preferred. |

## Remaining work (issue map)

- [x] **[#46](https://github.com/azerothl/Rbitnet/issues/46)** — Fused multi-seq: spike + **stall decision** ([FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md)); true Llama batched forward deferred; GPU via #22.
- [ ] **[#22](https://github.com/azerothl/Rbitnet/issues/22)** — GPU backends (CUDA token path + ROCm / Vulkan / Metal beyond parity stubs) + hybrid residency on hardware.
- [ ] **[#25](https://github.com/azerothl/Rbitnet/issues/25)** — Non-Llama / MoE / MLA loaders beyond Llama-shaped refuse path (+ dense Qwen3 golden CI).
- [x] **[#39](https://github.com/azerothl/Rbitnet/issues/39)** — SlimAttention tiled CPU attention (+ KIVI no-go); decode opt-in wired.
- [x] **[#44](https://github.com/azerothl/Rbitnet/issues/44)** — Lookahead Decoding **wontfix for now** — [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md).
- Keep stub/toy clearly labeled forever (do not remove — CI depends on them).

### Blocked on siblings (do not duplicate here)

| Blocker | Why epic #24 stays open |
|---------|-------------------------|
| **#22** | Non-CPU backends still parity stubs / incomplete CUDA; silent “accelerated” claims must become real token paths or stay explicitly unsupported. |
| **#25** | MoE expert routing / MLA still refused; exit needs ≥1 dense non-Llama golden **and** ≥1 MoE e2e. |

## Docs sync

- [LIMITATIONS.md](LIMITATIONS.md) — PREFIX_CACHE vs PREFIX_KV; fused multi-seq **stalled**; architecture refuse table (#25).
- [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) — serving pipeline + SlimAttention decode opt-in; Lookahead wontfix.
- [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) — Phase E includes PLD + KV Q8 + BitNet microbench + SlimAttention decode wire.
- [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md) — #46 stall decision.
- [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — #22 acceptance gates.

## Exit criteria (epic #24)

| Criterion | State |
|-----------|-------|
| Exhaustive stub audit | **This document** (refreshed 2026-10-02 post #51–#71) |
| No path *announced as supported* is a silent stub | **Improved** — CPU serving stubs closed or labeled; **remainder = #22 / #25** |
| Unsupported listed as unsupported | **Yes** for MoE/MLA/GPU / fused e2e gain / Lookahead / KIVI |
| Close epic | **Only when #22 and #25 exit** (or are explicitly deferred) |
