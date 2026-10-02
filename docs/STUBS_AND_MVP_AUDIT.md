# Stubs / MVP audit (epic #24)

Inventory of paths that are stubs, intentional smoke modes, shipped MVPs, or still open.
Last reviewed: **2026-10-02** against `main` (post #72 SlimAttention decode + #73/#75 MoE close + #74 CUDA Gate E).

**Scope of this refresh:** resolve a leftover merge conflict in this file, mark [#25](https://github.com/azerothl/Rbitnet/issues/25) closed, and leave epic [#24](https://github.com/azerothl/Rbitnet/issues/24) open **only** while GPU [#22](https://github.com/azerothl/Rbitnet/issues/22) hardware remainder remains.

## Epic #24 checklist (status map)

| Theme | Status | Owner issue |
|-------|--------|-------------|
| Audit: exhaustive stubs / MVP / silent fake parity | **This document** | [#24](https://github.com/azerothl/Rbitnet/issues/24) |
| Prefix-KV real (shared-prefix reuse) | **Shipped** | [#17](https://github.com/azerothl/Rbitnet/issues/17) — not response-cache-only |
| Continuous batching **fused** multi-seq | **Stalled / closed** | [#46](https://github.com/azerothl/Rbitnet/issues/46) — kernel + scheduler hook shipped; e2e concurrency gain stalled → [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md); GPU fused = #22 |
| Speculative beyond scheduler MVP | **Partial / deferred** | PLD / n-gram shipped (#18); Lookahead **wontfix for now** → [#44](https://github.com/azerothl/Rbitnet/issues/44) / [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md) |
| Roadmap loaders / non-Llama / MoE | **Exit met (#25 closed)** | Qwen3 dense golden CI (#73) + Mixtral MoE `/v1` e2e + CI golden (#75); roadmap tags still Llama-shaped-or-refuse; DeepSeek MLA = follow-up |
| GPU backends | **Open (hardware)** | [#22](https://github.com/azerothl/Rbitnet/issues/22) — Gate E quant residency API landed (#74); ROCm/Vulkan/Metal parity stubs; measured CUDA tok/s still open |
| SlimAttention / KIVI | **Shipped proto + decode opt-in** | [#39](https://github.com/azerothl/Rbitnet/issues/39) closed — `RBITNET_SLIM_ATTENTION=1` wired into Llama CPU/hybrid decode; KIVI no-go |
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
| `cuda` | **Partial (Gate E)** | Device-resident f32 + **quant** (`CudaDeviceQuantMatrix`) + Llama cuda/hybrid prefer quant residency ([GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md)); hardware tok/s + `librbitnet_cuda_quant` device kernels still open → [#22](https://github.com/azerothl/Rbitnet/issues/22). |
| `rocm` / `vulkan` / `metal` | **Parity stubs** | Library probe may set `is_native_accelerated`; **matvec still CPU**. Not claimed as full GPU inference → #22. |
| `hybrid` | **Partial** | Prefers quant residency over densify when type supported; placement budgets + CPU fallback (#22 Gate E). |

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
| Non-Llama dense / Mixtral MoE | **Shipped (spike exit)** | Qwen3 dense golden CI + Mixtral MoE `/v1` e2e (#25 closed). |
| Roadmap loaders (`glm4moe`, `gptoss`, `deepseek2`) | **Clear refuse / Llama-shaped only** | DeepSeek MLA / broader MoE = follow-up (not silent stubs). |
| `tokenizer.model` | **Shipped** | SentencePiece path; `tokenizer.json` preferred. |

## Remaining work (issue map)

- [x] **[#46](https://github.com/azerothl/Rbitnet/issues/46)** — Fused multi-seq: spike + **stall decision** ([FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md)); true Llama batched forward deferred; GPU via #22.
- [ ] **[#22](https://github.com/azerothl/Rbitnet/issues/22)** — **Sole epic blocker:** Gate E quant residency API landed (CI-safe); hardware CUDA vertical (`librbitnet_cuda_quant` device symbols + published tok/s + greedy parity) + ROCm/Vulkan/Metal beyond parity stubs.
- [x] **[#25](https://github.com/azerothl/Rbitnet/issues/25)** — Closed after [#73](https://github.com/azerothl/Rbitnet/pull/73) + [#75](https://github.com/azerothl/Rbitnet/pull/75) (Qwen3 dense golden CI + Mixtral MoE `/v1` e2e).
- [x] **[#39](https://github.com/azerothl/Rbitnet/issues/39)** — SlimAttention tiled CPU attention (+ KIVI no-go); decode opt-in wired.
- [x] **[#44](https://github.com/azerothl/Rbitnet/issues/44)** — Lookahead Decoding **wontfix for now** — [LOOKAHEAD_DECISION.md](LOOKAHEAD_DECISION.md).
- Keep stub/toy clearly labeled forever (do not remove — CI depends on them).

### Blocker (keeps epic #24 open)

| Blocker | Why epic #24 stays open |
|---------|-------------------------|
| **#22** | Non-CPU backends still parity stubs / incomplete measured CUDA; Gate E is API+CI-safe only — hardware token path + published numbers + ROCm/Metal remain. |

## Docs sync

- [LIMITATIONS.md](LIMITATIONS.md) — PREFIX_CACHE vs PREFIX_KV; fused multi-seq **stalled**; architecture table (Qwen3 + Mixtral supported; DeepSeek MLA not implemented).
- [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) — serving pipeline + SlimAttention decode opt-in; remaining stubs → #22 only.
- [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) — Phase E includes PLD + KV Q8 + BitNet microbench + SlimAttention decode wire.
- [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md) — #46 stall decision.
- [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md) — #22 acceptance gates A–E.

## Exit criteria (epic #24)

| Criterion | State |
|-----------|-------|
| Exhaustive stub audit | **This document** (refreshed 2026-10-02 post #72–#75; conflict markers removed) |
| No path *announced as supported* is a silent stub | **Improved** — CPU / MoE spike paths closed or labeled; **remainder = #22 only** |
| Unsupported listed as unsupported | **Yes** for DeepSeek MLA / GPU hardware / fused e2e gain / Lookahead / KIVI |
| Close epic | **Only when #22 exits** (or is explicitly deferred) |
