# Profiling hot paths (Phase 2)

Use this as a **working checklist** when investigating CPU time in `bitnet-core` (matmul, attention, dequantization). Publish a one-page summary in your PR when you change numerical kernels or the Llama forward.

**Archived snapshots:** store dated findings under [docs/profiling/](profiling/README.md) and link them from [BENCHMARKS.md](BENCHMARKS.md) for the same release.

## What to measure first

1. **`cargo bench -p bitnet-core`** — Criterion reports for ternary / matvec kernels (`benches/kernels.rs`).
2. **Wall time per token** — run `rbitnet-server` with `RUST_LOG=info`, a fixed prompt, and compare timestamps (or wrap with `time` / your APM).
3. **Sampling profiler** — on Linux, [perf](https://perf.wiki.kernel.org/) (`perf record -g -- ./target/release/rbitnet-server`) or [flamegraph](https://github.com/flamegraph-rs/flamegraph) on the binary.

## Host-side pipeline (CPU / HTTP first-token latency)

Industry write-ups often split **prefill** (compute-heavy) from **decode** (memory-bandwidth-heavy). Time-to-first-byte can still be dominated by **everything before the model**:

1. **HTTP / JSON** parsing and validation (`bitnet-server`).
2. **Tokenizer** `encode` (Rust `tokenizers` — compare wall time to **prefill_ms** from `Engine::complete_detailed` or Prometheus sums).
3. **Prompt assembly** in your client (templates, tools metadata).

If **encode_ms** rivals **prefill_ms** on long prompts, optimize templates or tokenizer loading before tuning kernels. Use **`GET /metrics`** phase counters (`rbitnet_inference_encode_ms_sum`, `…_prefill_ms_sum`, `…_decode_ms_sum`) after steady traffic, or log `InferenceStats` from **`complete_detailed`** for single-shot debugging.

## Likely hot spots (prioritized issues)

| Area | File / module | Notes |
|------|----------------|--------|
| Quantized matmul / matvec | `kernels.rs`, `ggml/dequant.rs`, **`ggml/quant_dot.rs`** | **`quant_dot`** implements mmap row GEMV for Llama (`RBITNET_LLAMA_WEIGHT_MODE`); SIMD there is the next win. |
| Attention + RoPE | `llama/model.rs` | Per-layer loops; KV cache access pattern matters. |
| GGUF dequant | `ggml/dequant.rs`, `tensor_to_f32` | Used for **`dense`** Llama weights and norms; mmap mode avoids full-weight `tensor_to_f32` for large matrices. |

## What to record

- Hardware (CPU model, RAM), Rust version, commit SHA.
- Command line and env (`RBITNET_MODEL`, context length, `max_tokens`).
- Before/after numbers for the same GGUF (or stub/toy for API-only changes).

Link the results from [BENCHMARKS.md](BENCHMARKS.md) when you publish a baseline.

## Sprint 3 profiling snapshot

Archived as [profiling/snapshot-2026-05.md](profiling/snapshot-2026-05.md) (update the date/SHA when you replace this content for a new release).

Scope:
- scheduler path with speculative decode enabled (`RBITNET_SPECULATIVE=1`)
- BitNet CPU MVP executor and backend abstraction layer
- kernel registry dispatch path (`llama/*`, `bitnet/*`)

Top hot paths observed (target order for Sprint 4):
1. `llama::model::forward` (attention loops and KV access)
2. `ggml::dequant::tensor_to_f32` + quantized dequant helpers
3. `kernels::matvec_ternary_i8` / `kernels::bitnet_cuda_matvec_mvp` compatibility path

Follow-ups:
- Replace fallback backend matvec in CUDA/ROCm/Vulkan paths with native kernels.
- Optimize speculative verify to avoid prompt re-tokenization duplication.
- Add per-op timing around dequant and attention decode for regression guards.
