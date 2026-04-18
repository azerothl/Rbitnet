# Profiling hot paths (Phase 2)

Use this as a **working checklist** when investigating CPU time in `bitnet-core` (matmul, attention, dequantization). Publish a one-page summary in your PR when you change numerical kernels or the Llama forward.

## What to measure first

1. **`cargo bench -p bitnet-core`** — Criterion reports for ternary / matvec kernels (`benches/kernels.rs`).
2. **Wall time per token** — run `rbitnet-server` with `RUST_LOG=info`, a fixed prompt, and compare timestamps (or wrap with `time` / your APM).
3. **Sampling profiler** — on Linux, [perf](https://perf.wiki.kernel.org/) (`perf record -g -- ./target/release/rbitnet-server`) or [flamegraph](https://github.com/flamegraph-rs/flamegraph) on the binary.

## Likely hot spots (prioritized issues)

| Area | File / module | Notes |
|------|----------------|--------|
| Quantized matmul / matvec | `crates/bitnet-core/src/kernels.rs`, `ggml/dequant.rs` | Dominates large models. |
| Attention + RoPE | `crates/bitnet-core/src/llama/model.rs` | Per-layer loops; KV cache access pattern matters. |
| GGUF dequant | `ggml/dequant.rs`, `ggml/tensor_to_f32` | Per-layer weight loads are amortized after first forward if cached in `f32` (current design keeps dequant in `LlamaModel`). |

## What to record

- Hardware (CPU model, RAM), Rust version, commit SHA.
- Command line and env (`RBITNET_MODEL`, context length, `max_tokens`).
- Before/after numbers for the same GGUF (or stub/toy for API-only changes).

Link the results from [BENCHMARKS.md](BENCHMARKS.md) when you publish a baseline.
