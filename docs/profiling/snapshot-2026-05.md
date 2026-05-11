# Profiling snapshot — 2026-05 (template)

**Commit:** (fill at release)  
**CPU / RAM:** (fill)  
**Rust:** (fill `rustc -V`)

## Commands

```bash
cargo bench -p bitnet-core
# Optional:
# perf record -g -- ./target/release/rbitnet-server
# cargo flamegraph --bin rbitnet-server
```

## Top hot paths (observed)

1. `llama::model::forward` — attention loops and KV access.
2. `ggml::dequant::tensor_to_f32` — quantized weight paths.
3. Kernel matvec / BitNet CUDA MVP compatibility paths.

## Follow-ups

- Replace fallback backend matvec on stub GPU paths with native kernels where applicable.
- Reduce redundant tokenization on speculative verify passes (scheduler).
- Optional: extra timers around dequant vs decode for regression guards.

_Link this file from [BENCHMARKS.md](../BENCHMARKS.md) baseline rows when publishing numbers for the same release._
