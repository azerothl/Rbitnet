# Benchmarks and performance baselines

Reproducible numbers belong here for **Phase 2** of [`PLAN_PRODUCTION.md`](PLAN_PRODUCTION.md).

For **CPU profiling** workflow (perf, flamegraph, what to inspect in code), see [PROFILING.md](PROFILING.md).

## Running Criterion benches (kernels)

From the repo root:

```bash
cargo bench -p bitnet-core
```

Use `--release` implicitly via Criterion’s profile. Capture the **CPU model**, **Rust version**, and **commit hash** when recording results.

## HTTP latency (manual)

1. Run `rbitnet-server` with a fixed GGUF and tokenizer (see [USAGE.md](USAGE.md)).
2. Send repeated `POST /v1/chat/completions` requests with a **fixed** JSON body (same `messages`, `max_tokens`, `temperature`).
3. Record **p50 / p95** latency and **tokens/s** (approximate from response length / wall time).

Template (fill in after measurement):

| Setup | Prompt tokens (approx.) | max_tokens | p50 ms | p95 ms | notes |
|-------|---------------------------|------------|--------|--------|--------|
| _CPU model / RAM_ | | | | | |

## Peak RAM

Rough peak RSS depends on model size, context, and OS. Document **model path basename**, **quantization**, and **observed RSS** for one reference machine if you publish a baseline.
