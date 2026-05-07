# Benchmarks and performance baselines

Reproducible numbers belong here for **Phase 2** of [`PLAN_PRODUCTION.md`](PLAN_PRODUCTION.md). The table below is a **template** until a baseline is measured and filled in (see [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md)).

**Cross-repo:** the Akasha reference parity matrix (`https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/reference-products-parity-matrix.md`) links **Perf / SLO** to this repo when documenting self-hosted OpenAI-compatible baselines.

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

## Gate A benchmark harness (CPU vs CUDA)

Use the helper script to produce comparable JSON output per backend:

```bash
# Terminal 1 (CPU)
RBITNET_BACKEND=cpu cargo run -p bitnet-server --bin rbitnet-server --release

# Terminal 2
python scripts/bench_backend_compare.py --model rbitnet-llama --runs 15
```

Then repeat with CUDA:

```bash
# Terminal 1 (CUDA)
RBITNET_BACKEND=cuda cargo run -p bitnet-server --bin rbitnet-server --release

# Terminal 2
python scripts/bench_backend_compare.py --model rbitnet-llama --runs 15
```

Copy `p50_ms`, `p95_ms`, and `mean_tok_s` for both runs into the table below and compute the ratio (`cuda_tok_s / cpu_tok_s`) as Gate A evidence.

Automated runner (starts CPU then CUDA server, benchmarks both, prints markdown row):

```bash
python scripts/bench_gate_a_runner.py --model rbitnet-llama --runs 12
```

Baseline interne (initiale, à mettre à jour par machine):

| Setup | Prompt tokens (approx.) | max_tokens | p50 ms | p95 ms | notes |
|-------|---------------------------|------------|--------|--------|--------|
| Ryzen 7 7840HS / 32 GB / GGUF q4_k_m | 700 | 128 | 820 | 1340 | commit local phase Hermes, endpoint `/v1/chat/completions` |

Baseline Sprint 1 (architecture multi-backend, mode CPU de référence):

| Setup | Backend | Model family | Prompt tokens (approx.) | max_tokens | p50 ms | p95 ms | tok/s | notes |
|-------|---------|--------------|--------------------------|------------|--------|--------|-------|-------|
| Ryzen 7 7840HS / 32 GB / toy path | `cpu` | `bitnet` | 300 | 128 | 112 | 167 | 74 | `RBITNET_MODEL_FAMILY=bitnet`, `RBITNET_BACKEND=cpu` |
| Ryzen 7 7840HS / 32 GB / stub path | `cpu` | `stub` | 300 | 128 | 9 | 14 | n/a | API overhead baseline only |

Validation de conformité inter-backend (Sprint 2 préparatoire):

| Test | CPU | CUDA MVP | ROCm stub | Vulkan stub | Metal stub |
|------|-----|----------|-----------|-------------|------------|
| `backend_numeric_parity_cpu_vs_stubs` | pass | pass | pass | pass | pass |

Backend matrix (phase Atlas-like):

| Backend | Runtime detection | Numeric parity test | Notes |
|---------|-------------------|---------------------|-------|
| `cpu` | n/a | pass | baseline reference |
| `cuda` | dynamic CUDA runtime load (`cudart`) | pass | uses native runtime copy path when available |
| `rocm` | dynamic HIP runtime load | pass | bootstrap path, optimization backlog |
| `vulkan` | dynamic Vulkan loader detection | pass | bootstrap path, optimization backlog |
| `metal` | dynamic Metal loader detection | pass | second-stage backend |

## Peak RAM

Rough peak RSS depends on model size, context, and OS. Document **model path basename**, **quantization**, and **observed RSS** for one reference machine if you publish a baseline.
