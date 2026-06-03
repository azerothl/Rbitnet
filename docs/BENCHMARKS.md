# Benchmarks and performance baselines

Reproducible numbers belong here for **Phase 2** of [`PLAN_PRODUCTION.md`](PLAN_PRODUCTION.md). Use the **frozen baseline procedure** below whenever you publish or refresh numbers (see [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md)).

**Cross-repo:** the Akasha reference parity matrix (`https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/reference-products-parity-matrix.md`) links **Perf / SLO** to this repo when documenting self-hosted OpenAI-compatible baselines.

For **CPU profiling** workflow (perf, flamegraph, what to inspect in code), see [PROFILING.md](PROFILING.md). Archived per-release notes live under [docs/profiling/](profiling/README.md).

Local one-off model performance notes for the current Windows developer machine live in [LOCAL_PERFORMANCE_2026-05-12.md](LOCAL_PERFORMANCE_2026-05-12.md).

## Frozen baseline procedure

When you record an official row for a release candidate:

1. Note **git SHA**, **`rustc -V`**, **OS**, **CPU model**, **RAM**.
2. Fix **`RBITNET_BACKEND`**, model family env if applicable, **`RBITNET_MODEL`** (or stub/toy flags).
3. Use a **fixed** HTTP JSON body (`messages`, `max_tokens`, `temperature`) and enough repetitions for stable **p50 / p95** (or script output from `scripts/bench_backend_compare.py`).
4. Record **tokens/s** when the runtime reports completion tokens (or approximate from response).
5. Optional: **peak RSS** for the server process during the run (`ps`, Activity Monitor, `/proc`). For Llama GGUFs, compare **`RBITNET_LLAMA_WEIGHT_MODE=dense`** vs **`auto`** / **`mmap_quant`** on the same checkpoint — mmap-eligible runs should track **file-sized** weight residency plus KV overhead; dense loads spike toward full **`f32`** parameter RAM.
6. Optional **memory stability:** send moderate sustained traffic (dozens of requests) and confirm RSS does not grow without bound on a fixed workload — document tool and duration.

Add or refresh one row in **Baseline reference (frozen)** below.

## Small reproducible matrix

For a cheap smoke benchmark that starts `RBITNET_STUB=1`, runs a few HTTP requests, and appends markdown to [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md):

```bash
scripts/bench_matrix.sh
```

```powershell
.\scripts\bench_matrix.ps1
```

For real model throughput, start `rbitnet-server` yourself with `RBITNET_MODEL` and `RBITNET_TOKENIZER`, then run the matrix script with `NO_START_SERVER=1` / `-NoStartServer` and set `MODEL` / `-Model` to the id returned by `/v1/models`.

The matrix wrappers call `scripts/bench_backend_compare.py` and append a reproducible markdown row. Use `docs/MODEL_MATRIX.md` to decide which model/tokenizer/template combination to test; do not publish placeholder rows as performance claims.

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

## Comparison vs llama.cpp (reference CPU)

Rbitnet does **not** ship llama.cpp; this section defines a **fair** procedure to measure the gap on your machine before claiming parity.

**Rules**

1. **Same GGUF file** for both engines.
2. **Same thread budget**: set llama.cpp `-t N` to match `RBITNET_THREADS` / physical cores you intend to use (avoid oversubscription vs Rbitnet’s internal pool).
3. **Comparable workload**: prefill-heavy vs decode-heavy — use `llama-bench` with fixed `-p` / `-n`, and Rbitnet `bench_backend_compare.py` with a prompt of similar **character length** and the same `max_tokens`.
4. Record **git SHA** of llama.cpp and Rbitnet, **rustc**, OS, CPU model.

**llama.cpp**

Build [llama.cpp](https://github.com/ggerganov/llama.cpp) and run `llama-bench` (example):

```bash
./build/bin/llama-bench -m /path/model.Q4_K_M.gguf -t 8 -p 512 -n 64
```

**Rbitnet**

Start `rbitnet-server` with `RBITNET_BACKEND=cpu`, `RBITNET_MODEL`, `RBITNET_TOKENIZER`, then:

```bash
export RBITNET_GGUF=/path/model.Q4_K_M.gguf
export RBITNET_THREADS=8
./scripts/compare_llamacpp_rbitnet.sh
```

PowerShell: `.\scripts\compare_llamacpp_rbitnet.ps1` (set `RBITNET_GGUF`, optional `LLAMA_BENCH`).

**Baseline row (fill in after first run)**

| Date | CPU | llama.cpp SHA | Rbitnet SHA | GGUF | threads | llama-bench tok/s (reported) | Rbitnet mean_tok_s (HTTP) | ratio |
|------|-----|---------------|-------------|------|---------|------------------------------|---------------------------|-------|
| *(fill)* | | | | | | | | |

Add refreshed rows here when changing kernels (quant pool, BLAS, CUDA cache).

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

### Baseline reference (frozen)

Copy the schema from the Sprint rows below; replace **`<GIT_SHA>`** per release.

| Release tag / SHA | Host | Backend | Model family | Prompt tok (approx.) | max_tokens | p50 ms | p95 ms | tok/s | Peak RSS | Notes |
|-------------------|------|---------|--------------|----------------------|------------|--------|--------|-------|----------|-------|
| *(example)* `v0.1.0` / `<GIT_SHA>` | Ryzen 7 7840HS / 32 GB | `cpu` | `bitnet` | 300 | 128 | 112 | 167 | 74 | *(optional)* | toy path: `RBITNET_TOY=1` |

### Historical / internal rows

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

Rough peak RSS depends on model size, context, and OS. For each **frozen** baseline row, add **model basename**, **quantization**, **peak RSS**, and how it was measured (`ps`, Task Manager, etc.).

| Model (basename) | Quant | Peak RSS | Host | Notes |
|------------------|-------|----------|------|-------|
| *(example)* `model.Q4_K_M.gguf` | Q4_K_M | *(MB)* | Ryzen 7 7840HS / 32 GB | Same run as HTTP baseline row |

Load guardrails (`RBITNET_MAX_LOAD_BYTES`, `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_VRAM_MB`, `RBITNET_BUDGET_MAX_SEQ`) are documented in [LIMITATIONS.md](LIMITATIONS.md) and [ENV_REFERENCE.md](ENV_REFERENCE.md).
