# Benchmarks and performance baselines

Reproducible numbers belong here for **Phase 2** of [`PLAN_PRODUCTION.md`](PLAN_PRODUCTION.md). Use the **frozen baseline procedure** below whenever you publish or refresh numbers (see [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md)).

**Cross-repo:** the Akasha reference parity matrix (`https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/reference-products-parity-matrix.md`) links **Perf / SLO** to this repo when documenting self-hosted OpenAI-compatible baselines.

For **CPU profiling** workflow (perf, flamegraph, what to inspect in code), see [PROFILING.md](PROFILING.md). Archived per-release notes live under [docs/profiling/](profiling/README.md).

Local one-off model performance notes for the current Windows developer machine live in [LOCAL_PERFORMANCE_2026-05-12.md](LOCAL_PERFORMANCE_2026-05-12.md).

## Real-model comparison: Rbitnet, Ollama and llama.cpp

The [second optimization round](benchmarks/2026-10-03-parity-round2/README.md) measures dense Qwen recurrent blocks with resident CUDA state, shared GPU output heads, sequence parity and concurrent runtime loading against fresh CPU/GPU references for all four models.

The [resident CUDA/SIMD optimization report](benchmarks/2026-10-03-parity/README.md) records the next four-model CPU/GPU measurements, actual CUDA graph execution, routed FFN fusion, validation and primary-source research. It distinguishes measured gains from unfinished work toward parity.

The [corrected native inference rerun](benchmarks/2026-10-03-optimized/README.md) measures all four Rbitnet models on CPU/CUDA after SIMD, resident quantized weights, fused attention, native GPT-OSS/MLA graphs and conversation-template fixes. The original report remains frozen; the rerun explicitly identifies reused Ollama/llama.cpp measurements and current Rbitnet measurements.

The [performance diagnosis of 3 October 2026](https://github.com/azerothl/Rbitnet/blob/main/docs/profiling/2026-10-03/README.md) traces the failed architectures and measures Llama operator costs plus six CPU/CUDA ablations on the same GGUF.

The [Windows CPU/GPU comparison of 3 October 2026](https://github.com/azerothl/Rbitnet/blob/main/docs/benchmarks/2026-10-03/README.md) uses four actual checkpoints: Llama 3.2 1B, Qwen3.5 2B, GPT-OSS 20B and GLM 4.7 Flash (MoE). Failed configurations are retained as failures, without substituted speeds. Raw token counts, generated answers, timings, weights SHA-256 and offload evidence accompany the report.

`scripts/benchmark_engines.py` runs the engines **sequentially**, with one request at a time. Within a model, all engines use the same GGUF, model-specific formatted prompt and greedy sampling. llama.cpp prepares the prompt fixtures and token IDs; Ollama receives the corresponding raw string, and Rbitnet uses `{user}` as its chat template. Every measured response must report the expected prompt-token count before its speed is eligible for comparison. Matching counts are a check against templating/BOS mistakes; they do not prove complete tokenizer equivalence for unsupported Rbitnet architectures.

The default workload is one excluded warm-up, three distinct story prompts capped at 32 generated tokens, one streaming latency probe, and three strict factual/short-memory answer checks. It reports actual runtime token counts, decoder phase throughput, full HTTP latency, process-tree working set and sampled global VRAM delta. The three questions detect basic answer failures; they are **not** a general language-model quality evaluation. Three short samples also do not justify p95 or long-context throughput claims.

Ollama is unloaded and preloaded with an empty prompt before each measured request so automatic prefix reuse does not improve its prefill unfairly. Loading time is outside the measured request, and the returned residual `load_duration` is retained. llama.cpp uses `cache_prompt=false`; Rbitnet prefix caching is disabled. GPU runs use each engine's native offload policy, with llama.cpp `-ngl auto`; CPU spill is recorded. Rbitnet CUDA keeps some operations on CPU. KV storage and context allocation differ between implementations and must be described with results rather than treated as identical memory workloads.

### Reproduce the comparison

Install the benchmark dependencies and build Rbitnet's release binary. CUDA acceleration also needs the optional native kernel DLL:

```powershell
python -m pip install -r scripts/requirements-benchmark.txt
cargo build --locked --release -p rbitnet-cli
./scripts/build_cuda_quant.ps1
Copy-Item scripts/engine_benchmark_manifest.example.json target/engine-benchmark-manifest.json
```

Edit the manifest's executable paths, environment/build identifiers and four model/tokenizer paths. Use absolute paths if `cwd` differs from the launch directory. Reference binaries are available from the [official llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/tag/b11351); keep CPU and CUDA builds on the same revision. Ollama must already be serving and have no model loaded in `/api/ps`.

For existing Ollama models, `ollama show MODEL --modelfile` exposes the `FROM` GGUF blob path. Check that llama.cpp can load it first: vendor GGUF architecture names or RoPE metadata can differ. The dated report pins compatible shared exports after encountering these differences. Import the same canonical export into Ollama with a dedicated alias using a Modelfile containing `FROM ABSOLUTE_PATH_TO_GGUF` and `ollama create ALIAS -f PATH_TO_MODELFILE`. The [Ollama Modelfile documentation](https://docs.ollama.com/modelfile) describes importing GGUF files. Rbitnet additionally needs the corresponding Hugging Face `tokenizer.json`; record its source revision and hash. Importing a model into Ollama does not add its topology to Rbitnet. For a separate Ollama daemon, configure `ollama_url` and `ollama_pid` in the manifest so memory is sampled from that instance.

```powershell
python scripts/benchmark_engines.py --manifest target/engine-benchmark-manifest.json --output target/engine-benchmark/results.json
python scripts/render_engine_benchmark.py target/engine-benchmark/results.json --output-dir target/engine-benchmark/report --plot
```

Use `--models ID`, `--engines rbitnet ollama llama.cpp` and `--backends cpu gpu` to select configurations; `--resume` skips already recorded completed rows with the same repetitions/token limit. Errors are completed observations, so use a separate output file when retrying a failed configuration. Prompt fixtures, logs and full JSON remain beside the output. Inspect both answer quality and the actual offload evidence before interpreting a tokens/s column. The [Ollama API](https://docs.ollama.com/api/generate) exposes counts and nanosecond phase durations; the harness converts durations to milliseconds explicitly.

`scripts/tests/test_benchmark_engines.py` checks token accounting, duration conversion, strict final-answer scoring and the protection against unloading a model the harness did not own. Run it with `python -m unittest discover -s scripts/tests -p test_benchmark_engines.py`.

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
2. **Same thread budget**: set llama.cpp `-t N` and Ollama `num_thread` to the available logical CPU budget. Set `RAYON_NUM_THREADS` for Rbitnet's Rayon operations; its quantized matvec pool independently uses `available_parallelism`, so do not assume `RBITNET_THREADS` controls that pool.
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
| *(see [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) for dated append-only sections; stub/API overhead is not a model tok/s claim)* | | | | | | | | |

Append helper:

```bash
RESULTS_MD=docs/BENCHMARKS_RESULTS.md ./scripts/compare_llamacpp_rbitnet.sh
# Without llama.cpp or GGUF on the machine:
SKIP_LLAMA=1 RESULTS_MD=docs/BENCHMARKS_RESULTS.md ./scripts/compare_llamacpp_rbitnet.sh
```

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
| `v0.1.0` / `serving-hooks-2026-06` | dev / Win32 | `cpu` | `stub` | 32 | 16 | *(fill)* | *(fill)* | *(fill)* | low | `RBITNET_STUB=1`; live SSE; see [profiling/2026-06-04-serving-hooks-baseline.md](profiling/2026-06-04-serving-hooks-baseline.md) |
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
