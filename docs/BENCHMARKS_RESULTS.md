# Benchmark Results

This file is the append-only target for the small local benchmark matrix scripts:

```bash
scripts/bench_matrix.sh
scripts/bench_paged_kv.sh   # dense vs paged vs RBITNET_KV_POOL @ concurrency 1/4/8
```

```powershell
.\scripts\bench_matrix.ps1
```

By default the scripts start `rbitnet-server` in `RBITNET_STUB=1` mode, run a tiny reproducible HTTP benchmark through `scripts/bench_backend_compare.py`, and append a markdown section below. For real model numbers, start the server yourself with `RBITNET_MODEL` and `RBITNET_TOKENIZER`, then run:

```bash
NO_START_SERVER=1 MODEL=rbitnet-llama RUNS=12 scripts/bench_matrix.sh
```

```powershell
.\scripts\bench_matrix.ps1 -NoStartServer -Model rbitnet-llama -Runs 12
```

Record hardware, model basename, quantization, backend, and peak RSS when publishing a release-quality row. Keep stub results clearly labeled as API overhead smoke tests, not model throughput proof.

## Paged KV E2E — 2026-09-26 (unit gate)

**Methodology:** `cargo test -p bitnet-core --test kv_storage_paged` (no GGUF in this agent image). Live RSS/tok/s matrix requires TinyLlama Q4 + `scripts/bench_paged_kv.sh` with `RBITNET_MODEL` / `RBITNET_TOKENIZER`.

| Check | Result |
|-------|--------|
| Dense↔paged offset roundtrip | pass |
| Local free-list reuse after `clear` | pass |
| Shared pool multi-seq open/close reclaim | pass |
| Shared pool concurrency page count &lt; dense-equivalent | pass |
| Shared `attention_scores_cpu` vs dense | pass |

Enable production path: `RBITNET_LLAMA_PAGED_KV=1` and/or `RBITNET_KV_POOL=1` (see [USAGE.md](USAGE.md)).

## Manual Result Template

Use this template when you benchmark a real GGUF outside the helper scripts:

| Host | OS | Rust | Backend | Model basename | Quant | Prompt tok | max_tokens | p50 ms | p95 ms | mean tok/s | Peak RSS | Command | Notes |
|------|----|------|---------|----------------|-------|------------|------------|--------|--------|------------|----------|---------|-------|
| TBD | TBD | `rustc -V` | `cpu` | `model.Q4_K_M.gguf` | `Q4_K_M` | TBD | 64 | TBD | TBD | TBD | TBD | `NO_START_SERVER=1 ...` | tokenizer/template source |

## llama.cpp comparison — 2026-09-25T20:07:31Z

**Methodology:** `scripts/compare_llamacpp_rbitnet.sh` with `SKIP_LLAMA=1` against `RBITNET_STUB=1` HTTP (no GGUF weights in this cloud agent image; `llama-bench` not installed). Numbers below are **API / stub overhead only** — not a model tok/s claim vs llama.cpp.

**Limitations / how to reproduce a fair row:**

1. Download the same TinyLlama Q4_K_M GGUF for both engines (`RBITNET_GGUF=…`).
2. Build llama.cpp `llama-bench` and set `LLAMA_BENCH`.
3. Start `RBITNET_BACKEND=cpu RBITNET_MODEL=… RBITNET_TOKENIZER=… rbitnet-server --release`.
4. Run `RESULTS_MD=docs/BENCHMARKS_RESULTS.md ./scripts/compare_llamacpp_rbitnet.sh` (unset `SKIP_LLAMA`).

| Field | Value |
|-------|-------|
| Date (UTC) | 2026-09-25T20:07:31Z |
| Host | Linux 6.12.94+ x86_64 |
| CPU | Intel(R) Xeon(R) Processor |
| Rust | `rustc 1.88.0 (6b00bc388 2025-06-23)` |
| Rbitnet SHA | `f5bc87e` (pre-change base; see PR for tip) |
| Threads | 4 |
| GGUF | `n/a` (stub) |
| llama-bench status | skipped_by_flag |
| Rbitnet HTTP rc | 0 |

### llama-bench output

```
(none — no llama.cpp binary / no GGUF in agent environment)
```

### Rbitnet HTTP output (stub)

```
{
  "runs": 5,
  "p50_ms": 1.525816999901508,
  "p95_ms": 2.069930200013914,
  "mean_ms": 1.4235430000098859,
  "mean_tok_s": 7093.551072528254
}
```

## Local bench 2026-09-25 20:07:40 +0000

- Git SHA: `f5bc87e`
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Base URL: `http://127.0.0.1:8080`
- Model: `rbitnet-stub`
- Runs: `8` warmup: `1` max_tokens: `32`

| Host | Backend | Model | p50 ms | p95 ms | mean ms | mean tok/s | Notes |
|------|---------|-------|--------|--------|---------|------------|-------|
| Unix local | stub/http | `rbitnet-stub` | 0.64 | 1.32 | 0.76 | 23530.82 | Small reproducible smoke bench |
