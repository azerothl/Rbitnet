# Model Matrix

This matrix tracks curated reference GGUFs for smoke + published benches. Entries stay `verified: false` in `data/compatible_models.json` until a local smoke records `/v1` success **and** a row lands in [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md).

## Reference set (issue #23)

| Priority | Model id | Quant | Why |
|----------|----------|-------|-----|
| P0 | `tinyllama-1.1b-chat-q4-k-m` | Q4_K_M | Smallest reproducible CPU gate |
| P1 | `llama-3.2-1b-instruct-q4-k-m` | Q4_K_M | Modern 1B dense Llama-shaped |
| P1 | `bitnet-b158-2b4t` | native b1.58 | Product differentiator |
| P2 | `llama-3.2-3b-instruct-q4-k-m` | Q4_K_M | Optional scale-up |
| P2 | `hermes-2-pro-llama-3-8b-q4-k-m` | Q4_K_M | Optional larger chat |

## Reproduction Commands

Download and write a local config:

```bash
rbitnet up TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --write-config
rbitnet models download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dir models/tinyllama-tokenizer \
  --file tokenizer.json
```

Run a smoke chat:

```bash
rbitnet serve --open-ui
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"temperature":0.7}'
```

Record benchmark rows:

```bash
NO_START_SERVER=1 MODEL=rbitnet-llama RUNS=12 MAX_TOKENS=64 scripts/bench_matrix.sh
```

```powershell
.\scripts\bench_matrix.ps1 -NoStartServer -Model rbitnet-llama -Runs 12 -MaxTokens 64
```

Fair vs llama.cpp (same GGUF, same thread budget):

```bash
export RBITNET_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
export LLAMA_BENCH=/path/to/llama-bench
RESULTS_MD=docs/BENCHMARKS_RESULTS.md ./scripts/compare_llamacpp_rbitnet.sh
```

## Current Matrix

| Model id | Repo / file | Status | Peak RSS (published) | e2e tok/s (published) | Reproduce |
|----------|-------------|--------|----------------------|-----------------------|-----------|
| `tinyllama-1.1b-chat-q4-k-m` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` / `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | **CPU measured 2026-09-30** (throughput/RSS; see results) | ~723 MiB | ~0.75 (8 tok, 4-thread Xeon agent) | `rbitnet up ...` + tokenizer from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `llama-3.2-1b-instruct-q4-k-m` | `unsloth/Llama-3.2-1B-Instruct-GGUF` / `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | API metadata checked; e2e **non mesuré** (#41) | non mesuré (≥4 GB class) | non mesuré | `rbitnet up unsloth/Llama-3.2-1B-Instruct-GGUF --file Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| `llama-3.2-3b-instruct-q4-k-m` | `unsloth/Llama-3.2-3B-Instruct-GGUF` / `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | API metadata checked; e2e **non mesuré** | non mesuré (≥8 GB class) | non mesuré | `rbitnet up unsloth/Llama-3.2-3B-Instruct-GGUF --file Llama-3.2-3B-Instruct-Q4_K_M.gguf` |
| `hermes-2-pro-llama-3-8b-q4-k-m` | `NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF` / `Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf` | API metadata checked; e2e **non mesuré** | non mesuré (≥16 GB class) | non mesuré | `rbitnet up ... --chat-format chatml` |
| `bitnet-b158-2b4t` | curated BitNet b1.58 2B4T bundle (`microsoft/bitnet-b1.58-2B-4T-gguf`) | **Install + native loader documented**; e2e **non mesuré** (#41/#45). Kernel microbench exists (I2_S/TL2/AVX2 + TQ stack scratch) — see [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md). Remaining gap vs bitnet.cpp is mostly **layout/wiring** (GGUF TQ1/TQ2 mmap GEMV ≠ research I2_S helpers), not more scalar LUT chasing. | non mesuré (expect multi-GB RSS class) | **unpublished** | [BITNET_NATIVE.md](BITNET_NATIVE.md) + `rbitnet models install microsoft-bitnet-b1.58-2b-4t` |

Full dated numbers: [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) (*TinyLlama Q4_K_M CPU — 2026-09-30*).

## Promotion Rule

Before changing `verified` to `true`, attach:

- OS, CPU, RAM, `rustc -V`, git SHA.
- Exact GGUF basename, tokenizer source, and chat format.
- `/v1/models` output and one `/v1/chat/completions` response.
- One row appended by `scripts/bench_matrix.*` or an equivalent command using `scripts/bench_backend_compare.py`.
