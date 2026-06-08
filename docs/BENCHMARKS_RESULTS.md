# Benchmark Results

This file is the append-only target for the small local benchmark matrix scripts:

```bash
scripts/bench_matrix.sh
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

## Manual Result Template

Use this template when you benchmark a real GGUF outside the helper scripts:

| Host | OS | Rust | Backend | Model basename | Quant | Prompt tok | max_tokens | p50 ms | p95 ms | mean tok/s | Peak RSS | Command | Notes |
|------|----|------|---------|----------------|-------|------------|------------|--------|--------|------------|----------|---------|-------|
| TBD | TBD | `rustc -V` | `cpu` | `model.Q4_K_M.gguf` | `Q4_K_M` | TBD | 64 | TBD | TBD | TBD | TBD | `NO_START_SERVER=1 ...` | tokenizer/template source |
