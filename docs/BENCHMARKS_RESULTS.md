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
