# CUDA Llama fused HTTP ablation attempt (#168)

**Status: not accepted; #168 remains open.**

The reproducible harness is
[`scripts/benchmark_cuda_fused_http.py`](../../../scripts/benchmark_cuda_fused_http.py).
It starts fresh CUDA servers for fused-on and fused-off controls, warms them,
uses a client barrier for 1/4/8 concurrent SSE requests, and saves every
per-request TTFT, inter-token interval, total latency, client tok/s, GPU
snapshot, and `/metrics` delta into `results.json`.

## Same-revision attempt

On `origin/main` revision `85252eb152e4dce9f831a0073135db462bca7bed`, built
locally as `C:\rbitnet-build-168main\release\rbitnet-server.exe`, the run used:

- RTX 4080 SUPER (16,376 MiB), driver 610.88;
- Llama-3.2-1B-Instruct-Q4_K_M GGUF SHA-256
  `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`;
- 32 requested greedy output tokens, two warm-up waves, three measured waves;
- F32 paged KV, `RBITNET_CUDA_KV_PAGE_LIMIT=128`, and requested 1/4/8 matrix.

The run stopped during the fused-off streaming warm-up when two clients
completed without an SSE content delta:

```text
RuntimeError: 2 streaming requests failed:
[{... 'SSE completed without content delta' ...},
 {... 'SSE completed without content delta' ...}]
```

No 1/4/8 comparison numbers are published because a partial matrix would not
be a valid fused-on/off ablation. This failure is a real result, not a
substituted timing.

## Capacity and telemetry contract

The harness records `rbitnet_core_gpu_llama_batch_{rows,waves,projections}_total`,
GPU upload/download bytes, managed CUDA live/peak/refusal counters, KV counters,
and `nvidia-smi` before/after snapshots where available. It also accepts
`--capacity-page-limits` for the 4-request fused probe. Keep
`RBITNET_CUDA_KV_PAGE_LIMIT` explicit: the prior 16 GiB dense-KV probe refused
a synchronized four-request fused wave
([evidence](../2026-10-08-cuda-fused-scheduler/HTTP_SARATHI_FUSED.md)).

## Reproduce

```powershell
python scripts/benchmark_cuda_fused_http.py `
  --model models/exported-llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf `
  --tokenizer models/exported-llama/tokenizer.json `
  --binary C:/rbitnet-build-168main/release/rbitnet-server.exe `
  --library target/performance-cache/llama-continuous-delivery-proof/cuda/rbitnet_cuda_quant64.dll `
  --output-dir C:/rbitnet-bench-168-main `
  --concurrency 1 4 8 --warmups 2 --repetitions 3 --max-tokens 32 `
  --kv-page-limit 128 --capacity-page-limits 64 128
```

The command exits non-zero on missing content, a request failure, or an
incomplete matrix. Do not close #168 until it completes and fused-on also has
client-observable multi-event SSE output, so ITL dispersion is measurable.
