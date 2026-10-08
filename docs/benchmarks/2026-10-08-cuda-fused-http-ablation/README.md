# CUDA Llama fused HTTP ablation (#168)

**Status: the 1/4/8 fused-on/off matrix completes and every successful
response includes a nonempty SSE content delta. #168 remains open**: this
buffered Sarathi bridge emits one whole-response delta, so client ITL
dispersion is still unobservable; its requested capacity sweep also needs
multiple prompt lengths.

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

The earlier failed control disabled `RBITNET_CONTINUOUS_BATCHING`, which made
it a singleton-runtime comparison rather than the non-fused Sarathi path.
The harness now holds that switch at `1` in both legs and varies only
`RBITNET_FUSED_MULTI_SEQ` (`0` versus `1`).

## Measured matrix

The raw 2026-10-08 run is committed under [`raw/`](raw/):
[`results.json`](raw/results.json) has every client sample, environment,
fixture and SHA-256; the server logs retain the process diagnostics.

- Host: RTX 4080 SUPER (16,376 MiB), driver 610.88; F32 paged KV, 128-page
  limit; two warm-ups and three measured waves for each 1/4/8 configuration.
- Fixture: Llama-3.2-1B-Instruct-Q4_K_M GGUF
  (`3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`),
  greedy 32-token requests.
- All 78 matrix responses completed with one nonempty content delta. The
  fused-off control records zero Native shared-wave counters; fused-on records
  1,760 batch rows / 445 waves / 49,855 projections for the matrix.
- The first measured single-request waves were 132.3 ms (241.9 requested
  tok/s) fused-off and 143.3 ms (223.3 requested tok/s) fused-on. At
  concurrency 4 and 8, the raw per-wave samples show roughly 16 requested
  aggregate tok/s in either leg; do not infer an improvement from this run.
- Fused-on 4-request capacity probes at 64 and 128 pages both completed; they
  cover only this prompt length and therefore do not meet the issue's required
  prompt-length capacity sweep.

Each response contains exactly one content event because the default HTTP
bridge releases the result after its Sarathi batch completes. Therefore
`itl_ms` is intentionally null, not a zero-latency claim.

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
  --binary C:/rbitnet-build-168-sse/release/rbitnet-server.exe `
  --library native/cuda_quant/build/rbitnet_cuda_quant64.dll `
  --output-dir docs/benchmarks/2026-10-08-cuda-fused-http-ablation/raw `
  --concurrency 1 4 8 --warmups 2 --repetitions 3 --max-tokens 32 `
  --kv-page-limit 128 --capacity-page-limits 64 128
```

The command exits non-zero on missing content, a request failure, or an
incomplete matrix. Before closing #168, repeat the capacity probes over the
agreed prompt-length set and use the live SSE mux to collect multiple
client-observable deltas per request for ITL dispersion.
