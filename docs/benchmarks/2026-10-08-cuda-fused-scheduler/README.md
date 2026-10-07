# CUDA fused scheduler decode (#96 progress)

This slice wires Sarathi `RBITNET_CONTINUOUS_BATCHING` +
`RBITNET_FUSED_MULTI_SEQ` to the existing Native `rbitnet_cuda_llama_batch_step`
kernels (shared projection GEMMs across rows).

## Enable (CUDA Llama, F32 KV)

```powershell
$env:RBITNET_BACKEND = 'cuda'
$env:RBITNET_CUDA_PREFILL = '1'
$env:RBITNET_CUDA_KV_FORMAT = 'f32'
$env:RBITNET_CONTINUOUS_BATCHING = '1'
$env:RBITNET_FUSED_MULTI_SEQ = '1'
$env:RBITNET_CUDA_FUSED_DECODE_SLOTS = '8'   # optional, default 8
```

Do **not** set `RBITNET_CUDA_CONTINUOUS=1` on the same process (HTTP continuous worker path).

## Proof hooks

- Prometheus: `rbitnet_core_gpu_llama_batch_rows_total` should exceed
  `rbitnet_core_gpu_llama_batch_waves_total` when a decode wave has 2+ rows.
- Unit (GPU): `RBITNET_LLAMA_FUSED_SCHEDULER_TEST=1` plus the usual
  `RBITNET_TEST_GGUF` / `RBITNET_TOKENIZER` runs
  `fused_scheduler_batch_shared_projections_exceed_serial`.

## Not in this slice

- HTTP `/v1` concurrency 1/4/8 publication (see
  [2026-10-04-continuous-llama](../2026-10-04-continuous-llama/README.md) for the
  continuous worker path).
- Shared multi-request **prefill** GEMM.
- Qwen / GPT / MoE fused forwards.
