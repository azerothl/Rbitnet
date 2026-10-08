# CUDA fused scheduler prefill + decode (#96 progress)

This slice wires Sarathi `RBITNET_CONTINUOUS_BATCHING` +
`RBITNET_FUSED_MULTI_SEQ` to the existing Native `rbitnet_cuda_llama_batch_step`
kernels (shared projection GEMMs across prompt-token and decode rows).

## Enable (CUDA Llama, F32 KV)

```powershell
$env:RBITNET_BACKEND = 'cuda'
$env:RBITNET_CUDA_PREFILL = '1'
$env:RBITNET_CUDA_KV_FORMAT = 'f32'
$env:RBITNET_CONTINUOUS_BATCHING = '1'
$env:RBITNET_FUSED_MULTI_SEQ = '1'
$env:RBITNET_CUDA_FUSED_DECODE_SLOTS = '8'   # optional, default 8
```

Set `RBITNET_CUDA_CONTINUOUS=0` (or leave it unset) for this buffered HTTP
probe. The live SSE experiment may instead set
`RBITNET_CUDA_CONTINUOUS=1` and `RBITNET_CUDA_LIVE_SSE_MUX=1`; that deliberately
selects the owned per-token worker while retaining the Sarathi fused flags.

## Proof hooks

- Prometheus: `rbitnet_core_gpu_llama_batch_rows_total` should exceed
  `rbitnet_core_gpu_llama_batch_waves_total` when a prefill or decode wave has
  2+ rows.
- Unit (GPU): `RBITNET_LLAMA_FUSED_SCHEDULER_TEST=1` plus the usual
  `RBITNET_TEST_GGUF` / `RBITNET_TOKENIZER` runs
  `fused_scheduler_batch_shared_projections_exceed_serial`.

## HTTP concurrency harness

This branch wires non-streaming `/v1/chat/completions` requests into the
Sarathi batch entry point: requests arriving during a short coalescing window
with the same loaded engine become one `Engine::complete_batch_detailed` call.
Run the 1/4/8 matrix on a CUDA host with a Llama GGUF:

```bash
RBITNET_MODEL=/path/model.gguf RBITNET_TOKENIZER=/path/tokenizer.json \
./scripts/bench_http_sarathi_fused.sh
```

The harness prints, but does not persist, wall time, requested aggregate tok/s,
and deltas for batch-row and batch-wave metrics. Publish the raw output with
GPU, driver, model quantization, prompt, completion length, and all relevant
`RBITNET_*` flags. No figures are recorded here because this checkout did not
run on a verified CUDA model host.

## Remaining gaps toward #96 close

- Streaming-request coalescing (the HTTP bridge currently covers non-streaming
  completions).
- Adaptive admission/backpressure and broader GPU/model validation.
- Qwen / GPT / MoE fused forwards.
