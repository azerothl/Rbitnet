# Native continuous Llama decoding

Enable `RBITNET_CUDA_CONTINUOUS=1` to route individual CUDA Llama requests
through one worker. Several active requests share actual Native projection
forwards, with separate KV contexts, sampling histories and random generators.
The option defaults to off. Prefill uses ordinary single-request chunks; only
decoding shares projection work across requests.

The implementation supports Native CUDA Llama F32, dense KV or one shared
Native page pool. Set `RBITNET_CUDA_PREFILL=1` and leave its block size at 128.
`RBITNET_CUDA_CONTINUOUS_SLOTS` defaults to 4 and accepts 1 through 8.
The queue defaults to 32 (`RBITNET_CUDA_CONTINUOUS_QUEUE`, maximum 64).
`RBITNET_CUDA_CONTINUOUS_TOKEN_BUDGET` defaults to 256. Its minimum is
128 plus the configured slot count, so one prefill chunk cannot starve active
decode rows. `RBITNET_MAX_CONCURRENT` must admit enough HTTP clients to make
batching useful.

An example PowerShell configuration for eight active requests:

```powershell
$env:RBITNET_BACKEND = 'cuda'
$env:RBITNET_CUDA_PREFILL = '1'
$env:RBITNET_CUDA_KV_FORMAT = 'f32'
$env:RBITNET_CUDA_SPLIT_KV = '0'
$env:RBITNET_CUDA_PREFILL_TF32X3 = '0'
$env:RBITNET_PREFIX_KV = '0'
$env:RBITNET_CUDA_CONTINUOUS = '1'
$env:RBITNET_CUDA_CONTINUOUS_SLOTS = '8'
$env:RBITNET_MAX_CONCURRENT = '16'
```

Set the GGUF, tokenizer and Native DLL paths as for ordinary CUDA serving.
Split KV, encoded KV, TF32 prefill, prefix snapshots, speculative bursts and
the legacy batching switches are refused with this option. There is no batch
CUDA graph, shared multi-request prefill GEMM or scheduler for Qwen/GPT/GLM.
`Engine.complete_batch` continues to use its existing separate API path.

The worker bounds admission and sends events through owned request channels.
Dropping a stream or failing its callback cancels that owner, while other
requests continue. Shutdown retires pending owners and releases Native contexts
and scratch memory. Unary timeout or an explicit stop string can leave bounded
remaining work before retirement; immediate cancellation of every such path
has not been established. Real subword JSON/tool grammar is refused.

Native counters distinguish successful shared waves, rows, matrix projections
and maximum rows from virtual scheduler activity. The public evidence is in
[the continuous Llama benchmark](benchmarks/2026-10-04-continuous-llama/README.md).
It shows a gain in aggregate serving throughput under concurrency and a small
regression with one slot. It does not establish faster single-request decoding
or parity with another inference engine.
