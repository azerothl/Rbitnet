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

## Adaptive admission (opt-in)

Set `RBITNET_CUDA_CONTINUOUS_ADMISSION=adaptive` to apply backpressure before
allocating a new request-local KV owner. In each decode-first iteration, active
decode rows reserve one token each and active prefill owners reserve their next
128-token partition. Only the remaining iteration budget can admit waiting
owners. This prevents a burst from consuming resident KV slots when its first
prefill partition cannot run in that iteration.

The default `fifo` policy preserves the earlier eager slot-filling behavior.
`adaptive` is an experimental Llama-only policy: it does not add a Qwen,
GPT-OSS, GLM, or MoE multi-sequence forward, and it has no throughput claim
until its CUDA benchmark gate is published.

## Live SSE mux for fused Sarathi flags

The default #96 HTTP bridge coalesces compatible SSE requests but returns their
content only after the batch completes. Set `RBITNET_CUDA_LIVE_SSE_MUX=1`
together with the Sarathi fused flags below to select this worker instead:

```powershell
$env:RBITNET_CONTINUOUS_BATCHING = '1'
$env:RBITNET_FUSED_MULTI_SEQ = '1'
$env:RBITNET_CUDA_CONTINUOUS = '1'
$env:RBITNET_CUDA_LIVE_SSE_MUX = '1'
```

Each admitted owner then receives its `FirstToken`, token deltas, and terminal
event as the shared decode worker advances. New requests are admitted when a
slot is free; cancellation or an SSE disconnect retires only that owner before
the next wave. Per-owner KV state, sampling history, RNG, completion limit, and
finish reason remain independent. This is an opt-in Llama CUDA/F32-KV path:
prefix snapshots, speculative decoding, Split KV, TF32 prefill, and structured
output remain refused.

The GPU regression fixture
`RBITNET_LLAMA_CONTINUOUS_TEST=1 cargo test -p bitnet-core
optional_actual_llama_continuous_arrivals_departures_sampling_and_request_local_cancel_exact`
checks staggered arrivals, replacement admission, per-request cancellation,
sampling parity, token-event reconstruction, and survivor completion. It needs
the real CUDA GGUF/tokenizer fixture; it is intentionally skipped on ordinary
CI hosts.

`optional_actual_llama_live_mux_heterogeneous_deadline_cancel_and_admit_parity`
also enables the live-mux Sarathi flags. A callback simulates a stream deadline
after two token deltas, then a heterogeneous replacement is admitted while the
survivor is still live. It proves request-local deadline/disconnect cancellation,
replacement overlap, exact survivor/replacement outputs, and shared Native
projections with dense and paged KV. It does not prove HTTP stop-string
preemption; the server's UTF-8 stop filter remains covered separately.

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
$env:RBITNET_CUDA_CONTINUOUS_ADMISSION = 'adaptive'
$env:RBITNET_MAX_CONCURRENT = '16'
```

Set the GGUF, tokenizer and Native DLL paths as for ordinary CUDA serving.
Split KV, encoded KV, TF32 prefill, prefix snapshots, speculative bursts and
the Sarathi switches without `RBITNET_CUDA_LIVE_SSE_MUX=1` are refused with
this option. There is no batch CUDA graph, shared multi-request prefill GEMM
or scheduler for Qwen/GPT/GLM.
`Engine.complete_batch` continues to use its existing separate API path.

The worker bounds admission and sends events through owned request channels.
Dropping a stream or failing its callback cancels that owner, while other
requests continue. Shutdown retires pending owners and releases Native contexts
and scratch memory. The live CUDA fixture establishes the bounded
deadline/disconnect cancellation path; explicit HTTP stop strings can still
leave bounded remaining work before retirement, since immediate stop-string
preemption has not been established. Real subword JSON/tool grammar is refused.

Native counters distinguish successful shared waves, rows, matrix projections
and maximum rows from virtual scheduler activity. The public evidence is in
[the continuous Llama benchmark](benchmarks/2026-10-04-continuous-llama/README.md).
It shows a gain in aggregate serving throughput under concurrency and a small
regression with one slot. It does not establish faster single-request decoding
or parity with another inference engine.
