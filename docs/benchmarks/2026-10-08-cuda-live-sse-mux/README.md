# CUDA live SSE mux validation (#169)

`RBITNET_CUDA_LIVE_SSE_MUX=1` is an explicit CUDA Llama/F32-KV opt-in. It
selects the owned continuous worker, which emits request-local SSE token
deltas while owners share decode waves. It is distinct from the default
Sarathi HTTP bridge:

- Default (`RBITNET_CUDA_CONTINUOUS=0`): compatible streaming requests may
  share work, but their content is buffered until the batch finishes. A client
  sees one content delta, so client-observable ITL is not measurable.
- Live mux (`RBITNET_CUDA_CONTINUOUS=1` plus
  `RBITNET_CUDA_LIVE_SSE_MUX=1`): every owner gets deltas during shared decode;
  the harness rejects a response with fewer than two content deltas, making
  reported TTFT and ITL real client observations.

The real-GPU regression fixture
`optional_actual_llama_continuous_heterogeneous_mid_wave_cancel_and_admit`
uses three intentionally different requests. It cancels one after token
delivery, admits a new owner while the surviving request is still decoding,
requires a subsequent two-owner decode wave, and verifies exact independent
sampling results for both surviving owners. It runs against dense and paged
Native KV fixtures.

An HTTP `stop` sequence now preempts its live-mux owner after the matched delta:
the endpoint emits the stop terminal packet and rejects the producer's next
callback, so the owner cannot contribute another decode row. The deterministic
HTTP regression is
`streaming_stop_preempts_the_producer_and_finishes_immediately`; it verifies
that the stop text is withheld and the client receives `finish_reason: "stop"`
plus `[DONE]`. The CUDA-gated
`optional_actual_cuda_live_mux_http_stop_preempts_heterogeneous_wave` derives a
real greedy stop string, stops that HTTP owner, and verifies that heterogeneous
survivor and replacement streams both finish under the live-mux flags.

The companion
`optional_actual_llama_live_mux_heterogeneous_deadline_cancel_and_admit_parity`
fixture exercises the owned controller selected by the live-mux flags. It
simulates a stream deadline after two deltas, admits a heterogeneous replacement
before the survivor completes, and checks exact terminal output plus shared
Native projection counters. This establishes deadline/disconnect cancellation
parity for a live heterogeneous wave; explicit HTTP stop-string preemption is
still not established.

## Measurement command

On a CUDA host with a compatible Llama GGUF, build the server from the exact
revision being measured, then run:

```powershell
python scripts/benchmark_cuda_fused_http.py `
  --live-sse-mux `
  --model path/to/model.gguf `
  --tokenizer path/to/tokenizer.json `
  --binary path/to/rbitnet-server.exe `
  --library path/to/rbitnet_cuda_quant64.dll `
  --output-dir docs/benchmarks/2026-10-08-cuda-live-sse-mux/raw `
  --concurrency 1 4 8 --warmups 2 --repetitions 3 --max-tokens 32 `
  --kv-page-limit 128
```

`results.json` records every request's TTFT, inter-token intervals, total
latency, event count, model/tokenizer/binary/library hashes, GPU snapshots,
configuration, and native counter deltas. The command fails if any streaming
request fails, emits no content, or exposes fewer than two content deltas.

## Measured evidence

The committed [`raw/results.json`](raw/results.json) was captured on
`851f1208f9269fab4ab0b9ce7cfde428efc42ed6` with an RTX 4080 SUPER (16,376
MiB), driver 610.88, and the Llama-3.2-1B-Instruct-Q4_K_M GGUF SHA-256
`3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.
It contains three measured waves after two warm-ups for each 1/4/8
concurrency, with 32 greedy completion tokens per request.

For the required fused-on live mux configuration, every request exposed 32
content events (31 client-observable ITLs). Client p50 TTFT / ITL were:

- concurrency 1: 68.0 ms / 2.50 ms;
- concurrency 4: 127.9 ms / 5.05 ms;
- concurrency 8: 219.2 ms / 9.43 ms.

The matching p95s were 77.4 / 3.28 ms, 194.8 / 6.39 ms, and 380.1 / 11.24
ms. The artifact also has a live-worker control with fused multi-sequence
disabled; it is a timing control, not a claim that the default buffered bridge
streams tokens. The real CUDA regression fixture passed with both dense and
256-page KV layouts, with the cancelled owner retired and its heterogeneous
replacement joining the survivor in a shared decode wave.
