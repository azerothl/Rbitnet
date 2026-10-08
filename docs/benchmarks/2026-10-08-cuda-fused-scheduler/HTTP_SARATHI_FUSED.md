# HTTP Sarathi fused Llama — synchronized 1/4/8 probe

**Date:** 2026-10-08  
**Revision:** `6e04b805c90b0b5762ef0a981f88745d2d162236`  
**Purpose:** establish that the HTTP rendezvous reaches the CUDA Llama fused
multi-row path. This is one probe per concurrency, not a performance claim or
an ablation.

## Fixture and configuration

- GPU: NVIDIA GeForce RTX 4080 SUPER, 16376 MiB; driver 610.88.
- Model: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`,
  SHA-256 `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.
- Tokenizer: `tokenizer.json`,
  SHA-256 `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`.
- Prompt: `Explain continuous batching in one short paragraph.`
- Request: OpenAI-compatible `/v1/chat/completions`, `temperature: 0`,
  `max_tokens: 32`; every successful response reported 32 completion tokens.
- Server: a release `bitnet-server` binary built from the revision above on
  2026-10-08. Each row starts a fresh server after readiness.
- Flags: `RBITNET_BACKEND=cuda`, `RBITNET_CUDA_PREFILL=1`,
  `RBITNET_CUDA_KV_FORMAT=f32`, `RBITNET_CUDA_KV_PAGE_LIMIT=128`,
  `RBITNET_CONTINUOUS_BATCHING=1`, `RBITNET_FUSED_MULTI_SEQ=1`,
  `RBITNET_CUDA_FUSED_DECODE_SLOTS=8`, `RBITNET_CUDA_CONTINUOUS=0`.

The page limit is material: leaving the dense, model-maximum KV allocation in
place made the synchronized four-request wave fail with `fused decode KV
allocation refused` on this 16 GiB device. It is therefore part of this
fixture, not a general default recommendation.

For concurrent rows, already-created in-process HTTP clients waited on one
manual-reset gate before issuing their POSTs. This prevents process/job startup
skew from missing the server's two-millisecond coalescing window. The existing
shell harness was adapted operationally for this Windows host; it did not
persist results automatically.

## Raw results

`requested_tok/s` is `concurrency × 32 / wall_s`; it is requested aggregate
output rate, not measured decode rate or a per-conversation speed.

| concurrency | wall_s | requested_tok/s | batch rows delta | batch waves delta | projection delta | scheduler decode waves delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.539 | 59.369 | 0 | 0 | 0 | 0 |
| 4 | 8.120 | 15.764 | 176 | 44 | 4929 | 129 |
| 8 | 10.574 | 24.210 | 220 | 44 | 4929 | 161 |

The 4- and 8-request rows are real HTTP 200 responses and have
`batch_rows_delta > batch_waves_delta`, which proves multi-row fused waves
were reached. The singleton result correctly has no multi-row wave.

## Interpretation and limits

This probe establishes routing and observability for the bounded Llama CUDA
vertical only. It does **not** demonstrate a throughput win: its single
samples are not repeated, have no fused-off/control comparison, and do not
measure client TTFT, per-request completion latency, or inter-token latency
dispersion. In particular, the requested aggregate rates at 4 and 8 are lower
than the singleton row in this run and must not be promoted as an acceleration
claim.

The first unsynchronized job-based attempt also completed HTTP requests but
recorded zero fused row/wave deltas, showing that it missed the short
rendezvous window. It is excluded from the table above. The synchronized
attempt with the dense KV default failed rather than falling back silently;
the bounded-page run above is the accepted evidence.

See [the #96 product decision](../../FUSED_MULTI_SEQ_DECISION.md) for the
shipping boundary and remaining acceptance work.
