# Measured Llama session tiers for #94

Date: 2026-10-08. These are real local HTTP measurements on `origin/main`
`9eb343efbd5f29fff222c20b4b7bff416c6d55f4` (#163), not extrapolations.

## Result

| path | TTFT samples (ms) | median | sample stdev | whole non-stream request median |
| --- | --- | ---: | ---: | ---: |
| cold recompute | 2356, 2367, 2342 | 2356 | 12.530 | 2368.176 ms |
| RAM restore | 13, 12, 13 | 13 | 0.577 | 35.529 ms |
| SSD restore | 50, 52, 51 | 51 | 1.000 | 830.376 ms |

For this 676-token Llama prompt, the instrumented first-token time falls
**99.45%** from cold recompute to RAM restore and **97.84%** to SSD restore.
Each restored response was the identical greedy text, `I see`, and restored
675 prompt tokens. RAM requests each reported one RAM hit; every SSD sample
used a fresh server process and reported exactly one disk hit.

TTFT is Rbitnet's `rbitnet_inference_ttft_ms_sum` delta: the model was already
loaded and the HTTP health endpoint ready, so it excludes process startup and
model loading. The request column is important: this test used non-streaming
HTTP, and the SSD response's full wall time still includes process-local work
after the instrumented first-token boundary. It is not a client-observed
streaming TTFT claim.

## Protocol and raw artifacts

- Llama 3.2 1B Instruct Q4_K_M, SHA-256
  `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`.
- Matching `tokenizer.json`, Native CUDA, resident graph and a rebuilt server
  from the revision above; hashes and complete per-request metrics are in
  [`results.json`](results.json).
- Prompt: the same 676-token rendered user message; temperature zero; two
  generated tokens; one excluded short warm-up before cold sampling; three
  samples per tier.
- RAM uses the process that captured the checkpoint. SSD preserves the sealed
  object, stops the writer, then starts a fresh process for each sample.
- The sealed checkpoint was 44,240,870 bytes. SSD reads measured 33.794,
  35.433 and 34.288 ms. Process RSS after requests was about 1.89 GiB cold,
  1.89 GiB RAM and 1.89 GiB SSD; managed CUDA memory was 1,337,361,460 bytes.
  The OS VRAM metric is unavailable, so the latter is Rbitnet's allocation
  accounting, not total board usage.
- [`measure_llama_session_tiers.py`](measure_llama_session_tiers.py) runs the
  experiment and [`raw/`](raw) preserves requests, metrics and server logs.

## Scope and limitations

This validates the Llama Native F32 prefix-checkpoint transport only. Qwen was
not rerun: the available 0.8B GGUF has no colocated tokenizer bundle on this
host. GPT-OSS and GLM/MLA remain unsupported by the context-tier transport;
they have no export/import ABI or restore integration.

This is a prefix key, not a proxy session identity. It does not prove
session-isolated promotion/eviction, proxy runner recycling, idle-unload
restore, HTTP/SSE cancellation or stop-string equivalence. The test also does
not cover Windows physical ENOSPC or timed write/rename crash recovery.

The measured Llama recompute/RAM/SSD latency slice is one input to the
accepted bounded Llama/Qwen product decision. Its unsupported architecture,
proxy, Qwen-rerun and failure-mode limits remain explicit in
[the decision](../../SESSION_TIERS_DECISION.md).
