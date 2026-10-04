# Fresh public-checkout Llama continuous serving validation

The Native DLL and CLI were rebuilt from public branch commit
`3d4f327676d1f43cf2b76ab4e486ec269fcb1fbf`, then tested on the actual
Llama-3.2-1B model with an RTX 4080 SUPER and Ryzen 7 9800X3D.
This fresh run supersedes the pending-build qualification in the earlier
[private source-equivalent evidence](../2026-10-04-continuous-llama/README.md).

## Concurrent HTTP measurements

Each wave has eight separate streamed requests and 798 generated tokens.
There is one warmup and two measured cycles per configuration, with alternate
clients arriving 20 ms later. Context capacity is 2048, F32 KV, ordinary
resident graphs enabled, split KV/TF32/prefix disabled. The exact prompts,
responses, request timings and counter snapshots are in `raw/live/results.json.gz`.

| Configuration | Median aggregate tokens/s | Median wave time (ms) | Change |
|---|---:|---:|---:|
| reference | 140.62 | 5674.96 | +0.0% |
| dense-slots1 | 137.73 | 5794.11 | -2.1% |
| dense-slots4 | 169.58 | 4705.76 | +20.6% |
| dense-slots8 | 186.81 | 4271.74 | +32.9% |
| paged-slots1 | 134.77 | 5921.49 | -4.2% |
| paged-slots4 | 167.59 | 4761.81 | +19.2% |
| paged-slots8 | 185.36 | 4305.17 | +31.8% |

These are aggregate HTTP rates, including prefill, CPU sampling, arrival delay
and streaming. They are not per-request decoding rates. Two measured cycles
do not establish statistical significance or parity with another engine.
Single-slot overhead remains a reason to keep the option disabled by default.
SSE chunk intervals are preserved without treating chunks as individual tokens.

## Validation

Workspace check, Clippy and tests passed (298 passed, one ignored).
The rebuilt Native driver passed 24 actual-model configurations and two thread
ownership/shutdown fixtures. Actual HTTP validation passed 54 serial JSON/SSE
identity records, six disconnect/survivor cases, six explicit-stop cases and
21 concurrent waves. Independent serial references preserve per-request RNG,
text and finish reason. The four-model ordinary-serving finish suite also passed.

`receipt.json` indexes 34 original gzip captures and exact helper hashes.
`raw/manifest.json.gz` binds all compiled sources, commands, executable/DLL
SHA-256 identities and the original proof commit. Full binaries and models
are omitted. Working-source bytes are bound separately from Git line endings.
The Native logs retain per-configuration scheduling traces and ownership results.
The orchestration checker uses local predecessor proof paths; the standalone
HTTP harness accepts explicit candidate/reference executables and DLLs as
documented in the earlier evidence directory.

Scope remains opt-in CUDA Llama F32, dense/paged, one to eight owners.
Shared prefill, other architectures, encoded KV and final combined-stack
comparison remain outside this change. Issue #96 stays open.
