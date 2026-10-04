# Continuous Native Llama: actual HTTP evidence

These measurements used an RTX 4080 SUPER, a Ryzen 7 9800X3D and the actual
Llama-3.2-1B GGUF/tokenizer from the existing four-engine benchmark manifest.
The implementation source content in this branch matches the successful private
build. A fresh build from the public checkout is still pending; the PR remains
a draft for that validation. The raw binary hashes identify the executed
artifacts, which are intentionally not stored in Git.

## Eight concurrent HTTP clients

Each wave contains eight separate streamed requests, with alternate clients
arriving 20 ms later. There are three cycles per configuration: one warmup and
two measured cycles. Requests cover long story/code prompts, seeded sampling,
frequency/presence penalties and a factual answer reaching actual EOS after
15 output tokens. Two requests finish at EOS and six exhaust a 128-token budget.
The wave contains 798 generated tokens. Context capacity is 2048, F32 KV,
split KV off, TF32 off, prefix reuse off and ordinary resident graphs enabled.

| Configuration | Median aggregate tokens/s | Median wave time | Change from ordinary serving |
|---|---:|---:|---:|
| Ordinary serialized reference | 148.69 | 5366.95 ms | — |
| Dense, 1 slot | 141.30 | 5648.09 ms | -5.0% |
| Dense, 4 slots | 177.58 | 4493.84 ms | +19.4% |
| Dense, 8 slots | 195.93 | 4073.06 ms | +31.8% |
| Paged, 1 slot | 138.90 | 5745.26 ms | -6.6% |
| Paged, 4 slots | 175.02 | 4559.56 ms | +17.7% |
| Paged, 8 slots | 195.43 | 4083.39 ms | +31.4% |

These are aggregate HTTP completion rates, including CPU sampling, prefilling,
arrival delays and streaming. They are not per-request decode rates. Two
measured cycles do not establish statistical significance. Client SSE chunk
intervals are preserved separately and are not assumed to be token intervals.
No Ollama or llama.cpp comparison was run for this scenario.

## Correctness and ownership

The actual model driver passed 24 configurations: dense/paged, capacity 1/4/8,
graphs off/on and both matrix-row orderings. Each request has an independent
serial Native reference with its own RNG. Tests cover variable arrivals,
departures, nonempty EOS, zero-token requests, partial-prefill cancellation and
admission refusal. Two additional thread fixtures cover shared rows, a surviving
request after another owner disconnects, and exact managed-memory categories
after shutdown.

The HTTP suite passed 66 records and 21 waves. The records contain 54 serial
JSON/SSE identity cases, six disconnect/survivor/resumption cases and six
explicit stop-string cases. Every concurrent request matches its serial
reference's content and finish reason. Completion token counts are checked in
the serial JSON references. This is bounded output validation for one actual
Llama model, not a general model-quality benchmark.

The independent Native batch study and its traces are retained separately under
`raw/llama-batch-proof`. That fixed-step study continues through EOS and omits
HTTP/CPU sampling overhead. Its larger kernel-level speedups must not be
substituted for the HTTP rates above. Profiling timings are excluded from the
quiet HTTP table. The dense admission bug and original EOS-budget failure were
preserved privately before their corrections and are not reported as passes.

## Provenance and use

`summary.json` contains the values behind the table. `receipt.json` binds the
public source files to the executed private sources, records the published Git
blob hashes separately from raw build hashes, and indexes every raw gzip
object with its original SHA-256. `SHA256.json` checks all published objects.
Gzip compression preserves the original bytes, including Windows line endings.
Git's ordinary CRLF/LF conversion is the only difference between published source
content and corresponding raw build files.
The original private checkers are included for provenance and depend on private
proof directories; they are not standalone public benchmark entry points.

The standalone public HTTP harness can be run from the repository root after
building the CLI and Native CUDA library, using an ordinary reference build
that includes the finish-reason fixes:

```powershell
python -B docs/benchmarks/2026-10-04-continuous-llama/harness/llama_continuous_live.py `
  --binary <candidate-rbitnet.exe> --library <candidate-native.dll> `
  --reference-binary <reference-rbitnet.exe> --reference-library <reference-native.dll> `
  --output <new-empty-evidence-directory>
```

The model paths in `docs/benchmarks/2026-10-03-parity-round2/manifest.json` must
exist locally. See [configuration and limits](../../CONTINUOUS_LLAMA.md).
Issue #96 remains open for other architectures, shared prefill and final
integration/comparison. This change does not implement general serving parity.
