# Resident compressed MLA, split attention and prefix cache

Measured source: `21bcfa7`. These files describe that exact build, including the
CLI/DLL hashes in [manifest.json](manifest.json). Later lifetime hardening requires
separate validation; these timings must not be attributed to a rebuilt DLL.

GLM-4.7-Flash Q4_K_M, Ryzen 7 9800X3D, RTX 4080 SUPER 16 GiB, Windows 11, CUDA 13.3.
One request at a time, configured/allocated native context 2048, managed cap 12 GiB
and free-device margin 256 MiB. Two long prompts use the same 72-note system prompt,
1554/1555 encoded tokens and 128 output tokens. Cycle 0 warms the
model; cycles 1/2 provide four long samples per mode. Each mode is a separate
process. No builds, native GPU tests or reference engines ran during timings.

## Fixed expert placement, cache disabled

| Mode | Prefill median [min,max] ms | Decode median [min,max] tok/s | HTTP median ms | Expert upload GiB/request | Cache hit rate |
|---|---:|---:|---:|---:|---:|
| baseline | 164426.0 [163384.0, 166403.0] | 6.54 [6.43, 6.59] | 183877.8 | 0.000 | disabled |
| mla | 104999.0 [103996.0, 105643.0] | 9.70 [9.47, 9.91] | 118280.6 | 0.000 | disabled |
| mla-split | 69554.5 [68596.0, 71312.0] | 22.22 [20.26, 22.57] | 75627.6 | 0.000 | disabled |
| mla-split-prefix | 313.5 [264.0, 321.0] | 21.94 [21.30, 22.52] | 6166.2 | 0.000 | disabled |

Fixed placement is partial under the cap: 30 routed layers use resident experts,
16 fall back. MLA keeps compressed KV, attention, router, shared FFN and head on
device; it retains CPU routed-FFN fallback. This does not make the oversized GLM
model entirely resident. Split attention composes exact tile softmax states.
Prefix mode restores immutable KV snapshots and recomputes the final token.

## Shared expert cache, 8192 MiB quota under the same device cap

| Mode | Prefill median [min,max] ms | Decode median [min,max] tok/s | HTTP median ms | Expert upload GiB/request | Cache hit rate |
|---|---:|---:|---:|---:|---:|
| baseline | 202450.0 [200809.0, 203430.0] | 6.50 [6.40, 6.54] | 222144.8 | 218.735 | 86.67% |
| mla | 153290.5 [151239.0, 156790.0] | 9.28 [9.26, 9.41] | 166997.7 | 219.301 | 86.64% |
| mla-split | 114852.0 [114294.0, 115657.0] | 20.60 [20.16, 20.98] | 121096.2 | 218.726 | 86.67% |
| mla-split-prefix | 433.5 [408.0, 456.0] | 20.36 [19.76, 21.42] | 6726.6 | 8.135 | 93.77% |

This quota shares the model's managed device cap with weights, KV, activations,
prefixes and scratch. Cached admission keeps selected expert groups leased until
the synchronous FFN completes. Misses retain original GGUF bytes and the original
router IDs/probabilities. CPU fallback applies when the selected group cannot fit.
The cache stays disabled by default; hit rate alone does not establish a benefit.

## Transfers and memory

Without the expert cache, logical CUDA API H2D/D2H traffic falls from about
12.99/12.21 GiB per long request to 0.257/0.259 GiB with MLA. Warm-prefix samples use
about 0.0204/0.0206 GiB. These are instrumented copy bytes, not a PCIe trace.
Per-mode device categories/peaks, RSS and whole-device samples are in the raw
reports. The ledger excludes opaque CUDA graph/driver/cuBLAS allocation; global
GPU samples also include other applications. Neither is an exact process VRAM
measurement. Retained TLS scratch can remain after model unload.

## Correctness and serving evidence

* Two ablations: 72 HTTP responses, 24 SSE comparisons and 8 explicit stop cases.
  Every optimized response matches its same-budget baseline exactly, including
  seeded sampling and penalties; SSE content matches non-streamed content and
  emits DONE. No replacement character appears in checked output.
* Two streaming suites:five cases each, including three client disconnect/replay
  cases, explicit stop and four simultaneous requests through a serialized runtime.
  This proves cancellation/recovery, not native multi-sequence batching.
* Lifecycle:fixed/cached/CPU-routed and legacy optional fallback each generate,
  unload, reload and generate identically in the same process. Legacy required
  MLA and insufficient state budget fail ready plus all six generation routes
  with 503. Successful unload releases model categories, apart from TLS scratch.
* Workspace: 253 passed, one ignored. Three synthetic native F64 oracle tests.
  Main real teacher forcing: 96 checked positions, 24 generations and six warm
  prefix reuses; worst KL 2.363e-11 and target NLL delta 1.526e-5. Tiny-cache CPU-routed
  validation: 64-token sequence, six positions, three generations and three warm
  prefix reuses; worst KL 1.780e-11 and target NLL delta 8.316e-6. Native regression:
  27 tests; real opt-in suites run separately rather than relying on guarded tests.

SSE TTFT is measured to the first nonempty content fragment after POST. Its
Unicode/sampling fixtures are separate from the two long-story timings. Prefix
SSE can reuse the entire previously queried prompt, so its TTFT must not be
interpreted as cold prefill of 1555 tokens. Responses are bounded to 128 tokens;
numerical parity does not establish broad factual accuracy or long-context quality.

The first cached streaming attempt passed all five response checks but failed a
harness assertion that treated MLA as partial Qwen. The included corrected
validation harness explicitly requires MLA split attention and validates the
tiny-cache CPU fallback separately. Both suites were rerun with the same frozen
CLI/DLL. Initial raw output and the assertion failure are retained.

## Reproduction and limits

[reproduce.ps1](reproduce.ps1) builds and validates the source at the checked-out
revision; to reproduce these hashes/timings, check out `21bcfa7` first. The script
requires the pinned local GGUF/tokenizer paths or explicit replacements. Raw
environment fields inherited from the older generic harness can name historical
builds; explicit row env, top-level executable/DLL hashes and this manifest take
precedence. Model content hashes are rechecked only after timing processes exit.

MLA is opt-in through `RBITNET_CUDA_MLA_FULL=1`; optional old DLLs fall back and
strict `RBITNET_REQUIRE_MLA_FULL=1` rejects unavailable pipelines. Split KV
and prefixes are independent options. No Ollama/llama.cpp parity claim is made
by these within-engine ablations. GPT dynamic experts, native block MLA prefill,
GPU paging/quantization, async prefetch and multi-sequence serving are later lots.

## Later lifetime hardening

The [separate manifest](safety-hardening/manifest.json) identifies the later safety
source commit and CLI/DLL hashes. The original timing tables above still belong
to `21bcfa7`; no complete performance ablation of this rebuilt DLL is claimed.

Native API cleanup now finishes queued copies and device work on every early
return, and ends/discards an abandoned capture before freeing caller buffers or
leases. A delayed pinned download and deliberately abandoned valid capture check
this cleanup without inducing a hardware fault. MLA rejects unfinished-token
outputs/checkpoints; monotonically assigned context generations reject a snapshot
that survives destruction and is offered to a newly created context.

This build passes 254 workspace tests (one ignored), Clippy, release/CUDA builds,
28 native regression tests, three synthetic F64 oracles and the two real GLM
teacher/sampling/prefix suites with the same KL/NLL bounds. All six lifecycle
cases and both five-case streaming suites were rerun on this exact safety build.
This includes tiny-cache all-CPU routed FFNs, client disconnect/recovery, stops
and four simultaneous requests handled by the serialized runtime.

The tiny-cache streaming invocation initially could not copy its executable
because E: was full; no server/model started. The failure log is retained. Only
regenerable untracked debug PDBs were removed, then this one suite was rerun with
the same executable, DLL and source hashes. Source, models and raw results were
retained. No hardware fault recovery or native batching is established by these
tests.
