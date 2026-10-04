# Optional CUDA expert arena

`RBITNET_MOE_ARENA=1` places the asynchronous GPT-OSS/GLM expert slots in one
managed CUDA allocation. It is disabled by default and requires
`RBITNET_MOE_ASYNC=1`, CUDA/hybrid execution and `RBITNET_MOE_EXECUTION=cache`.
The existing expert cache and device budgets still apply.

Each gate/up/down projection has a disjoint, 256-byte-aligned view. The layout
reserves the largest original quantized span for that projection across layers;
it does not requantize weights. The padding is charged before admission and may
reduce the number of usable groups. A view owns a lease on the whole allocation:
dropping a cache or model cannot free bytes retained by an external view.
Publication still waits for the copy event. Refills reuse the same slots, while
the existing pinned-buffer, eviction and poison/drain rules remain in effect.

This is fixed placement, without dynamic compaction or learned expert prediction.
It is a capacity option, not an established improvement in steady-state decode.

## Observed measurements

The [captured comparison](benchmarks/2026-10-04-expert-arena/README.md) used an
RTX 4080 SUPER, the same executable/library, context 2048, demand-only copies,
four system notes and up to 128 generated tokens. Each cell has one warm cycle
and two measured cycles. Rates below are medians for the story prompt.

| Model | Expert budget MiB | Sync tok/s | Async tok/s | Arena tok/s |
|---|---:|---:|---:|---:|
| GPT-OSS-20B | 512 | 8.83 | 8.14 | 8.13 |
| GPT-OSS-20B | 8192 | 75.08 | 75.41 | 76.37 |
| GLM-4.7-Flash | 512 | 5.24 | 8.52 | 8.47 |
| GLM-4.7-Flash | 8192 | 26.98 | 31.12 | 31.09 |

The large GLM improvement comes from the asynchronous pool. The incremental
arena effect on throughput is small in this capture. GPT-OSS at 512 MiB is slower
with either asynchronous variant than with the synchronous baseline.

At an 8192 MiB expert budget, sampled global GPU peak minus baseline changed
from 13170 to 9674 MiB for asynchronous GPT-OSS and from 13605 to 10569 MiB for
asynchronous GLM. The managed requested-byte peak stayed almost unchanged:
GPT-OSS increased by 249216 padding bytes; GLM was identical. Total managed CUDA
allocation calls fell from 2592 to 646 for GPT-OSS and from 5353 to 1145 for GLM.
These counters cover the full model/requests, not just the arena allocation.

Global GPU sampling is not process VRAM measurement. The observation does not
prove a specific driver allocation granularity or guarantee those savings on
another device. Two timing repeats do not establish statistical confidence, and
this comparison does not establish parity with Ollama or llama.cpp.

## Validation and limits

The isolated implementation passed workspace checks/tests, an actual CUDA
single-allocation/last-lease fixture, mixed Q4/Q6 cache refills and poison/drain
checks on both GGUF files. Full-model logits/prefix/generation/lifetime fixtures
passed at both cache budgets; worst measured KL was 2.363e-11 and worst absolute
target NLL delta 2.289e-5 against the fixtures' reference path.

All 108 quiet JSON responses, 36 SSE responses and 12 stop probes matched their
baseline controls. Both models also passed actual disconnect/resume tests for
greedy, seeded sampling and penalties, plus explicit stop and serialized
concurrency. This evidence uses the isolated candidate on the earlier source
base. Fresh validation of this public checkout, including observed EOS reasons,
is queued separately; it must not be inferred from the isolated result.

The seven implementation files match the measured candidate. Native kernel
content is unchanged from the source base after newline normalization. CPU,
ROCm, Intel and Metal arena execution are not covered. Combining the arena with
the other pending feature branches requires a separate integrated build/test.
