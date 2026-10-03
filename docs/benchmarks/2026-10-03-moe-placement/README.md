# Routed FFN placement and mapped GGUF weights — 3 October 2026

This lot adds mapped host backing, weakly registered per-layer MoE counters, and opt-in CPU/adaptive whole-FFN policies. `cache` remains the default. CPU expert mode retains the placement of attention, head and shared FFNs; it is not an entirely CPU model.

Production validation: 265 workspace tests passed, one ignored, Clippy/release passed; 3 mapped tests and 4 transfer/lifetime tests executed, with the actual CUDA fixture enabled. The native suite has 32 passing tests, including optional tests whose model-specific flags were not enabled. Five actual GPT-OSS/GLM budget cases compare 240 observed positions and 120 generated/replayed outputs plus 30 reference outputs; worst KL 1.851e-10, absolute target-NLL delta 3.815e-05, all observed argmaxes and tested generations agree.

Four live CPU/adaptive model suites verify scoped metrics, output, unload/reload, disconnection/resume, seeded sampling, penalties, explicit stop and four concurrent requests on the serialized executor. After unload there are no model counter owners and no remaining managed weight/KV/expert/prefix allocations; reload gets a different model_id.

Same-build policy ablations use identical requests, a 12 GiB managed cap, 2048 capacity, 24 common notes and maximum 128 output tokens. Cycle 0 warms up; medians below cover four long observations (two prompts × two measured cycles). Raw JSON retains all requests, output token counts, timings, model counters and memory; exact strings and HTTP/SSE output were checked across policies. Previous/current binary comparisons use the same harness, DLL and requests; their build manifests distinguish source provenance. This is a smaller prompt corpus than the earlier 72-note reports.

| Model / policy / cache | Long prefill median ms | Decode median tok/s | HTTP median ms | Peak process tree RSS GiB |
|---|---:|---:|---:|---:|
| glm47-flash-adaptive-cache0 | 24716.5 | 21.26 | 30785.4 | 17.07 |
| glm47-flash-adaptive-cache8192 | 26743.5 | 18.26 | 33753.9 | 16.73 |
| glm47-flash-cache-cache0 | 24535.5 | 21.51 | 30511.0 | 17.07 |
| glm47-flash-cache-cache8192 | 31659.0 | 24.48 | 37025.0 | 16.73 |
| glm47-flash-cpu-cache0 | 50103.5 | 10.87 | 61884.7 | 16.71 |
| glm47-flash-previous-cache0 | 24540.0 | 21.27 | 30613.7 | 28.54 |
| gpt-oss-20b-adaptive-cache0 | 6579.5 | 78.19 | 8222.4 | 10.95 |
| gpt-oss-20b-adaptive-cache8192 | 8957.5 | 50.42 | 11575.0 | 10.29 |
| gpt-oss-20b-cache-cache0 | 6382.5 | 81.22 | 7973.3 | 10.87 |
| gpt-oss-20b-cache-cache8192 | 6570.5 | 75.74 | 8288.3 | 10.29 |
| gpt-oss-20b-cpu-cache0 | 43234.0 | 13.90 | 52485.7 | 10.28 |
| gpt-oss-20b-previous-cache0 | 6330.0 | 80.43 | 7938.4 | 21.28 |

Matched mapped-backing comparison (same requests, policy, native DLL and managed cap):

| Model | Previous peak RSS GiB | Current peak RSS GiB | Peak RSS reduction | Current/previous decode |
|---|---:|---:|---:|---:|
| gpt-oss-20b | 21.28 | 10.87 | 48.9% | 1.010 |
| glm47-flash | 28.54 | 17.07 | 40.2% | 1.011 |

Policy interpretation is limited to these two measured cycles and these requests. Adaptive execution remains opt-in. In particular, a correct CPU/GPU decision mechanism does not by itself establish a throughput gain. Ratios below compare adaptive execution to the corresponding same-build cache policy (prefill below 1 and decode above 1 are favorable):

| Model / cache MiB | Adaptive/cache prefill | Adaptive/cache decode |
|---|---:|---:|
| gpt-oss-20b / 0 | 1.031 | 0.963 |
| gpt-oss-20b / 8192 | 1.363 | 0.666 |
| glm47-flash / 0 | 1.007 | 0.988 |
| glm47-flash / 8192 | 0.845 | 0.746 |

The policy decision is per layer for the entire selected routed FFN. GPU allocation/fill calibration, warm-cache locality and CPU parallel work can change its costs; these data do not justify enabling the selector for every model. A 0 MiB cache retains fixed placement. Explicit CPU mode ignores the expert cache budget and avoids placing unused expert banks. The mapped backing avoids a second owned host copy in this shared planner, while mapped pages remain physically resident when touched.

This report does not implement async overlap (#84), mixed per-expert execution (#86), device KV pages/formats (#92/#93), persistent KV tiers (#94), true multi-sequence forwards (#96), or a real GGUF draft (#97). The ignored block/fusion prototypes are outside this build and outside these measurements.

Reproduction: [serialized checks](scripts/validate_cost.py), [live suites](scripts/live.py), [policy benchmark](scripts/benchmark.py), [previous/current benchmark](scripts/benchmark_previous.py). [Manifest](manifest.json) stores distributions, source/binary hashes, original and published LF-normalized capture hashes. Initial empty ownership-filter output is preserved separately; only the subsequent four-test run is counted.
