# Exact GPT-OSS block prefill and optional MoE fusion — 4 October 2026

GPT fixed resident banks can now prefill up to 32 positions in one causal block. Projections consume multiple input rows per launch; routed work is grouped by expert and preserves selected-slot order, biases, OAI activation and original GGUF quantized bytes. Attention includes the original alternating windows and sinks. A native verification API returns all-position logits or greedy IDs; ordinary generation returns only its needed final output.

Production validation: 270 workspace tests passed, one ignored; Clippy and release passed with the existing warnings. Four explicit CUDA suites compare ordered projections, grouped experts, fusion and block forwarding against original GPU kernels and independently decoded F64 references. The actual GPT-OSS model compares 120 observed positions, 60 generated/replayed outputs and six references across 16/32 ordered blocks, 32 tiled blocks and optional fusion: observed KL and target-NLL differences are zero. Seeded sampling, penalties, exact prefix replay and cancellation/recovery pass.

The production ablation uses the same binary and DLL for every mode: 2048 configured capacity, 12 GiB managed budget, 24 common notes, maximum 128 output tokens. Cycle 0 warms up; distributions below cover two long prompts in cycles 1 and 2. Short prompts, original requests/responses, first visible SSE times, absolute managed gauges, model counters and process working set are retained in the raw captures. Production has 36 protocol observations, 12 unary/SSE pairs and four explicit stops; its additional network suite checks disconnect/resume in three sampling modes, stop, and four simultaneous requests on the serialized executor.

| Production mode | Long prefill median ms | Decode median tok/s | HTTP median ms | Peak RSS GiB | Managed peak MiB |
|---|---:|---:|---:|---:|---:|
| serial | 5605.5 | 90.78 | 7026.1 | 10.88 | 10891.3 |
| block16 | 2000.0 | 90.05 | 3429.4 | 10.90 | 10908.9 |
| block32 | 1926.5 | 90.11 | 3350.1 | 10.90 | 10926.6 |
| block16-prefix | 12.0 | 90.65 | 1437.3 | 10.90 | 11021.6 |

Ordered 32-token blocks reduce long prefill by 2.91× and HTTP time by 2.10× in this corpus. Decode speed is close to serial and slightly lower in several observations; this is a prefill gain. Warm prefix medians reuse almost all input state and cannot describe cold requests. Block capacity defaults to 16 and remains configurable; all options remain opt-in.

Separate five-mode prototype experiment (same requests; independently frozen binary/library):

| Prototype mode | Long prefill median ms | Decode median tok/s | HTTP median ms |
|---|---:|---:|---:|
| serial | 5538.0 | 91.89 | 6931.7 |
| fused | 5580.0 | 91.23 | 6989.6 |
| block16 | 2006.5 | 89.76 | 3433.8 |
| tile32 | 3998.5 | 89.76 | 5438.1 |
| tile32-prefix | 26.5 | 90.08 | 1450.9 |

The 32-token shared-byte tiled configuration is slower than ordered blocks; fusion is close to serial. Neither is promoted to a default. Three additional prototype network suites exercise ordered blocks, tiled blocks and fusion. The early prototype greedy mismatch is preserved in kernel-validation; rebuilding from the current matching sources removed it, but its individual cause was not isolated. Empty/inactive early filters are preserved and are not counted as GPU proof.

An additional integration fixture brings the workspace to 271 passing tests (one ignored). Its warmed Nsight capture confirms multi-token ordered-projection grids and joint routed-token grouped-expert kernels; the child completion nonce proves that the generation finished and matched its warm reference. Full grid/count records and capture hashes are in [trace analysis](profile/trace-analysis.json). Profiler timings are excluded from throughput measurements.

Limits: fixed resident GPT expert banks only; cache/partial placement and CPU/adaptive routing keep their serial segments. GLM/MLA block forwarding, true multi-sequence execution, a real draft model and reduced-precision KV are separate work. Fusion changes launch structure and does not currently reclaim its retained intermediate buffers. No new comparison with Ollama or llama.cpp is made.

Reproduce with [production checks](scripts/check_gpt_block_production.py), [quiet/network measurements](scripts/measure_gpt_block_production.py) and the copied benchmark/streaming harnesses. [Manifest](manifest.json) records distributions, source/binary/library hashes and original/published LF-normalized capture hashes.
