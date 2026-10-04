# Exact adaptive nucleus sampling: fresh validation

Compiled source: `cebe50dde577781100dd7861bb6ae71699e13501`.
`RBITNET_CPU_TOP_P_HEAP=1` remains optional; the default remains unchanged.
This preserves the stable token ordering, F32 accumulation and RNG draw of
the original top-p sampler, with at most 64 heap pops before sort fallback.

298 workspace tests pass, one network test is ignored. Check, Clippy, fresh
CUDA library and release CLI pass. 2,028 synthetic cases and 864 draws from
24 actual full-vocabulary GPT-OSS logit vectors preserve token and RNG state.
96 HTTP/SSE observations cover Llama, Qwen3.5, GPT-OSS and GLM, CPU and CUDA,
seeded sampling, penalties, greedy/zero budgets; baseline/heap outputs and
usage agree exactly. This is bounded equality evidence, not general quality.

## Single-client GPU HTTP wall tokens/s

Two writing prompts, top-p 0.9, temperature 0.7, seed 42, maximum 128 output
tokens; one excluded warmup and two measured samples per prompt. Rates below
are medians of the two individual output-token/wall-time rates. These rates
include prefill and HTTP overhead. They are not decode-only rates.

| Model | Prompt | Original | Heap | Change |
| --- | --- | ---: | ---: | ---: |
| llama32-1b | Write a detailed story… | 116.26 | 212.99 | +83.20% |
| llama32-1b | Describe a future muse… | 116.51 | 199.19 | +70.96% |
| qwen35-2b | Write a detailed story… | 60.67 | 117.92 | +94.37% |
| qwen35-2b | Describe a future muse… | 64.19 | 115.87 | +80.51% |
| gpt-oss-20b | Write a detailed story… | 43.63 | 59.67 | +36.75% |
| gpt-oss-20b | Describe a future muse… | 45.11 | 59.70 | +32.34% |
| glm47-flash | Write a detailed story… | 19.03 | 20.50 | +7.77% |
| glm47-flash | Describe a future muse… | 18.58 | 19.47 | +4.83% |

## Adoption limits

The optional sampler has local end-to-end benefit on these four workloads.
Keep it opt-in until more distributions and interleaved repeated timings are
covered. The order here is original then heap, not randomized/interleaved;
two measured samples per cell cannot establish confidence intervals.
Pure selection on eight actual vector cells reduced cost about 68–70%, but
uniform/concentrated synthetic fallback cells include increases up to about
13%. Do not describe it as universally faster. An earlier unguarded heap
prototype regressed on wide distributions; that separate attempt is retained
locally and is not the source validated in this report.

No model weights, binaries or full float vector archives are committed. Their
hashes and input index remain in the original manifest and supplied harness.
Local orchestration paths must be adapted. Only Windows/NVIDIA is exercised;
there is no Ollama/llama.cpp comparison or long-context claim in this ablation.
