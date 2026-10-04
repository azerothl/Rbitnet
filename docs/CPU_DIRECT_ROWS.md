# Optional direct CPU output bands

`RBITNET_CPU_DIRECT_ROWS=1` lets quantized matrix-vector workers write directly
into disjoint bands of the final output. It removes temporary band vectors,
gathering, sorting and the final copy. Each row uses the existing dot kernel,
F32 activations and accumulation order. The default remains disabled.

An isolated same-binary comparison on the Ryzen 7 9800X3D used 16 threads,
context capacity 512, one warmup and two measured cycles, with 128 generated
tokens per request. Rates below are tokens divided by decode-time counter
deltas. The differenced instantaneous TPS gauge in the raw capture is not a
rate and must not be used for this comparison.

| Model | Story original/direct tokens/s | Code original/direct tokens/s |
| --- | ---: | ---: |
| Llama 3.2 1B | 32.41 / 34.40 | 29.93 / 31.15 |
| Qwen 3.5 2B | 15.08 / 14.63 | 14.56 / 14.80 |
| GPT-OSS 20B | 11.06 / 11.24 | 11.05 / 11.15 |
| GLM 4.7 Flash | 7.19 / 7.12 | 7.07 / 7.24 |

The completed private experiment compared 240 synthetic row cases and 1,200
actual GGUF row cases bit for bit across five thread/SIMD configurations.
It also matched 72 ordinary responses, 16 SSE responses and eight stop cases
across the four CPU models. The private build predates the finish-reason base
of this branch; these results do not validate the current public candidate.
Fresh compilation and serving proof on this source remain pending.

Two measured cycles without counterbalancing do not establish a reliable
general gain. Llama improved by approximately 4–6% on these prompts; other
models ranged from -3% to +2.4%. This is an opt-in experiment, with no Ollama
or llama.cpp parity claim, no CPU multi-token GEMM implementation and no
change to activation quantization. A separate counter,
`rbitnet_core_cpu_direct_row_calls_total`, reports use of the path.
