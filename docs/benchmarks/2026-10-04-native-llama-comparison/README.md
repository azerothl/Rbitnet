# Corrected Native Llama CUDA comparison

This separately recorded run uses the sealed combined-stack executable/DLL
at source `f1257e5bd564543fa3b46a31acf351260d188f9c`. The earlier host-paged
hybrid capture remains preserved in the first four-model report. Its 80.05
tokens/s row did not exercise Native Llama; it must not be treated as a Native
regression or plotted as a like-for-like optimization gain.

Both Rbitnet profiles disable host paging and the host KV pool, enable split
attention and Native graphs, and use F32 KV. Dense sets page limit zero; Native
paged sets 256 physical pages. Both actually record positive graph replay and
split-attention counters (529 and 58,672 in each final capture).

## Median measurements

| Profile | Engine | Decode tokens/s | HTTP wall ms | Prefill ms |
| --- | --- | ---: | ---: | ---: |
| dense | llama.cpp | 470.40 | 309.26 | 18.45 |
| dense | ollama | 503.86 | 319.60 | 59.92 |
| dense | rbitnet | 372.09 | 881.29 | 525.00 |
| paged | llama.cpp | 467.40 | 296.03 | 18.19 |
| paged | ollama | 500.98 | 332.31 | 59.49 |
| paged | rbitnet | 386.71 | 864.47 | 527.00 |

## What this establishes

Native Rbitnet decode reaches approximately 372 dense / 387 paged tokens/s,
below llama.cpp 467–470 and Ollama 501–504 on this protocol. Its total response
time is about 864–881 ms versus 296–332 ms for the references. Prefill remains
a substantial part of the gap. The two Rbitnet runs are separate sequential
captures; the small difference does not establish a statistically robust page
speed advantage. Native pages also serve capacity/sharing goals independently
of single-client speed.

One client, context 2048, 16 threads, maximum 128 outputs, 24-note long prompt,
one excluded warmup and three different measured prompts. Exact prompt counts
are verified; complete outputs and counters, quality/streaming probes, versions,
commands, model identities and source hashes are retained. The three short
quality probes pass in all rows; this is not a general quality assessment.
Rbitnet/llama.cpp use F32 KV; Ollama uses F16. Dtypes are not identical.
GPU memory measurements are global; only this Windows/NVIDIA machine is covered.
Original local paths need adaptation. No CPU or MoE result is replaced here.
