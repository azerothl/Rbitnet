# First combined-stack CPU/GPU comparison

This is a complete capture of four models, three engines and CPU/GPU modes
(24 rows), using the previously sealed combined binaries. It is not a claim
that every requested optional Native path ran.

## Llama configuration error retained

The Llama GPU row sets `RBITNET_LLAMA_PAGED_KV=1`, which selects host KV pages
and disables full Native residency. `RBITNET_REQUIRE_RESIDENT` has no implemented
consumer in this revision. Graph replays and split-attention counters are zero.
The observed 80.05 tokens/s therefore describe the host-paged hybrid path, not
Native paged CUDA. No result is replaced or relabelled as Native. A separately
queued dense/Native-paged comparison uses `RBITNET_CUDA_KV_PAGE_LIMIT` and requires
positive graph-replay and split-attention counters.

## Measured median decode tokens/s

| Model | Backend | Rbitnet | Ollama | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| llama32-1b | CPU | 32.32 | 46.13 | 43.74 |
| llama32-1b | GPU / host-paged hybrid for Rbitnet | 80.05 | 499.25 | 463.84 |
| qwen35-2b | CPU | 14.07 | 17.97 | 18.47 |
| qwen35-2b | GPU | 231.88 | 222.85 | 224.41 |
| gpt-oss-20b | CPU | 10.06 | 15.72 | 15.47 |
| gpt-oss-20b | GPU | 90.59 | 186.83 | 191.78 |
| glm47-flash | CPU | 5.69 | 13.03 | 12.36 |
| glm47-flash | GPU | 22.41 | 75.48 | 69.25 |

## Protocol and interpretation

One client, context capacity 2048, 16 threads, 128 generated tokens maximum,
one excluded warmup and three measured prompts. The long prompt contains 24
notes; exact prompt IDs/counts are verified and retained in the captures.
GGUF/tokenizer hashes, commands, engine versions, counter snapshots, responses,
streaming probes, memory observations and prefill/wall/decode times are included.
Rbitnet and llama.cpp use F32 KV; Ollama uses F16 KV. The KV precision is not
identical. GPU memory observations are global to the machine.

All 24 rows completed without execution errors. Llama/GPT-OSS/GLM pass three
short checks; Qwen passes two in every engine/backend. These probes do not
establish general answer quality. Rbitnet Qwen GPU is locally competitive on
this protocol; GPT-OSS and GLM remain slower. GLM Rbitnet GPU measured rates
range from 8.56 to 23.17 tokens/s: the dispersion must remain visible.
CPU Rbitnet remains slower than both references on these four models.

Only NVIDIA on this Windows machine is covered. A recorded completed run is
distinct from validation of each intended feature or overall engine parity.
Original local paths need adaptation to reproduce the harness.
