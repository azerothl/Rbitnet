# Q8 reuse: correct, faster draft verification, slower than direct decoding

On this RTX 4080 SUPER, sharing Q8 decoded weights across four ordered token
projections improves speculative throughput on the two measured writing prompts.
It does **not** make the tested Qwen3.5-0.8B draft / Qwen3.5-2B target pair faster
than Qwen3.5-2B alone. Default speculative adoption remains a **no-go**.

| Mode | Prompt | Tile 0 tok/s | Tile 4 tok/s | Relative change |
|---|---:|---:|---:|---:|
| Target only | 0 | 225.95 | 225.56 | -0.18% |
| Target only | 1 | 227.36 | 226.15 | -0.53% |
| Draft 1 | 0 | 136.10 | 141.91 | +4.27% |
| Draft 1 | 1 | 124.09 | 129.69 | +4.51% |
| Draft 4 | 0 | 88.76 | 99.57 | +12.18% |
| Draft 4 | 1 | 80.77 | 90.02 | +11.45% |
| Draft 8 | 0 | 60.14 | 70.95 | +17.96% |
| Draft 8 | 1 | 53.06 | 62.31 | +17.42% |

Same fresh executable and CUDA DLL, tile 0 versus 4, context 2048, 128 maximum
output tokens, one warmup and two measured repetitions per prompt/mode. Rates
come from completed-token and decode-time counter deltas. Two repeats provide
no confidence interval. The one-token capital reply is excluded from the speed
table. Target-only differences do not demonstrate a benefit: the tile kernel
is selected only during draft verification. No cross-engine comparison is made.

The first measured tile-4/draft-4 writing request illustrates the remaining
cost: 214 proposed tokens yielded 72 accepted tokens (33.6%), with 46 rollbacks
over 55 verification blocks. Draft generation consumed 534.5 ms and verification
377.0 ms before other bookkeeping/rollback costs, for 128 completed output
tokens. A matching direct run needs approximately 566 ms of decode. These
counter deltas explain why improved projection reuse is insufficient for this
request; they are not a general acceptance-rate estimate for other prompts.
The exact counters and request are in `raw/quiet-tile4/results.json.gz`.

The serial owner completed check, Clippy, 299 workspace tests (one ignored),
fresh CUDA and release CLI builds, 300 projection oracles including 20 Q8 reuse
cases, real .8B/2B complete-vector and GDN rollback checks with split attention
off/on and graphs/eager, and greedy/coupled draft output/RNG validation. Both
tile captures contain 36 JSON responses, 12 SSE comparisons and four stops;
JSON choices and usage match across tiles. Network validation covers streaming,
disconnect/replay, penalties, stops and serialized concurrent requests. Default
EOS/length handling was also checked on all four actual models.

`receipt.json` binds 61 gzip-preserved raw captures, the exact harnesses, the
compiled Rust/Native source hashes, DLL/executable identities and command logs.
`raw/analysis.json.gz` contains both unrounded samples for every table cell.
`raw/manifest.json.gz` records the compiled source commit `976cd172d9c63bac69891dd3dcab59e09900d485`;
this publication changes only documentation. Executables, GGUF files and local
full float vectors are omitted; their identifiers and test records remain.

Reproduction uses `harness/check_q8_weight_reuse.py` and its copied helpers.
Adjust orchestration/model paths to the local machine and serialize all GPU
tests/builds/measurements. `RBITNET_QWEN_SPEC_Q8_TILE=4` is optional; `0` is the
default. The parent experimental serving PR #123 remains draft. Classical
distribution-correcting q/p rejection sampling and MoE speculative decoding
are not implemented by this change. Refs #97 and #98.
