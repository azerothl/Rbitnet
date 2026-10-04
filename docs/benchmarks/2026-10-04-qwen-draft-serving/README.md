# Qwen draft serving: correctness pass, performance no-go

The fresh run completed on 2026-10-04 at 14:51:36 UTC. Qwen3.5-0.8B Q8_0 proposes tokens for Qwen3.5-2B Q8_0 on an RTX 4080 SUPER. Both models remain on the GPU. The experiment is optional and disabled by default.

`cargo check`, Clippy, the native DLL build and the workspace suite passed (299 tests, one ignored). Four actual-model draft suites cover greedy/coupled proposals with split attention off/on. They verify target IDs and RNG state, EOS, cancellation and prefix refusal. Four-model JSON/SSE finish-reason checks, 20 network cases and the quiet ablation also passed. The ablation records 36 JSON responses, 12 SSE responses and four stop-string probes matching the target-only baseline.

## Measured decode rates

Same binary and DLL, context 2048, maximum 128 generated tokens, one warmup and two measured cycles per prompt/mode. These ablation requests use greedy sampling; sampled correctness is covered separately. Rates are medians computed from completion-token and decode-duration counter deltas, rather than deltas of the instantaneous throughput gauge.

| Prompt | Target only, tokens/s | Draft depth 1 | Depth 4 | Depth 8 |
| --- | ---: | ---: | ---: | ---: |
| Library story (128 output tokens) | 235.29 | 142.70 | 92.23 | 62.61 |
| Museum story (128 output tokens) | 235.51 | 130.15 | 83.92 | 54.96 |
| Capital of France (one output token) | 333.33 | 142.86 | 57.19 | 31.75 |

For the two sustained outputs, depth 1 loses 39.4% and 44.7%; deeper proposals lose more. The one-token response has coarse millisecond timing and is not a steady-state throughput estimate. There is no confidence interval or cross-engine parity claim.

Draft generation and ordered verification both add cost. Across the six measured requests, median draft/verification wall times are approximately 187/419 ms at depth 1, 516/511 ms at depth 4 and 925/735 ms at depth 8. These timers include synchronization/readbacks; checkpointing, replay and sampling also contribute to total decode time. The recorded global GPU baseline is 1961 MiB; observed peaks are 4427 MiB target-only and 5531/5569/5609 MiB for depths 1/4/8. These are global device measurements, not isolated process VRAM.

The next optimization target is the ordered verification projections: their current launch grid assigns a separate block to each token and does not explicitly share decoded weight loads across tokens. Weight reuse is an unvalidated opportunity; it is not evidence that speculative decoding will become faster.

## Traceability and reproduction limits

`receipt.json` lists SHA-256 values for the original captures, their gzip files and exact harnesses. `raw/manifest.json.gz` binds the full Rust/native/Cargo source files in the compiled checkout. The CLI SHA-256 is `f659a36e25e136268183de62f561c1767bf4b5480454f26848304ac888068f04`; the DLL SHA-256 is `507960aa97c4f79687febc0d696d52e4a1582230a541ecab579db4edddcda094`. Models and binaries are not committed.

The original capture bytes are preserved in `raw/`. The `before-repair` diagnostic records a build-cache investigation and is not passing validation evidence; the sealed journal and final check/release logs identify the successful run after removing stale project artifacts. Some diagnostic logs may use Windows console encoding. Source hashes identify working-file bytes; Git line-ending conversion may make those differ from checkout bytes on another platform. Changed source files were checked for line-ending-normalized equality against the actual staged Git blobs. Harnesses and captures have `-text` attributes and were checked as exact Git blobs.

The saved checker uses local Windows paths and serial hardware ownership journals. Reproduction requires supplying equivalent model/tokenizer paths and adapting those orchestration paths. Do not run multiple hardware checkers concurrently. The evidence covers this dense pair only; it does not validate large MoE drafts, a q/p rejection sampler, structured grammar, tiered cache/draft serving or a speed benefit.
