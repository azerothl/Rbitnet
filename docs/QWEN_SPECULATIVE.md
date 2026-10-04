# Experimental Qwen draft serving

The optional CUDA draft path verifies proposals from a second dense Qwen model against the target and restores/replays state after a mismatch. Emitted IDs are sampled by the target. It is disabled by default: the measured Qwen3.5-0.8B draft / Qwen3.5-2B target pair is slower than target-only decoding on the tested RTX 4080 SUPER.

Enable with `RBITNET_QWEN_SPECULATIVE=1`, `RBITNET_QWEN_DRAFT_GGUF=<dense draft GGUF>` and `RBITNET_QWEN_SPEC_DEPTH=1..8` (default 4). Both models must use the full resident CUDA pipeline, identical tokenizer configurations, vocabulary and GGUF token metadata. CPU and MoE pairs are rejected. `RBITNET_PREFIX_KV=1` cannot be combined with this experiment. Tiered prompt-cache reuse is bypassed by the draft route; this change does not establish a combined cache/draft serving implementation.

`RBITNET_QWEN_SPEC_DRAFT_SAMPLING` selects `greedy` (default) or `coupled` proposals. Coupled proposals clone the target RNG stream; rejected proposals do not consume the target's emitted-token RNG. This is not the classical q/p rejection sampler. The actual-model tests check exact output IDs and target RNG state, EOS, cancellation, state restoration and prefix refusal, with split attention both disabled and enabled. Structured-output grammar is unsupported.

The public [proof and measurements](benchmarks/2026-10-04-qwen-draft-serving/README.md) contain the fresh compiled-source identities, compressed original logs, JSON/SSE responses and harnesses. Correctness evidence is limited to the recorded model pair and configurations. A demonstrated speed improvement is required before enabling this path by default or claiming issue #97 complete.
