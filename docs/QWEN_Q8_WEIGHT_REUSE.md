# Qwen ordered Q8 weight-reuse prototype

`RBITNET_QWEN_SPEC_Q8_TILE=4` selects a four-token Q8 ordered projection kernel only for speculative verification. Each warp decodes a weight once and applies it to four independent original FMA chains. Token tails retain bounds checks; other GGUF formats and ordinary prompt/token forwards retain their existing kernels. The default is `0`.

Rust passes this option through a dedicated Native ABI before prompt state or graphs exist. Native configuration refuses unsupported values and already populated/captured contexts; an older DLL without the ABI refuses the explicit option. This does not rely on the Windows DLL observing Rust environment updates.

The fresh serial proof passed: 299 workspace tests, projection bits/F64 checks, actual-model logits/recurrent state and draft target IDs, same-binary/DLL tile 0/4 JSON/SSE identities and quiet timings. See [raw captures and adoption decision](benchmarks/2026-10-04-q8-weight-reuse-fresh/README.md). Weight reuse improved the tested draft modes by 4-18%, but every tested draft mode remained slower than direct target decoding. The option stays disabled by default; this is an experimental result with a no-go decision for default speculative serving.
