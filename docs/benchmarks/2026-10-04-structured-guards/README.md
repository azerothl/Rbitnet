# Actual structured-capability refusals

These captures bind a fresh CLI and Native library to the source hashes in
`raw/manifest.json.gz`. Source hashes describe the exact working-file bytes read
for compilation; Git checkout line endings can differ. Compressed captures and
copied harnesses retain their original bytes, listed in `receipt.json`.

On Windows, Ryzen 7 9800X3D and RTX 4080 SUPER, actual Llama 3.2 1B, Qwen 3.5 2B,
GPT-OSS 20B and GLM 4.7 Flash passed 208 refusals across CPU and CUDA. Requested
JSON/schema/tool generation returned HTTP 501 with an ordinary JSON error,
before SSE and without changing forward counters. Sixteen before/after control
records each contain ordinary JSON and SSE responses. These controls prove
continuation after refusal, not a throughput improvement.

Check, Clippy, 300 workspace tests (one ignored), Native compilation and the
actual network suite completed. Workspace optional model tests are distinct
from the explicit four-model HTTP suite. Original command logs are retained;
diagnostic decoding sidecars identify any non-UTF-8 console bytes without
changing the originals.

The GPT control budget is 64. Its prompt needs 16 total tokens including hidden
reasoning to produce `Paris`; the separate observed eight-token response was
empty and ended with `length`, equally in JSON and SSE. This protocol adjustment
does not relax refusal checks or remove inference-counter comparisons.

True tokenizer-aware JSON/schema/tool grammar remains unimplemented, and issue
#24 remains open. This evidence does not validate AMD, Intel, Metal, every
tokenizer or every combination of other optional engine features.
