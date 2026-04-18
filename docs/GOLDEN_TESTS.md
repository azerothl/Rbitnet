# Golden tests (numerical regression)

## Goal

Keep Rust kernels and, later, the full graph aligned with the **bitnet.cpp** reference (or the official Python inference scripts).

## Recommended pipeline

1. Pick a small model (e.g. dummy from BitNet `utils/generate-dummy-bitnet-model.py`) or a lightweight official GGUF.
2. Fix a **seed**, **prompt**, and **generation length**.
3. Export from the reference:
   - either **logits** for the first step (binary or compact JSON), or
   - full generated text for an end-to-end check.
4. Store expected artifacts under `tests/data/golden/` (or CI cache).
5. Rust tests load those files and compare with a documented tolerance (`eps` on floats).

## Current state in Rbitnet

- Kernel tests **without external files** already validate reference matmul (ternary) with embedded expected values.
- Extending to vectors exported from bitnet.cpp is documented here for follow-up work.

## Tolerance guidelines

- FP32 ops: `1e-5` relative or absolute depending on accumulation depth.
- Quantized weights: compare dequantized values or logits with a looser tolerance initially.
