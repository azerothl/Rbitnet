# Technical Debt

## Strict `bitnet-core` Clippy

`cargo clippy -p bitnet-core --all-targets --all-features -- -D warnings` is not clean yet. The current blocker is broad pre-existing lint debt across low-level numeric/runtime code, not the OpenAI sampling/template plumbing.

Observed buckets:

- `backend.rs`: CUDA FFI type complexity and range-loop cleanup.
- `ggml/dequant.rs` and `ggml/quant_dot.rs`: `manual_is_multiple_of`, `identity_op`, and range-loop lints in quantization kernels.
- `model/toy.rs` and `prefix_kv.rs`: iterator/clamp helpers and `len_without_is_empty`.
- `qwen35/*`: iterator rewrites, lifetime elision, `too_many_arguments` in GDN, and small API cleanups.

Recommended approach: fix by module with local numeric parity tests after each batch, starting with mechanical lints (`is_multiple_of`, `first`, `clamp`, lifetime elision), then review range-loop rewrites in kernels separately to avoid changing indexing semantics.
