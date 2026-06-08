# Delegating inference to an external backend

Rbitnet is native-first: normal operation runs through `bitnet-core` and does not require an external inference engine. This note is only for development experiments, parity checks, or application-level routing outside the supported Rbitnet runtime path.

## Patterns

1. **Router swap:** Point `llm_router.yaml` (Akasha) or the application’s HTTP client at the external base URL instead of `rbitnet-server`.
2. **Sidecar:** Run `rbitnet-server` for development / GGUF validation and a GPU stack for production; switch via configuration only.
3. **Experimental Rbitnet hook:** Any in-tree HTTP delegation must be compiled behind the dev-only `experimental-external-backends` feature and remain disabled by default.

## Trade-offs

- **Pros:** Best-in-class kernels and batching without maintaining them in Rust.
- **Cons:** Extra hop latency, operational complexity, and possibly different tokenizer / chat-template behaviour—validate parity per model family.

When delegating outside the native path, treat Rbitnet’s GGUF path as the **compatibility reference** for weights and tokenizer files (`docs/TRAINING_AND_COMPATIBILITY.md`). See [NATIVE_FIRST.md](NATIVE_FIRST.md) for the product policy.
