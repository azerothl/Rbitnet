# Delegating inference to an external backend

For maximum throughput on hardware Rbitnet does not yet fuse end-to-end (see [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) phase D), deployments may **delegate** generation to another OpenAI-compatible server (vLLM, TensorRT-LLM, commercial APIs) while keeping Akasha or other clients unchanged.

## Patterns

1. **Router swap:** Point `llm_router.yaml` (Akasha) or the application’s HTTP client at the external base URL instead of `rbitnet-server`.
2. **Sidecar:** Run `rbitnet-server` for development / GGUF validation and a GPU stack for production; switch via configuration only.
3. **Future Rbitnet hook:** A thin HTTP or gRPC proxy could route subsets of models to delegates; this is **not** implemented in-tree today—design aligns with [RUNNER_PROXY_SPEC.md](RUNNER_PROXY_SPEC.md) for multi-process isolation.

## Trade-offs

- **Pros:** Best-in-class kernels and batching without maintaining them in Rust.
- **Cons:** Extra hop latency, operational complexity, and possibly different tokenizer / chat-template behaviour—validate parity per model family.

When delegating, treat Rbitnet’s GGUF path as the **compatibility reference** for weights and tokenizer files (`docs/TRAINING_AND_COMPATIBILITY.md`).
