# Kernel routing index (Llama / GGUF)

Maps high-level inference phases to in-tree implementations. Inspired by pegainfer `KERNELS.md`.

| Phase | CPU path | CUDA path | Notes |
|-------|----------|-----------|-------|
| Quant matvec | `ggml/quant_dot.rs` | `backend.rs` cuBLAS dynamic | `RBITNET_QUANT_KERNEL=auto\|scalar\|cuda` |
| Dense forward | `llama/model.rs` | `llama/model.rs` hybrid plan | Layer offload via `RBITNET_HYBRID_*` |
| Attention | `llama/model.rs` CPU loops | Planned FlashInfer wrapper | See `docs/GPU_NATIVE_ROADMAP.md` |
| KV write | `llama/kv_storage.rs` dense/paged | `RBITNET_KV_BACKEND=gpu` planned | `RBITNET_LLAMA_PAGED_KV=1` |
| Decode graph | N/A | `llama/cuda_graph.rs` | `RBITNET_CUDA_GRAPH=1` metrics scaffold |
| Sampling | `sampling.rs` | same | Top-p / penalties |

Fusion backlog (phase D): norm+quant+residual, RoPE+KV write — track in `INFERENCE_STACK_V2.md`.
