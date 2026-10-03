# Kernel routing index (Llama / GGUF)

Maps high-level inference phases to in-tree implementations. Inspired by pegainfer `KERNELS.md`.

| Phase | CPU path | CUDA path | Notes |
|-------|----------|-----------|-------|
| Quant matvec | `ggml/quant_dot.rs`, `ggml/quant_simd.rs` | `native/cuda_quant/src/quant_matvec.cu` | Packed AVX2/AVX512 and format-specialized CUDA kernels; F32 activation precision |
| Dense forward | `llama/model.rs` | `llama/model.rs` hybrid plan | Layer offload via `RBITNET_HYBRID_*` |
| Attention | `llama/model.rs` SIMD dots | `native/attention.rs`, `llama_resident.cuh` | Resident F32 KV, fused GQA/MLA, windows and sinks where supported |
| KV write | `llama/kv_storage.rs` dense/paged | resident Llama RoPE/KV fusion | F32 dense GPU cache; paged GPU KV remains unfinished |
| Decode graph | N/A | `llama/resident.rs`, `llama_resident.cuh` | Actual capture/replay with device position; dense fully offloaded Llama, dense KV, no prefix restore |
| Routed FFN | `native/graph.rs` | `native/moe.rs`, `moe_resident.cuh` | Selected experts and biased SwiGLU stay on CUDA; partial offload falls back per layer |
| Recurrent Qwen block | `qwen35/recurrent.rs`, in-place `qwen35/gdn.rs` | `native/qwen_recurrent.rs`, `qwen_recurrent.cuh` | Dense blocks retain convolution, GDN state, norm/gates, projections, FFN and residuals; one host synchronization per layer |
| Sampling | `sampling.rs` | resident Llama argmax; opt-in `native/head.rs`, `output_head.cuh` for Qwen/GPT/MLA | GPU greedy without penalties or masks; native shared head requires `RBITNET_CUDA_HEAD=1` (no measured speed gain yet); other sampling uses the shared Rust sampler |

Resident Llama fuses residual addition with RMSNorm and RoPE with KV writes. Dense Qwen GDN layers now use private resident CUDA state; Qwen full-attention blocks still cross the CPU/GPU boundary. Batched prefill and complete resident MLA/MoE attention remain unfinished; see the [initial optimization report](benchmarks/2026-10-03-parity/README.md) and [second round](benchmarks/2026-10-03-parity-round2/README.md).
