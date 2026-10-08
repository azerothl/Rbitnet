# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-08

### Added

- CUDA Llama serving slice: Sarathi fused multi-seq decode and shared prefill (`RBITNET_CONTINUOUS_BATCHING` + `RBITNET_FUSED_MULTI_SEQ`), HTTP coalescing, and opt-in live SSE mux with measured 1/4/8 TTFT/ITL.
- Hierarchical session tiers for Llama/Qwen: RAM and SSD prefix restore, idle reload, and a measured Llama 3.2 1B CUDA TTFT (cold recompute vs RAM vs SSD).
- Opt-in adaptive admission for the CUDA Llama live worker (`RBITNET_CUDA_CONTINUOUS_ADMISSION=adaptive`).
- CUDA block prefill for Llama, Qwen, GPT-OSS, and GLM MLA, plus resident/paged KV decision records.
- HTTP **501** `vision_not_supported` when OpenAI/Anthropic requests include `image_url` / `image` / `input_image` (issue #143 phase 0 — no silent text drop).
- Explicit GGUF refuse for `spark2_5` / Spark-X2.5 (issue #142) so those files no longer fall through to the Llama loader.
- CPU MVP loader for Spark-X2.5 (`spark2_5`): fused QKV, ISWA, head-wise sigmoid gate, GELU FFN; 1.7B Q4_K_M `/v1` smoke (issue #142).
- `MmprojInfo` + `RBITNET_MMPROJ` / sibling `mmproj-*.gguf` discovery for inspect-only vision sidecars (issue #143; encoder still TODO).
- Research note [`docs/research/2026-10-06-inference-optima.md`](docs/research/2026-10-06-inference-optima.md) + TinyLlama PLD ablation (issue #144).

- GGML I-quant mmap GEMV + `tensor_to_f32` for **IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_S / IQ1_M / IQ4_XS** (issue #138; GSQ-RCO mixed packs). `inspect_gguf` prints type names and decode flags. **Q2_K** mmap GEMV so mixed GSQ-RCO rows that use type 10 actually decode.
- [CHANGELOG.md](CHANGELOG.md) (this file), [docs/ENV_REFERENCE.md](docs/ENV_REFERENCE.md), [docs/profiling/](docs/profiling/README.md), [docs/INFERENCE_STACK_V2.md](docs/INFERENCE_STACK_V2.md).
- `rbitnet serve` / `rbitnet-server`: optional `--api-key` and `--bind` when env vars are unset.
- `RBITNET_MAX_PROMPT_TOKENS` optional HTTP guard; `Engine::count_prompt_tokens` / `ModelExecutor::count_prompt_tokens`.
- Startup validation for zero-valued caps; bundle install validates GGUF/tokenizer files on disk.
- Optional `optional_engine_load_from_env_smoke` (`RBITNET_TEST_GGUF` + tokenizer beside GGUF).
- **#22 Gate E hardware vertical:** ship `native/cuda_quant` (`librbitnet_cuda_quant` / `rbitnet_cuda_quant64.dll`) with host + device-resident Q4_0/Q8_0/Q4_K/Q6_K matvec ABI; `RBITNET_CUDA_QUANT_LIB`; CUDA 13 loader paths; `RBITNET_BACKEND=auto`; ROCm hipBLAS f32 GEMV; Intel→Vulkan alias; smoke scripts `build_cuda_quant.*` / `smoke_cuda_quant_residency.ps1`.
- **#22 Gate E:** `CudaDeviceQuantMatrix` + `device_resident_quant_gemv_calls`; Llama `cuda`/`hybrid` prefers quantized device residency (Q4_0/Q8_0/Q4_K/Q6_K) with CPU golden fallback; CI tests in `cuda_quant_residency`; opt-in `RBITNET_BENCH_CUDA=1` rows + `scripts/smoke_cuda_quant_residency.sh`.

### Changed

- Default `RBITNET_BACKEND` is now **`auto`** (CUDA→ROCm→Metal→Vulkan→CPU) instead of `cpu`. Pin `RBITNET_BACKEND=cpu` for reproducible benches/goldens.
- Documentation: stubs/MVP audit refreshed after #25 close — epic #24 remainder is **#22 only**; removed merge-conflict markers from [docs/STUBS_AND_MVP_AUDIT.md](docs/STUBS_AND_MVP_AUDIT.md).
- Documentation: stubs/MVP audit refreshed post #51–#71 ([docs/STUBS_AND_MVP_AUDIT.md](docs/STUBS_AND_MVP_AUDIT.md)); #46/#39/#44 closed in the map.
- Documentation: published **real TinyLlama Q4_K_M CPU** throughput/RSS row (issue #23); frozen reference models in MODEL_MATRIX; release bench checklist; stub/MVP audit ([docs/STUBS_AND_MVP_AUDIT.md](docs/STUBS_AND_MVP_AUDIT.md)); LIMITATIONS/STATUS/INFERENCE_STACK_V2 synced for prefix-KV, Sarathi, PLD, tokenizer.model (issue #24); DEPLOYMENT rate-limit sketch, STATUS/USAGE cross-links.
- [docs/GPU_NATIVE_ROADMAP.md](docs/GPU_NATIVE_ROADMAP.md): Gate E quantized residency acceptance + hardware next step.
