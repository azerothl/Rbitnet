# Optional native CUDA quantized matvec library (`librbitnet_cuda_quant`)

Ships the ABI expected by `bitnet-core` for [#22](https://github.com/azerothl/Rbitnet/issues/22) Gate E:

| Symbol | Role |
|--------|------|
| `rbitnet_cuda_q4_0_matvec` / `_device` | GGML Q4_0 |
| `rbitnet_cuda_q8_0_matvec` / `_device` | GGML Q8_0 |
| `rbitnet_cuda_q4_k_matvec` / `_device` | GGML Q4_K |
| `rbitnet_cuda_q6_k_matvec` / `_device` | GGML Q6_K |
| `rbitnet_cuda_f32_matvec` / `_device` | F32 |
| `rbitnet_cuda_q5_0_matvec` / `_device` | GGML Q5_0 |
| `rbitnet_cuda_q5_k_matvec` / `_device` | GGML Q5_K |
| `rbitnet_cuda_mxfp4_matvec` / `_device` | GGML MXFP4 (39) |
| `rbitnet_cuda_quant_matvec_batch_device` | Independent input for each resident matrix slab, e.g. MLA heads |
| `rbitnet_cuda_attention_create`, `_step`, `_reset`, `_destroy` | Resident F32 KV and fused all-head GQA/MQA, optional window and attention sinks |
| `rbitnet_cuda_llama_create`, `_step`, `_destroy` | Complete dense Llama token graph, private stream/KV, reusable CUDA graphs, optional device greedy reduction |
| `rbitnet_cuda_moe_create`, `_step`, `_destroy` | Selected expert FFN graph, optional biases, standard/OAI SwiGLU and weighted combination on GPU |

`*_device` variants take **device-resident** weight payloads (`CudaDeviceQuantMatrix`); host `x`/`y` are uploaded/downloaded inside the library.

One warp cooperates on each matrix output row. Kernels specialize the format at compile time and process packed groups of four weights for K formats, Q8_0 and MXFP4. Scratch allocations belong to the calling thread; resident graphs own their streams and buffers. Weight pointers borrowed by graph contexts must remain alive until those contexts are destroyed (the Rust wrappers retain device matrix clones).

Standalone attention launches one block per query head. Only new K/V rows are copied during sequential decode; reset clears the filled length, and the next step uploads a restored host prefix when necessary. Capacity is at most 8192 positions. Rust uses CPU attention for unsupported cache layouts or absent attention symbols.

The resident Llama path also retains activations on GPU, fuses residual/RMSNorm and RoPE/KV writes, and captures three modes: intermediate prefill without logits, F32 logits for general sampling, and greedy token reduction. Position lives in a device buffer so advancing decode does not recapture. Starting at position zero resets the valid KV length. Immutable prefix snapshots retain used K/V on device. Missing symbols, partial offload, per-head Q/K normalization or host paged KV keep the existing runtime; an older DLL without snapshots uses the host prefix fallback. Disable with `RBITNET_CUDA_RESIDENT=0`, or disable only graph replay with `RBITNET_CUDA_RESIDENT_GRAPH=0`.

Routed FFN contexts read current expert IDs and probabilities from device buffers. Gate/up projections, biased activation, down projection and ordered weighted reduction execute in one reusable graph, with one host synchronization per layer. A layer needs all three expert matrices resident; unsupported/partially offloaded layers retain their CPU/GPU path. Disable with `RBITNET_CUDA_MOE=0`.

Dense Qwen3.5 recurrent blocks (`qwen_recurrent.cuh`) retain their convolution history and F32 GDN matrices on CUDA. Eight projections, both norms/residuals, convolution/SiLU, L2 Q/K, recurrent decay/update, per-head RMS/gate and the FFN execute in one captured block graph. One warp owns a value row of the GDN state. Q/K normalization preserves GGUF's `max(sum, epsilon)` semantics. Position zero clears both histories; subsequent positions must be sequential. Supported equal key/value head widths are at most 256; unsupported/MoE blocks and partial offload use the prior path. Small alpha/beta projections count against the same weight budget. Disable with `RBITNET_CUDA_QWEN_RECURRENT=0`, or use eager block kernels with `RBITNET_CUDA_QWEN_RECURRENT_GRAPH=0`.

The shared resident output head (`output_head.cuh`) combines RMSNorm, quantized projection and an optional two-level argmax. Greedy mode downloads four bytes and preserves Rust's NaN, signed-zero and last-ID tie policy. General sampling, penalties and structured masks download F32 logits. Enable experimentally with `RBITNET_CUDA_HEAD=1`; current Windows ablations show no speed gain, so the default uses the prior output path. Each head/block owns its buffers/stream, and Rust retains cloned immutable device weights until after context destruction. CUDA failures are propagated; an executing sequence never silently switches state to CPU.

`RBITNET_CUDA_PREFILL=1` enables shared-weight quantized SIMT GEMM for Llama blocks of 1..128 positions. Ordinary prefill returns only the final logits. The optional `llama_verify` API returns logits or greedy IDs for every position of a block of 1..16 tokens, with separate CUDA graphs per size/mode and a stable device position buffer. `llama_truncate` rolls dense K/V back by reducing its valid length; causal reads ignore the discarded tail. These operations support the token-based PLD path enabled by `RBITNET_SPECULATIVE_PLD=1` (or the legacy `RBITNET_SPECULATIVE=1`). Qwen full attention and MLA attention activations are not fully resident in these graphs. All ABI additions remain optional for older DLLs.

Rust serializes only cold runtime loading and completes lazy driver initialization with `cudaFree(nullptr)` before returning a runtime. Per-model metrics/buffers stay independent and inference is not serialized by this lock. Hardware regression tests include eight simultaneous runtime loaders/uploads and parallel resident graph tests.

## Build

**Windows (PowerShell):**

```powershell
.\scripts\build_cuda_quant.ps1
# DLL under native\cuda_quant\build\rbitnet_cuda_quant64.dll
$env:PATH = "$(Resolve-Path .\native\cuda_quant\build);$env:PATH"
# or:
$env:RBITNET_CUDA_QUANT_LIB = (Resolve-Path .\native\cuda_quant\build\rbitnet_cuda_quant64.dll).Path
```

**Linux / macOS (NVIDIA):**

```bash
./scripts/build_cuda_quant.sh
export LD_LIBRARY_PATH="$PWD/native/cuda_quant/build:${LD_LIBRARY_PATH:-}"
# or:
export RBITNET_CUDA_QUANT_LIB="$PWD/native/cuda_quant/build/librbitnet_cuda_quant.so"
```

Requires a working `nvcc` (CUDA Toolkit). Default CI does **not** build or link this library.

## Validate

```powershell
.\scripts\smoke_cuda_quant_residency.ps1
```

```bash
RBITNET_BENCH_CUDA=1 ./scripts/smoke_cuda_quant_residency.sh
```

Without the DLL, Rbitnet still runs correct CPU goldens; `device_resident_quant_gemv_calls` stays 0.

The expanded hardware tests cover all eight formats, row views, independent-head batches, concurrent calls and attention parity with GQA, sinks, sliding windows and restored prefixes:

```powershell
$env:RBITNET_CUDA_QUANT_SMOKE = '1'
cargo test --release -p bitnet-core --test cuda_quant_residency
cargo test --release -p bitnet-core --lib native::attention::tests
```

Real-model results and limits: [native inference validation](../../docs/benchmarks/2026-10-03-optimized/README.md).

Dense Qwen3.5 has an optional full token pipeline (`RBITNET_CUDA_QWEN_FULL=1`).
Recurrent and full-attention contexts share a private stream and hidden vector;
the graph reads a device position updated outside capture. Full attention includes
per-head Q/K RMSNorm, partial NeoX RoPE, GQA, sigmoid query gating and dense FFN.
Modes select no head, complete logits, or device argmax. The latter is used only
when Rust sampling options permit it; seeded sampling and penalties retain Rust
sampling. Native attention snapshots retain only used F32 K/V and pair with exact
GDN/convolution checkpoints. All layer contexts and weights must outlive the
pipeline, and callers must serialize access to the borrowed contexts.

```powershell
$env:RBITNET_CUDA_QUANT_SMOKE = '1'
cargo test --release -p bitnet-core --lib native::qwen_full::tests -- --test-threads=1
pwsh -NoProfile -File scripts/validate_qwen_full.ps1
pwsh -NoProfile -File scripts/validate_qwen_full.ps1 -Eager
```

The mode stays opt-in. Missing new symbols retain the prior recurrent/host
pipeline; `RBITNET_REQUIRE_QWEN_FULL=1` makes that fallback a load error.
