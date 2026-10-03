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

`*_device` variants take **device-resident** weight payloads (`CudaDeviceQuantMatrix`); host `x`/`y` are uploaded/downloaded inside the library.

One warp cooperates on each matrix output row. Scratch allocations belong to the calling thread, so concurrent runtimes do not overwrite another call's inputs. MXFP4 uses packed E2M1 magnitudes without a dynamically indexed local table.

Attention launches one block per query head, with all heads of a layer in one launch. Only new K/V rows are copied during sequential decode; reset clears the filled length, and the next step uploads a restored host prefix when necessary. Capacity is at most 8192 positions. This is decode-style attention, not tiled parallel prefill or full device-resident activations. Rust uses CPU attention for unsupported cache layouts or absent attention symbols.

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
