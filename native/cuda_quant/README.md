# Optional native CUDA quantized matvec library (`librbitnet_cuda_quant`)

Ships the ABI expected by `bitnet-core` for [#22](https://github.com/azerothl/Rbitnet/issues/22) Gate E:

| Symbol | Role |
|--------|------|
| `rbitnet_cuda_q4_0_matvec` / `_device` | GGML Q4_0 |
| `rbitnet_cuda_q8_0_matvec` / `_device` | GGML Q8_0 |
| `rbitnet_cuda_q4_k_matvec` / `_device` | GGML Q4_K |
| `rbitnet_cuda_q6_k_matvec` / `_device` | GGML Q6_K |

`*_device` variants take **device-resident** weight payloads (`CudaDeviceQuantMatrix`); host `x`/`y` are uploaded/downloaded inside the library.

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
