#!/usr/bin/env bash
# Hardware smoke for #22 Gate E — device-resident quantized Llama matvec.
# CI must NOT run this as a required job. Opt-in on a CUDA box with cuBLAS +
# optional librbitnet_cuda_quant (device symbols).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${RBITNET_BUILD_CUDA_QUANT:-0}" == "1" ]]; then
  echo "== Build librbitnet_cuda_quant =="
  "$ROOT/scripts/build_cuda_quant.sh"
  export LD_LIBRARY_PATH="$ROOT/native/cuda_quant/build:${LD_LIBRARY_PATH:-}"
  export RBITNET_CUDA_QUANT_LIB="$ROOT/native/cuda_quant/build/librbitnet_cuda_quant.so"
fi

echo "== CPU golden (always) =="
cargo test -p bitnet-core --test cuda_quant_residency --test backend_conformance

if [[ -n "${RBITNET_CUDA_QUANT_LIB:-}" ]]; then
  echo "== Opt-in device-resident quant kernel smoke =="
  RBITNET_CUDA_QUANT_SMOKE=1 cargo test -p bitnet-core --test cuda_quant_residency \
    opt_in_device_resident_quant_kernel_when_lib_present -- --nocapture
else
  echo "Skip device kernel smoke: set RBITNET_BUILD_CUDA_QUANT=1 or RBITNET_CUDA_QUANT_LIB"
fi

if [[ "${RBITNET_BENCH_CUDA:-0}" != "1" ]]; then
  echo "Set RBITNET_BENCH_CUDA=1 to run Criterion CUDA rows (requires CUDA runtime)."
  exit 0
fi

echo "== Opt-in CUDA benches (f32 host GEMV + resident f32 + quant Q4_0) =="
RBITNET_BENCH_CUDA=1 cargo bench -p bitnet-core --bench kernels -- --warm-up-time 1 --measurement-time 3

MODEL="${RBITNET_MODEL:-}"
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "Skip token smoke: set RBITNET_MODEL=/path/to/llama.gguf for greedy CPU vs CUDA parity."
  exit 0
fi

echo "== Greedy token smoke: CPU vs CUDA (same GGUF) =="
export RBITNET_BACKEND=cpu
echo "(run your usual short generate against \$RBITNET_MODEL and record first tokens)"
export RBITNET_BACKEND=cuda
echo "(repeat with RBITNET_BACKEND=cuda; compare greedy first tokens; inspect device_resident_quant_gemv_calls)"
echo "See docs/GPU_NATIVE_ROADMAP.md Gate E checklist."
