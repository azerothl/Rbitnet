#!/usr/bin/env bash
# Build optional librbitnet_cuda_quant.so (#22 Gate E). Requires nvcc.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/native/cuda_quant/build}"
mkdir -p "$OUT_DIR"
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found" >&2
  exit 1
fi
SRC="$ROOT/native/cuda_quant/src/quant_matvec.cu"
INC="$ROOT/native/cuda_quant/include"
OUT="$OUT_DIR/librbitnet_cuda_quant.so"
ARCH_FLAGS="${RBITNET_CUDA_GENCODE:--gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89}"
# shellcheck disable=SC2086
"$NVCC" -shared -Xcompiler -fPIC -O3 -std=c++17 -I "$INC" -o "$OUT" "$SRC" $ARCH_FLAGS -lcudart
ln -sfn "$(basename "$OUT")" "$OUT_DIR/librbitnet_cuda_quant.so.1" 2>/dev/null || true
echo "Built $OUT"
echo "Add to LD_LIBRARY_PATH or set RBITNET_CUDA_QUANT_LIB=$OUT"
