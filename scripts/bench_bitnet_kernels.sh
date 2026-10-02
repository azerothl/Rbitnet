#!/usr/bin/env bash
# Microbench Rbitnet ternary matmul (i8 / I2_S / TL2-LUT / auto) — NATIVE_FIRST, no FFI.
# Also runs TQ1/TQ2 stack-scratch vs heap decode-then-dot when RUN_TQ=1.
# Appends markdown rows to docs/BENCHMARKS_RESULTS.md when APPEND_RESULTS=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N="${N:-64}"
K="${K:-1024}"
ITERS="${ITERS:-50}"

export CXX="${CXX:-g++}"
export RUSTFLAGS="${RUSTFLAGS:--C link-arg=-L/usr/lib/gcc/x86_64-linux-gnu/13}"

echo "== ternary kernel unit tests =="
cargo test -p bitnet-core --lib kernels::tests --locked -- --nocapture

echo "== ternary microbench ${N}x${K} iters=${ITERS} =="
ROW="$(N="$N" K="$K" ITERS="$ITERS" cargo run -p bitnet-core --example ternary_microbench --release --locked 2>/dev/null | tail -n 1)"
echo "ROW: $ROW"

TQ_ROW=""
if [[ "${RUN_TQ:-1}" == "1" ]]; then
  echo "== TQ1/TQ2 stack vs heap microbench =="
  TQ_ROW="$(TQ_ITERS="${TQ_ITERS:-200}" TQ_ROWS="${TQ_ROWS:-64}" cargo run -p bitnet-core --example tq_dot_microbench --release --locked 2>/dev/null | tail -n 1)"
  echo "TQ_ROW: $TQ_ROW"
fi

if [[ "${APPEND_RESULTS:-0}" == "1" ]]; then
  {
    echo
    echo "## BitNet ternary kernels — $(date -u +%Y-%m-%d) (NATIVE_FIRST)"
    echo
    echo "| Shape / paths | ns/call | bit_exact | widest gap | notes |"
    echo "|---------------|---------|-----------|------------|-------|"
    echo "$ROW"
    if [[ -n "$TQ_ROW" ]]; then
      echo "$TQ_ROW"
    fi
  } >>docs/BENCHMARKS_RESULTS.md
  echo "appended to docs/BENCHMARKS_RESULTS.md"
fi

echo "done."
