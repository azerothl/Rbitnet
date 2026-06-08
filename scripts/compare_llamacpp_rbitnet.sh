#!/usr/bin/env bash
# Fair CPU comparison: llama.cpp llama-bench vs Rbitnet HTTP (same GGUF, same thread budget).
# Prerequisites: built llama.cpp (llama-bench on PATH), rbitnet-server running with RBITNET_MODEL/RBITNET_TOKENIZER.
#
# Usage:
#   export LLAMA_BENCH=/path/to/build/bin/llama-bench
#   export RBITNET_GGUF=/path/model.gguf
#   export RBITNET_THREADS=8
#   # Terminal 1: RBITNET_BACKEND=cpu RBITNET_MODEL=... rbitnet-server --release
#   # Terminal 2:
#   ./scripts/compare_llamacpp_rbitnet.sh

set -euo pipefail
THREADS="${RBITNET_THREADS:-8}"
GGUF="${RBITNET_GGUF:?set RBITNET_GGUF to the same .gguf used by rbitnet-server}"
BENCH="${LLAMA_BENCH:-llama-bench}"

echo "=== llama.cpp (llama-bench) ==="
echo "Command: $BENCH -m \"$GGUF\" -t \"$THREADS\" -p 512 -n 64"
if command -v "$BENCH" >/dev/null 2>&1; then
  "$BENCH" -m "$GGUF" -t "$THREADS" -p 512 -n 64 || true
else
  echo "(skip: $BENCH not found; set LLAMA_BENCH to the binary path)"
fi

echo ""
echo "=== Rbitnet (HTTP, same prompt length target via script defaults) ==="
python3 scripts/bench_backend_compare.py \
  --base-url "${RBITNET_BASE_URL:-http://127.0.0.1:8080}" \
  --model "${RBITNET_BENCH_MODEL:-rbitnet-llama}" \
  --prompt "$(python3 -c "print('x'*480)")" \
  --max-tokens 64 \
  --temperature 0.0 \
  --runs 5 \
  --warmup 1

echo ""
echo "Document both outputs in docs/BENCHMARKS.md (section llama.cpp comparison)."
