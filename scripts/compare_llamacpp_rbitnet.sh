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
#
# Optional:
#   RESULTS_MD=docs/BENCHMARKS_RESULTS.md  # append a markdown section
#   SKIP_LLAMA=1                           # document Rbitnet-only / methodology run

set -euo pipefail
THREADS="${RBITNET_THREADS:-$(nproc 2>/dev/null || echo 8)}"
GGUF="${RBITNET_GGUF:-}"
BENCH="${LLAMA_BENCH:-llama-bench}"
RESULTS_MD="${RESULTS_MD:-}"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUSTC="$(rustc -V 2>/dev/null || echo unknown)"
HOST="$(uname -srm 2>/dev/null || echo unknown)"
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || echo unknown)"

LLAMA_STATUS="skipped"
LLAMA_OUT=""
RBITNET_OUT=""

echo "=== environment ==="
echo "date_utc=$DATE_UTC"
echo "rbitnet_sha=$GIT_SHA"
echo "rustc=$RUSTC"
echo "host=$HOST"
echo "cpu=$CPU_MODEL"
echo "threads=$THREADS"
echo "gguf=${GGUF:-"(unset — set RBITNET_GGUF for llama-bench)"}"

echo ""
echo "=== llama.cpp (llama-bench) ==="
if [[ "${SKIP_LLAMA:-0}" == "1" ]]; then
  echo "(skip: SKIP_LLAMA=1)"
  LLAMA_STATUS="skipped_by_flag"
elif [[ -z "$GGUF" ]]; then
  echo "(skip: RBITNET_GGUF unset)"
  LLAMA_STATUS="skipped_no_gguf"
elif ! command -v "$BENCH" >/dev/null 2>&1; then
  echo "(skip: $BENCH not found; set LLAMA_BENCH to the binary path)"
  LLAMA_STATUS="skipped_no_binary"
else
  echo "Command: $BENCH -m \"$GGUF\" -t \"$THREADS\" -p 512 -n 64"
  set +e
  LLAMA_OUT="$("$BENCH" -m "$GGUF" -t "$THREADS" -p 512 -n 64 2>&1)"
  LLAMA_RC=$?
  set -e
  echo "$LLAMA_OUT"
  if [[ $LLAMA_RC -eq 0 ]]; then
    LLAMA_STATUS="ok"
  else
    LLAMA_STATUS="failed_rc_$LLAMA_RC"
  fi
fi

echo ""
echo "=== Rbitnet (HTTP, same prompt length target via script defaults) ==="
set +e
RBITNET_OUT="$(python3 scripts/bench_backend_compare.py \
  --base-url "${RBITNET_BASE_URL:-http://127.0.0.1:8080}" \
  --model "${RBITNET_BENCH_MODEL:-rbitnet-llama}" \
  --prompt "$(python3 -c "print('x'*480)")" \
  --max-tokens 64 \
  --temperature 0.0 \
  --runs 5 \
  --warmup 1 2>&1)"
RBITNET_RC=$?
set -e
echo "$RBITNET_OUT"
if [[ $RBITNET_RC -ne 0 ]]; then
  echo "(rbitnet HTTP bench failed rc=$RBITNET_RC — is rbitnet-server up?)"
fi

if [[ -n "$RESULTS_MD" ]]; then
  {
    echo ""
    echo "## llama.cpp comparison — $DATE_UTC"
    echo ""
    echo "| Field | Value |"
    echo "|-------|-------|"
    echo "| Date (UTC) | $DATE_UTC |"
    echo "| Host | $HOST |"
    echo "| CPU | $CPU_MODEL |"
    echo "| Rust | \`$RUSTC\` |"
    echo "| Rbitnet SHA | \`$GIT_SHA\` |"
    echo "| Threads | $THREADS |"
    echo "| GGUF | \`${GGUF:-n/a}\` |"
    echo "| llama-bench status | $LLAMA_STATUS |"
    echo "| Rbitnet HTTP rc | $RBITNET_RC |"
    echo ""
    echo "### llama-bench output"
    echo ""
    echo '```'
    echo "${LLAMA_OUT:-(none)}"
    echo '```'
    echo ""
    echo "### Rbitnet HTTP output"
    echo ""
    echo '```'
    echo "${RBITNET_OUT:-(none)}"
    echo '```'
  } >> "$RESULTS_MD"
  echo ""
  echo "Appended section to $RESULTS_MD"
fi

echo ""
echo "Document both outputs in docs/BENCHMARKS.md / docs/BENCHMARKS_RESULTS.md."
