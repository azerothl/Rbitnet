#!/usr/bin/env bash
# Compare paged KV F32 vs Q8 resident bytes + optional live RSS/tok/s.
# Gate: unit tests (roundtrip error + resident_bytes). Live GGUF optional.
#
# Usage:
#   ./scripts/bench_kv_q8.sh
#   export RBITNET_MODEL=/path/to/tinyllama-*.Q4_K_M.gguf
#   export RBITNET_TOKENIZER=/path/to/tokenizer.json
#   ./scripts/bench_kv_q8.sh
#
# Without a GGUF, runs in-crate Q8 unit tests only and exits 0.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_MD="${RESULTS_MD:-docs/BENCHMARKS_RESULTS.md}"
MAX_TOKENS="${MAX_TOKENS:-8}"
PROMPT="${PROMPT:-Hello, summarize KV Q8 in one short sentence.}"
PORT_BASE="${PORT_BASE:-18890}"

echo "== bitnet-core KV Q8 unit tests (always) =="
cargo test -p bitnet-core --test kv_storage_paged q8_ -- --nocapture
cargo test -p bitnet-core q8_roundtrip_preserves_row_shape -- --nocapture

if [[ -z "${RBITNET_MODEL:-}" || ! -f "${RBITNET_MODEL}" ]]; then
  echo "RBITNET_MODEL unset or missing — skip live RSS/tok/s F32 vs Q8 matrix."
  echo "Install TinyLlama Q4 then re-run, e.g.:"
  echo "  rbitnet models download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \\"
  echo "    --dir ./models --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  echo
  echo "Enablement (CPU paged path):"
  echo "  export RBITNET_LLAMA_PAGED_KV=1"
  echo "  export RBITNET_KV_QUANT=q8"
  echo "  # optional: rbitnet tune throughput  (sets pool; add KV_QUANT=q8 yourself)"
  exit 0
fi

if [[ -z "${RBITNET_TOKENIZER:-}" || ! -f "${RBITNET_TOKENIZER}" ]]; then
  echo "RBITNET_TOKENIZER required when RBITNET_MODEL is set" >&2
  exit 1
fi

BIN="${BIN:-./target/release/rbitnet-server}"
if [[ ! -x "$BIN" ]]; then
  cargo build -p bitnet-server --release
fi

rss_kb() {
  local pid="$1"
  if [[ -r "/proc/$pid/status" ]]; then
    awk '/VmRSS:/ {print $2}' "/proc/$pid/status"
  else
    echo "0"
  fi
}

run_quant() {
  local quant="$1"
  local port="$2"
  export RBITNET_LLAMA_PAGED_KV=1
  export RBITNET_KV_POOL=1
  export RBITNET_KV_QUANT="$quant"
  export RBITNET_HOST=127.0.0.1
  export RBITNET_PORT="$port"
  export RBITNET_MAX_CONCURRENT=1
  export RBITNET_STUB=0
  export RBITNET_TOY=0
  export RBITNET_BACKEND=cpu

  "$BIN" >/tmp/rbitnet-kv-q8-"$quant".log 2>&1 &
  local spid=$!
  trap 'kill '"$spid"' 2>/dev/null || true' RETURN

  local ready=0
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${port}/ready" >/dev/null; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [[ "$ready" != "1" ]]; then
    echo "server not ready (quant=$quant); log:" >&2
    tail -n 40 /tmp/rbitnet-kv-q8-"$quant".log >&2 || true
    kill "$spid" 2>/dev/null || true
    return 1
  fi

  local t0 t1 ms body
  t0=$(date +%s%N)
  body=$(curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"rbitnet\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,os; print(json.dumps(os.environ["PROMPT"]))')}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0}" \
    || true)
  t1=$(date +%s%N)
  ms=$(( (t1 - t0) / 1000000 ))
  local rss
  rss=$(rss_kb "$spid")
  local fmt
  fmt=$(curl -sf "http://127.0.0.1:${port}/metrics" | awk -F' ' '/rbitnet_core_kv_quant_format_code/ {print $2; exit}' || echo "?")
  echo "quant=$quant port=$port wall_ms=$ms rss_kb=$rss kv_quant_format_code=$fmt"
  if [[ -n "$body" ]]; then
    echo "  completion_bytes=${#body}"
  fi
  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  trap - RETURN
}

echo "== live F32 vs Q8 (paged + pool) =="
run_quant off $((PORT_BASE))
run_quant q8 $((PORT_BASE + 1))

{
  echo
  echo "## KV Q8 live — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "**Methodology:** \`scripts/bench_kv_q8.sh\` with \`RBITNET_MODEL\` set. Compare \`off\` vs \`q8\` RSS + wall ms."
  echo
  echo "| Quant | Notes |"
  echo "|-------|-------|"
  echo "| off/f32 | baseline paged pool |"
  echo "| q8 | compact INT8 pages; decode-on-read |"
  echo
} >>"$RESULTS_MD"

echo "Appended stub section to $RESULTS_MD (fill numbers from console above)."
