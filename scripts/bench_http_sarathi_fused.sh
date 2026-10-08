#!/usr/bin/env bash
# Measure HTTP concurrency through the Sarathi + CUDA fused Llama path (#96).
#
# Requires a CUDA build, an NVIDIA GPU, a Llama-compatible GGUF, and its tokenizer.
# It intentionally prints measurements only; it never appends fabricated results.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 4 8}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT="${PROMPT:-Explain continuous batching in one short paragraph.}"
PORT_BASE="${PORT_BASE:-18880}"
BIN="${BIN:-./target/release/rbitnet-server}"

for tool in curl jq nvidia-smi; do
  command -v "$tool" >/dev/null || {
    echo "missing required tool: $tool" >&2
    exit 1
  }
done
[[ -n "${RBITNET_MODEL:-}" && -f "$RBITNET_MODEL" ]] || {
  echo "set RBITNET_MODEL to a readable Llama-compatible GGUF" >&2
  exit 1
}
[[ -n "${RBITNET_TOKENIZER:-}" && -f "$RBITNET_TOKENIZER" ]] || {
  echo "set RBITNET_TOKENIZER to a readable tokenizer.json/model" >&2
  exit 1
}
[[ -x "$BIN" ]] || cargo build -p bitnet-server --release
nvidia-smi -L >/dev/null

metric() {
  local name="$1" metrics="$2"
  awk -v metric="$name" '$1 == metric { print $2; found=1 } END { if (!found) print 0 }' <<<"$metrics"
}

run_case() {
  local concurrency="$1" port="$2"
  export RBITNET_HOST=127.0.0.1 RBITNET_PORT="$port"
  export RBITNET_BACKEND=cuda RBITNET_CUDA_PREFILL=1 RBITNET_CUDA_KV_FORMAT=f32
  export RBITNET_CONTINUOUS_BATCHING=1 RBITNET_FUSED_MULTI_SEQ=1
  export RBITNET_CUDA_FUSED_DECODE_SLOTS=8 RBITNET_CUDA_CONTINUOUS=0
  export RBITNET_MAX_CONCURRENT="$concurrency"

  "$BIN" >"/tmp/rbitnet-sarathi-fused-$concurrency.log" 2>&1 &
  local pid=$!
  trap 'kill "$pid" 2>/dev/null || true' RETURN
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/ready" >/dev/null && break
    sleep 0.5
  done
  curl -sf "http://127.0.0.1:$port/ready" >/dev/null || {
    cat "/tmp/rbitnet-sarathi-fused-$concurrency.log" >&2
    return 1
  }

  local body before after start end elapsed total_tps rows waves
  body="$(jq -nc --arg prompt "$PROMPT" --argjson max_tokens "$MAX_TOKENS" \
    '{model:"local",messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,temperature:0}')"
  before="$(curl -sf "http://127.0.0.1:$port/metrics")"
  start="$(date +%s%N)"
  for _ in $(seq 1 "$concurrency"); do
    curl -sfS -X POST "http://127.0.0.1:$port/v1/chat/completions" \
      -H 'content-type: application/json' -d "$body" >/dev/null &
  done
  wait
  end="$(date +%s%N)"
  after="$(curl -sf "http://127.0.0.1:$port/metrics")"
  elapsed="$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", (e-s)/1e9 }')"
  total_tps="$(awk -v tokens="$((concurrency * MAX_TOKENS))" -v seconds="$elapsed" \
    'BEGIN { if (seconds > 0) printf "%.3f", tokens / seconds; else print 0 }')"
  rows="$(( $(metric rbitnet_core_gpu_llama_batch_rows_total "$after") - $(metric rbitnet_core_gpu_llama_batch_rows_total "$before") ))"
  waves="$(( $(metric rbitnet_core_gpu_llama_batch_waves_total "$after") - $(metric rbitnet_core_gpu_llama_batch_waves_total "$before") ))"
  printf '| %s | %s | %s | %s | %s |\n' \
    "$concurrency" "$elapsed" "$total_tps" "$rows" "$waves"

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - RETURN
}

echo "# HTTP Sarathi fused decode ($(date -u +%F))"
echo
echo "| concurrency | wall_s | requested_tok/s | batch_rows_delta | batch_waves_delta |"
echo "|---:|---:|---:|---:|---:|"
index=0
for concurrency in $CONCURRENCY_LIST; do
  run_case "$concurrency" "$((PORT_BASE + index))"
  index=$((index + 1))
done
