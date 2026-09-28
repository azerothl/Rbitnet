#!/usr/bin/env bash
# Exercise Sarathi-style continuous batching unit gates (+ optional live ≥2 sessions).
#
# Usage:
#   ./scripts/bench_sarathi.sh
#   export RBITNET_MODEL=/path/to/tinyllama-*.Q4_K_M.gguf
#   export RBITNET_TOKENIZER=/path/to/tokenizer.json
#   ./scripts/bench_sarathi.sh
#
# Without a GGUF, runs scheduler unit tests only and exits 0.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== bitnet-core Sarathi / continuous batching unit tests (always) =="
cargo test -p bitnet-core --test scheduler_speculative -- --nocapture

if [[ -z "${RBITNET_MODEL:-}" || ! -f "${RBITNET_MODEL}" ]]; then
  echo "RBITNET_MODEL unset or missing — skip live multi-session smoke."
  echo "Enablement:"
  echo "  export RBITNET_CONTINUOUS_BATCHING=1"
  echo "  export RBITNET_PREFILL_CHUNK_TOKENS=256"
  echo "  export RBITNET_ITERATION_TOKEN_BUDGET=512"
  echo "  export RBITNET_SESSIONS=1"
  echo "  # or: rbitnet tune throughput"
  echo "Metrics: rbitnet_core_scheduler_stall_free_iters_total,"
  echo "         rbitnet_core_scheduler_prefill_chunks_total,"
  echo "         rbitnet_core_scheduler_decode_waves_total"
  exit 0
fi

if [[ -z "${RBITNET_TOKENIZER:-}" || ! -f "${RBITNET_TOKENIZER}" ]]; then
  echo "RBITNET_TOKENIZER required when RBITNET_MODEL is set" >&2
  exit 1
fi

BIN="${BIN:-./target/release/rbitnet-server}"
PORT="${PORT:-18910}"
if [[ ! -x "$BIN" ]]; then
  cargo build -p bitnet-server --release
fi

export RBITNET_CONTINUOUS_BATCHING=1
export RBITNET_PREFILL_CHUNK_TOKENS="${RBITNET_PREFILL_CHUNK_TOKENS:-256}"
export RBITNET_ITERATION_TOKEN_BUDGET="${RBITNET_ITERATION_TOKEN_BUDGET:-512}"
export RBITNET_SESSIONS=1
export RBITNET_HOST=127.0.0.1
export RBITNET_PORT="$PORT"
export RBITNET_MAX_CONCURRENT=4
export RBITNET_STUB=0
export RBITNET_TOY=0
export RBITNET_BACKEND=cpu
export RBITNET_CUDA_GRAPH=0

"$BIN" >/tmp/rbitnet-sarathi.log 2>&1 &
spid=$!
trap 'kill '"$spid"' 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/ready" >/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "$ready" != "1" ]]; then
  echo "server not ready; log:" >&2
  tail -n 40 /tmp/rbitnet-sarathi.log >&2 || true
  exit 1
fi

PROMPT_LONG="${PROMPT_LONG:-Write a short paragraph about continuous batching and chunked prefill for local LLM serving.}"
for i in 1 2; do
  curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"rbitnet\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT_LONG} (#${i})\"}],\"max_tokens\":8,\"temperature\":0}" \
    >/tmp/rbitnet-sarathi-resp-"$i".json &
done
wait

echo "== /metrics (scheduler) =="
curl -sf "http://127.0.0.1:${PORT}/metrics" | grep -E 'rbitnet_core_scheduler_(stall_free|prefill_chunks|decode_waves|iteration_budget)' || true
echo "Sarathi live smoke done (see /tmp/rbitnet-sarathi*.json)."
