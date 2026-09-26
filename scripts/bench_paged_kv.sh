#!/usr/bin/env bash
# Compare dense vs paged vs shared KV pool at concurrency 1/4/8.
# Gate: same greedy path as optional golden when RBITNET_GOLDEN_JSON is set.
#
# Usage:
#   export RBITNET_MODEL=/path/to/tinyllama-*.Q4_K_M.gguf
#   export RBITNET_TOKENIZER=/path/to/tokenizer.json
#   ./scripts/bench_paged_kv.sh
#
# Without a GGUF, runs the in-crate paged pool unit tests only and exits 0.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_MD="${RESULTS_MD:-docs/BENCHMARKS_RESULTS.md}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 4 8}"
MAX_TOKENS="${MAX_TOKENS:-8}"
PROMPT="${PROMPT:-Hello, summarize paged KV in one short sentence.}"
MODES="${MODES:-dense paged pool}"

echo "== bitnet-core paged KV unit tests (always) =="
cargo test -p bitnet-core --test kv_storage_paged -- --nocapture

if [[ -z "${RBITNET_MODEL:-}" || ! -f "${RBITNET_MODEL}" ]]; then
  echo "RBITNET_MODEL unset or missing — skip live RSS/tok/s matrix."
  echo "Install TinyLlama Q4 then re-run, e.g.:"
  echo "  rbitnet models download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \\"
  echo "    --dir ./models --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
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

run_mode() {
  local mode="$1"
  local concurrency="$2"
  local port="$3"
  unset RBITNET_LLAMA_PAGED_KV RBITNET_KV_POOL || true
  case "$mode" in
    dense)
      export RBITNET_LLAMA_PAGED_KV=0
      export RBITNET_KV_POOL=0
      ;;
    paged)
      export RBITNET_LLAMA_PAGED_KV=1
      export RBITNET_KV_POOL=0
      ;;
    pool)
      export RBITNET_LLAMA_PAGED_KV=1
      export RBITNET_KV_POOL=1
      ;;
    *)
      echo "unknown mode $mode" >&2
      return 1
      ;;
  esac

  export RBITNET_HOST=127.0.0.1
  export RBITNET_PORT="$port"
  export RBITNET_MAX_CONCURRENT="$concurrency"
  export RBITNET_STUB=0
  export RBITNET_TOY=0
  export RBITNET_BACKEND=cpu

  "$BIN" >/tmp/rbitnet-paged-kv-"$mode"-"$concurrency".log 2>&1 &
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
    echo "server not ready ($mode c=$concurrency); log:" >&2
    tail -n 40 /tmp/rbitnet-paged-kv-"$mode"-"$concurrency".log >&2 || true
    kill "$spid" 2>/dev/null || true
    return 1
  fi

  local body
  body="$(jq -nc --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
    '{model:"local",messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0}')"

  local start end elapsed
  start="$(date +%s%N)"
  for i in $(seq 1 "$concurrency"); do
    curl -sf -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
      -H 'content-type: application/json' \
      -d "$body" >/tmp/rbitnet-paged-kv-out-"$mode"-"$concurrency"-"$i".json &
  done
  wait
  end="$(date +%s%N)"
  elapsed="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", (e-s)/1e9}')"

  local rss
  rss="$(rss_kb "$spid")"
  local metrics
  metrics="$(curl -sf "http://127.0.0.1:${port}/metrics" || true)"
  local frag pages free_pages decode_tps
  frag="$(echo "$metrics" | awk '/^rbitnet_core_kv_pool_fragmentation_permille /{print $2}')"
  pages="$(echo "$metrics" | awk '/^rbitnet_core_kv_pool_allocated_pages /{print $2}')"
  free_pages="$(echo "$metrics" | awk '/^rbitnet_core_kv_pool_free_pages /{print $2}')"
  decode_tps="$(echo "$metrics" | awk '/^rbitnet_inference_decode_tokens_per_sec /{print $2}')"
  frag="${frag:-0}"
  pages="${pages:-0}"
  free_pages="${free_pages:-0}"
  decode_tps="${decode_tps:-n/a}"

  local toks_total
  toks_total=$((concurrency * MAX_TOKENS))
  local tok_s
  tok_s="$(awk -v t="$toks_total" -v e="$elapsed" 'BEGIN{ if(e>0) printf "%.3f", t/e; else print 0 }')"

  printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$mode" "$concurrency" "$elapsed" "$tok_s" "$decode_tps" "$rss" "$pages/$free_pages" "$frag"

  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  trap - RETURN
}

echo
echo "## Paged KV E2E ($(date -u +%Y-%m-%d))"
echo
echo "- model: \`${RBITNET_MODEL}\`"
echo "- tokenizer: \`${RBITNET_TOKENIZER}\`"
echo "- max_tokens: ${MAX_TOKENS}"
echo "- prompt: \`${PROMPT}\`"
echo
echo '| mode | concurrency | wall_s | approx_tok/s | decode_tok/s_metric | rss_kb | pool_pages alloc/free | frag_permille |'
echo '|------|-------------|--------|--------------|---------------------|--------|-----------------------|---------------|'

port_base="${PORT_BASE:-18080}"
pi=0
for mode in $MODES; do
  for c in $CONCURRENCY_LIST; do
    port=$((port_base + pi))
    pi=$((pi + 1))
    run_mode "$mode" "$c" "$port" || true
  done
done

if [[ "${APPEND_RESULTS:-0}" == "1" ]]; then
  {
    echo
    echo "## Paged KV E2E ($(date -u +%Y-%m-%d))"
    echo
    echo "See \`scripts/bench_paged_kv.sh\` output above (APPEND_RESULTS=1)."
  } >>"$RESULTS_MD"
fi

if [[ -n "${RBITNET_GOLDEN_JSON:-}" ]]; then
  echo
  echo "== optional golden (dense then pool) =="
  unset RBITNET_KV_POOL RBITNET_LLAMA_PAGED_KV || true
  cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
  export RBITNET_LLAMA_PAGED_KV=1
  export RBITNET_KV_POOL=1
  cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
fi

echo "done."
