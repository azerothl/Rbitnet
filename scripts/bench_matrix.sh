#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-rbitnet-stub}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
MAX_TOKENS="${MAX_TOKENS:-32}"
OUT="${OUT:-docs/BENCHMARKS_RESULTS.md}"
NO_START_SERVER="${NO_START_SERVER:-0}"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_health() {
  i=0
  while [ "$i" -lt 90 ]; do
    if command -v curl >/dev/null 2>&1 && curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    i=$((i + 1))
    sleep 0.5
  done
  echo "Timed out waiting for $BASE_URL/health" >&2
  return 1
}

if [ "$NO_START_SERVER" != "1" ]; then
  export RBITNET_STUB=1
  : "${RBITNET_BIND:=$(printf "%s" "$BASE_URL" | sed 's#^https\?://##')}"
  export RBITNET_BIND
  (cd "$ROOT_DIR" && cargo run -p bitnet-server --bin rbitnet-server --release >/tmp/rbitnet-bench-server.log 2>&1) &
  SERVER_PID="$!"
  wait_health
fi

JSON=$(
  cd "$ROOT_DIR" && python3 scripts/bench_backend_compare.py \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --runs "$RUNS" \
    --warmup "$WARMUP" \
    --max-tokens "$MAX_TOKENS"
)

SHA=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || printf "unknown")
RUSTC=$(rustc -V 2>/dev/null || printf "unknown")
STAMP=$(date "+%Y-%m-%d %H:%M:%S %z")
P50=$(printf "%s" "$JSON" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["p50_ms"], 2))')
P95=$(printf "%s" "$JSON" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["p95_ms"], 2))')
MEAN=$(printf "%s" "$JSON" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["mean_ms"], 2))')
TOKS=$(printf "%s" "$JSON" | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["mean_tok_s"], 2))')

cat >> "$ROOT_DIR/$OUT" <<EOF

## Local bench $STAMP

- Git SHA: \`$SHA\`
- Rust: \`$RUSTC\`
- Base URL: \`$BASE_URL\`
- Model: \`$MODEL\`
- Runs: \`$RUNS\` warmup: \`$WARMUP\` max_tokens: \`$MAX_TOKENS\`

| Host | Backend | Model | p50 ms | p95 ms | mean ms | mean tok/s | Notes |
|------|---------|-------|--------|--------|---------|------------|-------|
| Unix local | stub/http | \`$MODEL\` | $P50 | $P95 | $MEAN | $TOKS | Small reproducible smoke bench |
EOF

echo "Appended results to $ROOT_DIR/$OUT"
printf "%s\n" "$JSON"
