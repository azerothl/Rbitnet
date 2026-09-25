#!/usr/bin/env bash
# Smoke: OpenAI-compatible surface Akasha depends on (stub-friendly).
# Usage:
#   RBITNET_STUB=1 cargo run -p bitnet-server --release &
#   ./scripts/smoke_openai.sh
# Or against an already-running server:
#   RBITNET_BASE_URL=http://127.0.0.1:8080 ./scripts/smoke_openai.sh

set -euo pipefail
BASE="${RBITNET_BASE_URL:-http://127.0.0.1:8080}"

echo "=== GET /health ==="
curl -fsS "$BASE/health"
echo

echo "=== GET /ready ==="
curl -fsS "$BASE/ready"
echo

echo "=== GET /v1/models ==="
curl -fsS "$BASE/v1/models" | head -c 400
echo

echo "=== POST /v1/chat/completions ==="
curl -fsS "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"rbitnet-stub","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' \
  | head -c 600
echo

echo "=== GET /metrics (Akasha series sample) ==="
curl -fsS "$BASE/metrics" | grep -E 'rbitnet_(chat_requests_total|inference_ttft_ms_|completion_tokens_total|core_prefix_cache_hits_total)' | head -20
echo
echo "smoke_openai: OK"
