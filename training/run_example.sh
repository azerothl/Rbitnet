#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/training/recipes/sft_lora.py" \
  --model-id "${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
  --dataset-jsonl "$ROOT/training/data/example.jsonl" \
  --output-dir "${OUTPUT_DIR:-$ROOT/out/sft-example}" \
  --max-steps "${MAX_STEPS:-20}" \
  --bf16 \
  "$@"
