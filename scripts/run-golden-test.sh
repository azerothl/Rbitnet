#!/usr/bin/env bash
set -euo pipefail
# Run optional Llama / Qwen3 golden test (requires GGUF + tokenizer + golden JSON on disk).
# Usage:
#   export RBITNET_GOLDEN_JSON=tests/data/golden/my.golden.json
#   export RBITNET_TEST_GGUF=/path/model.gguf
#   export RBITNET_TOKENIZER=/path/tokenizer.json
#   # optional: export RBITNET_ARCHITECTURE=qwen3
#   ./scripts/run-golden-test.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
