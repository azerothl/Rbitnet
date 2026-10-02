#!/usr/bin/env bash
# Microbench: fused dense_matvec_multi_seq vs sequential @ batch 4/8 (#46).
#
# Usage:
#   ./scripts/bench_fused_multi_seq.sh
#   cargo bench -p bitnet-core --bench kernels -- fused_multi_seq
#
# This measures the CPU kernel building block only — not e2e HTTP concurrency.
# See docs/FUSED_MULTI_SEQ.md for the stall decision.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== fused_batch unit equivalence =="
cargo test -p bitnet-core fused_batch -- --nocapture

echo ""
echo "== criterion kernel microbench (fused_multi_seq filter) =="
echo "Release profile recommended; first run may compile for a while."
cargo bench -p bitnet-core --bench kernels -- fused_multi_seq

echo ""
echo "Note: e2e concurrency ≥4 tok/s gain is STALLED — see docs/FUSED_MULTI_SEQ.md"
echo "GPU fused waves remain #22."
