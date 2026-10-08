#!/usr/bin/env bash
# Copy reference pages into website/docs/content.
# user-guide.md and advanced-guide.md are written for the site. Do not overwrite them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/website/docs/content"
mkdir -p "$DEST"

copy() {
  local src="$1"
  local slug="$2"
  if [[ ! -f "$src" ]]; then
    echo "missing: $src" >&2
    exit 1
  fi
  cp "$src" "$DEST/$slug.md"
  echo "synced $slug"
}

copy "$ROOT/docs/GET_STARTED.md" "get-started"
copy "$ROOT/docs/USAGE.md" "usage"
copy "$ROOT/docs/STATUS_AND_ROADMAP.md" "status-and-roadmap"
copy "$ROOT/docs/NATIVE_FIRST.md" "native-first"
copy "$ROOT/docs/TRAINING_AND_COMPATIBILITY.md" "training-and-compatibility"
copy "$ROOT/docs/UNSLOTH_TO_RBITNET.md" "unsloth-to-rbitnet"
copy "$ROOT/docs/INTEGRATIONS.md" "integrations"
copy "$ROOT/docs/ENV_REFERENCE.md" "env-reference"
copy "$ROOT/docs/LIMITATIONS.md" "limitations"
copy "$ROOT/docs/BITNET_NATIVE.md" "bitnet-native"
copy "$ROOT/docs/AKASHA_INFER.md" "akasha-infer"
copy "$ROOT/docs/DEPLOYMENT.md" "deployment"
copy "$ROOT/docs/BRAND.md" "brand"
copy "$ROOT/docs/BENCHMARKS.md" "benchmarks"
copy "$ROOT/CHANGELOG.md" "changelog"

# Secondary French quickstart (linked from EN get-started)
copy "$ROOT/docs/DEMARRAGE_5MIN.md" "demarrage-5min"

echo "done -> $DEST"
