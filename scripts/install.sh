#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust/Cargo not found. Install rustup first: https://rustup.rs/" >&2
  exit 1
fi

echo "Installing rbitnet CLI from $ROOT_DIR"
cargo install --path "$ROOT_DIR/crates/rbitnet-cli" --locked

echo
echo "Installed: rbitnet"
echo "Try: rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
echo "Docs: $ROOT_DIR/README.md and $ROOT_DIR/docs/USAGE.md"
