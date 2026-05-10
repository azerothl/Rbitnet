#!/usr/bin/env sh
set -eu

ROOT_DIR=${RBITNET_REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
CLEANUP_DIR=""

cleanup() {
  if [ -n "$CLEANUP_DIR" ] && [ -d "$CLEANUP_DIR" ]; then
    rm -rf "$CLEANUP_DIR"
  fi
}
trap cleanup EXIT INT TERM

if [ ! -d "$ROOT_DIR/crates/rbitnet-cli" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "This script is not running inside a Rbitnet checkout and git was not found." >&2
    echo "Install git or run from an existing checkout with RBITNET_REPO_ROOT=/path/to/Rbitnet." >&2
    exit 1
  fi
  CLEANUP_DIR=$(mktemp -d)
  echo "No local checkout found; cloning Rbitnet into $CLEANUP_DIR"
  git clone --depth 1 https://github.com/azerothl/Rbitnet "$CLEANUP_DIR/Rbitnet"
  ROOT_DIR="$CLEANUP_DIR/Rbitnet"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust/Cargo not found. Install rustup first: https://rustup.rs/" >&2
  exit 1
fi

echo "Installing rbitnet CLI from $ROOT_DIR"
cargo install --path "$ROOT_DIR/crates/rbitnet-cli" --locked

echo
echo "Installed: rbitnet"
echo "Local release binaries are built with:"
echo "  cargo build -p bitnet-server -p rbitnet-cli --release --locked"
echo "  $ROOT_DIR/target/release/rbitnet"
echo "  $ROOT_DIR/target/release/rbitnet-server"
echo
echo "Tagged GitHub releases publish tarballs named:"
echo "  rbitnet-server-vX.Y.Z-linux-\$(uname -m).tar.gz"
echo "  rbitnet-server-vX.Y.Z-macos-\$(uname -m).tar.gz"
echo "Direct releases: https://github.com/azerothl/Rbitnet/releases"
echo "Try: rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
echo "Docs: $ROOT_DIR/README.md and $ROOT_DIR/docs/USAGE.md"
