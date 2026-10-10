#!/usr/bin/env sh
# Install Rbitnet release binaries. No Rust, Git, or repository checkout.
#
#   curl -fsSL https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.sh | sh
#   ./scripts/install.sh --uninstall
#   ./scripts/install.sh --version 0.2.0
set -eu

REPO="azerothl/Rbitnet"
UNINSTALL=0
VERSION=""
NO_PATH=0
INSTALL_DIR="${RBITNET_INSTALL_DIR:-$HOME/.local/share/rbitnet}"

while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall) UNINSTALL=1 ;;
    --version)
      shift
      VERSION="${1:-}"
      ;;
    --version=*) VERSION="${1#--version=}" ;;
    --no-path) NO_PATH=1 ;;
    --prefix)
      shift
      INSTALL_DIR="${1:-}"
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

path_files() {
  printf '%s\n' "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zprofile"
}

remove_path_lines() {
  file="$1"
  [ -f "$file" ] || return 0
  tmp="$(mktemp)"
  grep -v '# rbitnet-installer' "$file" >"$tmp" || true
  mv "$tmp" "$file"
}

add_path_line() {
  file="$1"
  mkdir -p "$(dirname "$file")"
  touch "$file"
  if grep -q '# rbitnet-installer' "$file" 2>/dev/null; then
    return 0
  fi
  printf '\nexport PATH="%s:$PATH" # rbitnet-installer\n' "$INSTALL_DIR" >>"$file"
}

if [ "$UNINSTALL" -eq 1 ]; then
  rm -rf "$INSTALL_DIR"
  if [ "$NO_PATH" -eq 0 ]; then
    for file in $(path_files); do
      remove_path_lines "$file"
    done
  fi
  echo "Removed $INSTALL_DIR"
  echo "Open a new terminal. rbitnet is no longer on PATH."
  exit 0
fi

os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Linux) platform="linux-$arch" ; ext="tar.gz" ;;
  Darwin) platform="macos-$arch" ; ext="tar.gz" ;;
  *)
    echo "unsupported OS: $os (use scripts/install.ps1 on Windows)" >&2
    exit 1
    ;;
esac

ua="rbitnet-install"
if [ -n "$VERSION" ]; then
  tag="$VERSION"
  case "$tag" in
    v*) ;;
    *) tag="v$tag" ;;
  esac
  release_url="https://api.github.com/repos/$REPO/releases/tags/$tag"
else
  release_url="https://api.github.com/repos/$REPO/releases/latest"
fi

release_json="$(mktemp)"
curl -fsSL -A "$ua" -H "Accept: application/vnd.github+json" "$release_url" -o "$release_json"
tag="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$release_json" | head -n 1)"
asset="rbitnet-server-${tag}-${platform}.${ext}"
download="$(sed -n "s/.*\"browser_download_url\"[[:space:]]*:[[:space:]]*\"\\([^\"]*${asset}\\)\".*/\\1/p" "$release_json" | head -n 1)"
rm -f "$release_json"
if [ -z "$tag" ] || [ -z "$download" ]; then
  echo "Release has no $asset" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
for name in rbitnet rbitnet-server rbitnet-runner rbitnet-proxy; do
  rm -f "$INSTALL_DIR/$name"
done

tmp="$(mktemp)"
echo "Downloading $download"
curl -fL -A "$ua" "$download" -o "$tmp"
tar -xzf "$tmp" -C "$INSTALL_DIR"
rm -f "$tmp"

catalog="$INSTALL_DIR/compatible_models.json"
echo "Downloading catalog for $tag"
curl -fsSL -A "$ua" "https://raw.githubusercontent.com/$REPO/$tag/data/compatible_models.json" -o "$catalog"

for name in rbitnet rbitnet-server rbitnet-runner rbitnet-proxy; do
  if [ ! -x "$INSTALL_DIR/$name" ] && [ ! -f "$INSTALL_DIR/$name" ]; then
    echo "Archive $asset did not contain $name" >&2
    exit 1
  fi
  chmod +x "$INSTALL_DIR/$name"
done

if [ "$NO_PATH" -eq 0 ]; then
  for file in $(path_files); do
    add_path_line "$file"
  done
fi

echo
echo "Installed $tag into $INSTALL_DIR"
echo "  rbitnet"
echo "  rbitnet-server"
echo "  rbitnet-runner"
echo "  rbitnet-proxy"
echo "  compatible_models.json"
if [ "$NO_PATH" -eq 0 ]; then
  echo "Added to the user PATH via ~/.profile, ~/.bashrc and ~/.zprofile."
  echo "Open a new terminal, then run: rbitnet --version"
else
  echo "PATH was not changed (--no-path)."
fi
echo "Uninstall: curl -fsSL https://raw.githubusercontent.com/$REPO/main/scripts/install.sh | sh -s -- --uninstall"
