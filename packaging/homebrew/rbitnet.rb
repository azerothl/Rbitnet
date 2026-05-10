# Homebrew tap formula template for Rbitnet.
#
# Local test from the repository:
#   brew install --formula ./packaging/homebrew/rbitnet.rb
#
# Tap usage after copying this file to a tap:
#   brew tap <owner>/rbitnet
#   brew install rbitnet
#
# TODO(release): update version from the GitHub tag without leading "v".
# TODO(release): URLs must match release.yml asset names:
#   rbitnet-server-vX.Y.Z-linux-x86_64.tar.gz
#   rbitnet-server-vX.Y.Z-linux-aarch64.tar.gz
#   rbitnet-server-vX.Y.Z-macos-x86_64.tar.gz
#   rbitnet-server-vX.Y.Z-macos-arm64.tar.gz
# TODO(release): replace each sha256 with the uploaded asset SHA-256.
# See docs/RELEASE_PACKAGING.md before publishing this formula to a tap.
class Rbitnet < Formula
  desc "Pure Rust GGUF inference and OpenAI-compatible local HTTP server"
  homepage "https://github.com/azerothl/Rbitnet"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/azerothl/Rbitnet/releases/download/v0.1.0/rbitnet-server-v0.1.0-macos-arm64.tar.gz"
      sha256 "REPLACE_WITH_MACOS_ARM64_TAR_SHA256"
    else
      url "https://github.com/azerothl/Rbitnet/releases/download/v0.1.0/rbitnet-server-v0.1.0-macos-x86_64.tar.gz"
      sha256 "REPLACE_WITH_MACOS_X86_64_TAR_SHA256"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/azerothl/Rbitnet/releases/download/v0.1.0/rbitnet-server-v0.1.0-linux-aarch64.tar.gz"
      sha256 "REPLACE_WITH_LINUX_AARCH64_TAR_SHA256"
    else
      url "https://github.com/azerothl/Rbitnet/releases/download/v0.1.0/rbitnet-server-v0.1.0-linux-x86_64.tar.gz"
      sha256 "REPLACE_WITH_LINUX_X86_64_TAR_SHA256"
    end
  end

  def install
    bin.install "rbitnet"
    bin.install "rbitnet-server"
    bin.install "rbitnet-runner"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/rbitnet --version")
  end
end
