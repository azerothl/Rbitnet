# Homebrew tap formula template for Rbitnet.
#
# Local test from the repository:
#   brew install --formula ./packaging/homebrew/rbitnet.rb
#
# Tap usage after copying this file to a tap:
#   brew tap <owner>/rbitnet
#   brew install rbitnet
#
# Update version, URLs, and sha256 values for each tagged GitHub release.
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
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/rbitnet --version")
  end
end
