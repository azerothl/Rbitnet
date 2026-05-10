param(
    [switch] $NoLocked
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust/Cargo not found. Install rustup first: https://rustup.rs/"
}

Write-Host "Installing rbitnet CLI from $RootDir"
$args = @("install", "--path", (Join-Path $RootDir "crates/rbitnet-cli"))
if (-not $NoLocked) {
    $args += "--locked"
}
cargo @args

Write-Host ""
Write-Host "Installed: rbitnet"
Write-Host "Try: rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
Write-Host "Docs: $RootDir\README.md and $RootDir\docs\USAGE.md"
