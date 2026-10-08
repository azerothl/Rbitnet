# Install the rbitnet CLI from this checkout, or clone the repo when the script
# is not already inside one (same behavior as scripts/install.sh).
#
# From any PowerShell window (do not nest powershell.exe; some sessions deny it):
#   irm https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.ps1 | iex
# From the repository root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\install.ps1

param(
    [switch] $NoLocked
)

$ErrorActionPreference = "Stop"
# git clone and cargo write progress to stderr. Do not treat that as a failed install.
$PSNativeCommandUseErrorActionPreference = $false

function Test-RbitnetCheckout([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    return Test-Path -LiteralPath (Join-Path $Path "crates\rbitnet-cli\Cargo.toml")
}

function Resolve-RbitnetRoot {
    if (Test-RbitnetCheckout $env:RBITNET_REPO_ROOT) {
        return (Resolve-Path -LiteralPath $env:RBITNET_REPO_ROOT).Path
    }
    if ($PSScriptRoot -and (Test-RbitnetCheckout (Join-Path $PSScriptRoot ".."))) {
        return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    }
    $cwd = (Get-Location).Path
    if (Test-RbitnetCheckout $cwd) {
        return (Resolve-Path -LiteralPath $cwd).Path
    }
    return $null
}

$CleanupDir = $null
try {
    $RootDir = Resolve-RbitnetRoot
    if (-not $RootDir) {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Write-Error "This script is not running inside a Rbitnet checkout and git was not found. Install git, or run from an existing checkout (repository root: .\scripts\install.ps1), or set `$env:RBITNET_REPO_ROOT to that checkout."
        }
        $CleanupDir = Join-Path ([System.IO.Path]::GetTempPath()) ("rbitnet-install-" + [guid]::NewGuid().ToString("n"))
        New-Item -ItemType Directory -Path $CleanupDir | Out-Null
        $RootDir = Join-Path $CleanupDir "Rbitnet"
        Write-Host "No local checkout found; cloning Rbitnet into $RootDir"
        & git clone --depth 1 https://github.com/azerothl/Rbitnet $RootDir
        if ($LASTEXITCODE -ne 0) {
            Write-Error "git clone failed (exit $LASTEXITCODE)."
        }
    }

    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Error "Rust/Cargo not found. Install rustup first: https://rustup.rs/"
    }

    Write-Host "Installing rbitnet CLI from $RootDir"
    $cargoArgs = @("install", "--path", (Join-Path $RootDir "crates\rbitnet-cli"))
    if (-not $NoLocked) {
        $cargoArgs += "--locked"
    }
    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "cargo install failed (exit $LASTEXITCODE)."
    }

    Write-Host ""
    Write-Host "Installed: rbitnet"
    Write-Host "Local release binaries are built with:"
    Write-Host "  cargo build -p bitnet-server -p rbitnet-cli --release --locked"
    Write-Host "  $RootDir\target\release\rbitnet.exe"
    Write-Host "  $RootDir\target\release\rbitnet-server.exe"
    Write-Host ""
    Write-Host "Tagged GitHub releases publish zip assets named:"
    Write-Host "  rbitnet-server-vX.Y.Z-windows-x86_64.zip"
    Write-Host "Direct releases: https://github.com/azerothl/Rbitnet/releases"
    Write-Host "Try: rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    Write-Host "Docs: $RootDir\README.md and $RootDir\docs\USAGE.md"
} finally {
    if ($CleanupDir -and (Test-Path -LiteralPath $CleanupDir)) {
        Remove-Item -LiteralPath $CleanupDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
