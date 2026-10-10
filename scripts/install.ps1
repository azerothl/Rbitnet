# Install Rbitnet release binaries. No Rust, Git, or repository checkout.
#
# From any PowerShell window:
#   irm https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.ps1 | iex
# Uninstall:
#   $env:RBITNET_UNINSTALL=1; irm https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.ps1 | iex
#   .\scripts\install.ps1 -Uninstall
# A specific tag (without the v prefix, or with it):
#   .\scripts\install.ps1 -Version 0.2.0

param(
    [switch] $Uninstall,
    [string] $Version = "",
    [string] $InstallDir = "",
    [switch] $NoPath
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if ($env:RBITNET_UNINSTALL -eq "1") { $Uninstall = $true }
if ([string]::IsNullOrWhiteSpace($Version) -and $env:RBITNET_INSTALL_VERSION) {
    $Version = $env:RBITNET_INSTALL_VERSION
}

$Repo = "azerothl/Rbitnet"
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "Rbitnet"
}

function Get-UserPathEntries {
    $raw = [Environment]::GetEnvironmentVariable("Path", "User")
    if ([string]::IsNullOrWhiteSpace($raw)) { return @() }
    return @($raw -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Set-UserPathEntries([string[]] $Entries) {
    $clean = @($Entries | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $value = ($clean -join ';')
    [Environment]::SetEnvironmentVariable("Path", $value, "User")
}

function Remove-InstallDirFromUserPath([string] $Dir) {
    $full = [System.IO.Path]::GetFullPath($Dir).TrimEnd('\')
    $kept = @(Get-UserPathEntries | Where-Object {
        [System.IO.Path]::GetFullPath($_).TrimEnd('\') -ne $full
    })
    Set-UserPathEntries $kept
}

function Add-InstallDirToUserPath([string] $Dir) {
    $full = [System.IO.Path]::GetFullPath($Dir).TrimEnd('\')
    $entries = @(Get-UserPathEntries)
    $already = $false
    foreach ($entry in $entries) {
        if ([System.IO.Path]::GetFullPath($entry).TrimEnd('\') -eq $full) {
            $already = $true
            break
        }
    }
    if (-not $already) {
        $entries = @($full) + $entries
        Set-UserPathEntries $entries
    }
}

$BinaryNames = @(
    "rbitnet.exe",
    "rbitnet-server.exe",
    "rbitnet-runner.exe",
    "rbitnet-proxy.exe"
)

if ($Uninstall) {
    if (Test-Path -LiteralPath $InstallDir) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force
    }
    if (-not $NoPath) {
        Remove-InstallDirFromUserPath $InstallDir
    }
    Write-Host "Removed $InstallDir"
    Write-Host "Open a new terminal. rbitnet is no longer on PATH."
    return
}

$headers = @{
    "User-Agent" = "rbitnet-install"
    "Accept"     = "application/vnd.github+json"
}

$tag = $Version.Trim()
if ($tag -ne "" -and -not $tag.StartsWith("v")) {
    $tag = "v$tag"
}
if ($tag -eq "") {
    $release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Repo/releases/latest"
} else {
    $release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Repo/releases/tags/$tag"
}
$tag = [string] $release.tag_name
$assetName = "rbitnet-server-$tag-windows-x86_64.zip"
$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
if (-not $asset) {
    Write-Error "Release $tag has no $assetName. Published assets: $($release.assets.name -join ', ')"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
foreach ($name in $BinaryNames) {
    $existing = Join-Path $InstallDir $name
    if (Test-Path -LiteralPath $existing) {
        Remove-Item -LiteralPath $existing -Force
    }
}

$zip = Join-Path ([System.IO.Path]::GetTempPath()) ("rbitnet-" + [guid]::NewGuid().ToString("n") + ".zip")
try {
    Write-Host "Downloading $($asset.browser_download_url)"
    Invoke-WebRequest -Headers @{ "User-Agent" = "rbitnet-install" } -Uri $asset.browser_download_url -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $InstallDir -Force
} finally {
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
}

$catalogUrl = "https://raw.githubusercontent.com/$Repo/$tag/data/compatible_models.json"
$catalogPath = Join-Path $InstallDir "compatible_models.json"
Write-Host "Downloading catalog $catalogUrl"
Invoke-WebRequest -Headers @{ "User-Agent" = "rbitnet-install" } -Uri $catalogUrl -OutFile $catalogPath

foreach ($name in $BinaryNames) {
    if (-not (Test-Path -LiteralPath (Join-Path $InstallDir $name))) {
        Write-Error "Archive $assetName did not contain $name"
    }
}

if (-not $NoPath) {
    Add-InstallDirToUserPath $InstallDir
}

Write-Host ""
Write-Host "Installed $tag into $InstallDir"
Write-Host "  rbitnet.exe"
Write-Host "  rbitnet-server.exe"
Write-Host "  rbitnet-runner.exe"
Write-Host "  rbitnet-proxy.exe"
Write-Host "  compatible_models.json"
if ($NoPath) {
    Write-Host "PATH was not changed (-NoPath)."
} else {
    Write-Host "Added to the user PATH. Open a new terminal, then run: rbitnet --version"
}
Write-Host "Uninstall: `$env:RBITNET_UNINSTALL=1; irm https://raw.githubusercontent.com/$Repo/main/scripts/install.ps1 | iex"
