param(
    [string] $BaseUrl = "http://127.0.0.1:8080",
    [string] $Model = "rbitnet-stub",
    [int] $Runs = 3,
    [int] $Warmup = 1,
    [int] $MaxTokens = 32,
    [string] $Out = "docs/BENCHMARKS_RESULTS.md",
    [switch] $NoStartServer
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutPath = Join-Path $RootDir $Out

function Wait-RbitnetHealth {
    param([string] $Url, [int] $TimeoutSeconds = 45)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$Url/health" -TimeoutSec 2 | Out-Null
            return
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Timed out waiting for $Url/health"
}

$server = $null
try {
    if (-not $NoStartServer) {
        $env:RBITNET_STUB = "1"
        if (-not $env:RBITNET_BIND) {
            $env:RBITNET_BIND = $BaseUrl -replace '^https?://', ''
        }
        $server = Start-Process -FilePath "cargo" `
            -ArgumentList @("run", "-p", "bitnet-server", "--bin", "rbitnet-server", "--release") `
            -WorkingDirectory $RootDir `
            -PassThru `
            -WindowStyle Hidden
        Wait-RbitnetHealth -Url $BaseUrl
    }

    $json = python (Join-Path $RootDir "scripts/bench_backend_compare.py") `
        --base-url $BaseUrl `
        --model $Model `
        --runs $Runs `
        --warmup $Warmup `
        --max-tokens $MaxTokens
    $result = $json | ConvertFrom-Json
    $sha = (git -C $RootDir rev-parse --short HEAD) 2>$null
    if (-not $sha) { $sha = "unknown" }
    $rustc = (rustc -V) 2>$null
    if (-not $rustc) { $rustc = "unknown" }
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"

    $section = @"

## Local bench $stamp

- Git SHA: `$sha`
- Rust: `$rustc`
- Base URL: `$BaseUrl`
- Model: `$Model`
- Runs: `$Runs` warmup: `$Warmup` max_tokens: `$MaxTokens`

| Host | Backend | Model | p50 ms | p95 ms | mean ms | mean tok/s | Notes |
|------|---------|-------|--------|--------|---------|------------|-------|
| Windows local | stub/http | `$Model` | $([math]::Round($result.p50_ms, 2)) | $([math]::Round($result.p95_ms, 2)) | $([math]::Round($result.mean_ms, 2)) | $([math]::Round($result.mean_tok_s, 2)) | Small reproducible smoke bench |
"@
    Add-Content -Path $OutPath -Value $section
    Write-Host "Appended results to $OutPath"
    Write-Host $json
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
