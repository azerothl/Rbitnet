# Fair CPU comparison: llama.cpp llama-bench vs Rbitnet HTTP.
# Prerequisites: llama-bench.exe, rbitnet-server with RBITNET_MODEL set to the same GGUF.
#
# Example:
#   $env:RBITNET_GGUF = "C:\models\model.gguf"
#   $env:RBITNET_THREADS = "8"
#   $env:LLAMA_BENCH = "C:\path\llama.cpp\build\bin\Release\llama-bench.exe"
#   .\scripts\compare_llamacpp_rbitnet.ps1

$ErrorActionPreference = "Continue"
if (-not $env:RBITNET_GGUF) { throw "Set RBITNET_GGUF to the .gguf path" }
$threads = if ($env:RBITNET_THREADS) { $env:RBITNET_THREADS } else { "8" }
$bench = if ($env:LLAMA_BENCH) { $env:LLAMA_BENCH } else { "llama-bench" }

Write-Host "=== llama.cpp (llama-bench) ==="
if (Test-Path $bench) {
    & $bench -m $env:RBITNET_GGUF -t $threads -p 512 -n 64
} else {
    Write-Host "(skip: $bench not found; set LLAMA_BENCH)"
}

Write-Host ""
Write-Host "=== Rbitnet (HTTP) ==="
$prompt = "x" * 480
python (Join-Path $PSScriptRoot "bench_backend_compare.py") `
  --base-url $(if ($env:RBITNET_BASE_URL) { $env:RBITNET_BASE_URL } else { "http://127.0.0.1:8080" }) `
  --model $(if ($env:RBITNET_BENCH_MODEL) { $env:RBITNET_BENCH_MODEL } else { "rbitnet-llama" }) `
  --prompt $prompt `
  --max-tokens 64 `
  --temperature 0.0 `
  --runs 5 `
  --warmup 1
