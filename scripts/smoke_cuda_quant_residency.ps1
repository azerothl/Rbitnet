# Hardware smoke for #22 Gate E — device-resident quantized Llama matvec (Windows).
# CI must NOT run this as a required job.
param(
    [switch]$Bench,
    [switch]$BuildLib
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($BuildLib) {
    Write-Host "== Build librbitnet_cuda_quant =="
    & "$Root\scripts\build_cuda_quant.ps1"
    $buildDir = Join-Path $Root "native\cuda_quant\build"
    $env:PATH = "$buildDir;$env:PATH"
    $dll = Join-Path $buildDir "rbitnet_cuda_quant64.dll"
    if (Test-Path $dll) {
        $env:RBITNET_CUDA_QUANT_LIB = (Resolve-Path $dll).Path
    }
}

Write-Host "== CPU golden (always) =="
cargo test -p bitnet-core --test cuda_quant_residency --test backend_conformance

if ($env:RBITNET_CUDA_QUANT_LIB -or (Get-Command "rbitnet_cuda_quant64.dll" -ErrorAction SilentlyContinue)) {
    Write-Host "== Opt-in device-resident quant kernel smoke =="
    $env:RBITNET_CUDA_QUANT_SMOKE = "1"
    cargo test -p bitnet-core --test cuda_quant_residency opt_in_device_resident_quant_kernel_when_lib_present -- --nocapture
} else {
    Write-Host "Skip device kernel smoke: build with -BuildLib or set RBITNET_CUDA_QUANT_LIB"
}

if ($Bench -or $env:RBITNET_BENCH_CUDA -eq "1") {
    Write-Host "== Opt-in CUDA benches =="
    $env:RBITNET_BENCH_CUDA = "1"
    cargo bench -p bitnet-core --bench kernels -- --warm-up-time 1 --measurement-time 3
}

Write-Host "See docs/GPU_NATIVE_ROADMAP.md Gate E checklist."
