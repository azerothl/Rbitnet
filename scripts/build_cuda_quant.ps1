# Build optional librbitnet_cuda_quant / rbitnet_cuda_quant64.dll (#22 Gate E).
# Requires NVIDIA CUDA Toolkit (nvcc) and MSVC (cl.exe) on Windows.
param(
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) {
    $OutDir = Join-Path $Root "native\cuda_quant\build"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Import-VsDevEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { return }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { return }
    $envDump = cmd /c "`"$vcvars`" >NUL && set"
    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

Import-VsDevEnvironment
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "cl.exe not found. Install Visual Studio C++ tools (MSVC) or open a Developer PowerShell."
}

$Nvcc = $null
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $Nvcc = (Get-Command nvcc).Source
} elseif ($env:CUDA_PATH) {
    $cand = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $cand) { $Nvcc = $cand }
}
if (-not $Nvcc) {
    Write-Error "nvcc not found. Install CUDA Toolkit or set CUDA_PATH."
}

$Src = Join-Path $Root "native\cuda_quant\src\quant_matvec.cu"
$Inc = Join-Path $Root "native\cuda_quant\include"
$OutDll = Join-Path $OutDir "rbitnet_cuda_quant64.dll"

Write-Host "nvcc: $Nvcc"
Write-Host "cl:   $((Get-Command cl.exe).Source)"
Write-Host "out:  $OutDll"

& $Nvcc `
    -shared `
    -O3 `
    -std=c++17 `
    -I $Inc `
    -o $OutDll `
    $Src `
    "-gencode=arch=compute_89,code=sm_89" `
    "-gencode=arch=compute_86,code=sm_86" `
    "-gencode=arch=compute_75,code=sm_75" `
    -Xcompiler "/MD /EHsc" `
    -lcudart

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Copy-Item -Force $OutDll (Join-Path $OutDir "rbitnet_cuda_quant.dll")
Write-Host "Built OK. Add to PATH or set RBITNET_CUDA_QUANT_LIB=$OutDll"
Get-Item $OutDll | Format-List FullName, Length, LastWriteTime
