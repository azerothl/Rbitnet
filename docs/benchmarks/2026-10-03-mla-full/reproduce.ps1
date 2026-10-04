param(
    [string]$Python='C:/Users/azero/anaconda3/python.exe',
    [string]$Config='docs/benchmarks/2026-10-03-parity-round2/manifest.json',
    [string]$LegacyLibrary='target/device-memory/cuda-final/rbitnet_cuda_quant64.dll'
)
$ErrorActionPreference='Stop'
$lot='target/mla-full-reproduction'
$library="$lot/cuda/rbitnet_cuda_quant64.dll"
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir "$lot/cuda"
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
cargo build -p rbitnet-cli --release
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:RBITNET_CUDA_QUANT_LIB=(Resolve-Path -LiteralPath $library).Path
$env:RBITNET_CUDA_DEVICE_BUDGET_MB='12288'
$env:RBITNET_CUDA_DEVICE_MARGIN_MB='256'
$env:RAYON_NUM_THREADS='16'
pwsh -NoProfile -File scripts/validate_mla_full.ps1 -Library $library
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:RBITNET_CUDA_QUANT_SMOKE='1'
$env:RBITNET_MOE_CACHE_TEST='1'
$env:RBITNET_GPT_FULL_TEST='0'
$env:RBITNET_GPT_LAYER_DIAG='0'
$env:RBITNET_MLA_FULL_TEST='0'
$env:RBITNET_MLA_FALLBACK_TEST='0'
$env:RBITNET_TEST_GGUF='D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf'
cargo test -p bitnet-core --release --lib native -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:RBITNET_CUDA_QUANT_SMOKE='0'
$env:RBITNET_MOE_CACHE_TEST='0'
$env:RBITNET_MLA_FULL_TEST='0'
$env:RBITNET_MLA_FALLBACK_TEST='0'
& $Python docs/benchmarks/2026-10-03-mla-full/lifecycle.py --config $Config --binary target/release/rbitnet.exe --library $library --legacy-library $LegacyLibrary --output-dir "$lot/lifecycle"
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
foreach ($cache in @(0,8192)) {
    # Run sequentially; no builds, CUDA tests or reference engines during timing.
    & $Python scripts/benchmark_cache_stack.py --config $Config --model glm47-flash --backend gpu --binary target/release/rbitnet.exe --library $library --cycles 3 --mla-full --split-kv --moe-cache $cache --device-mib 12288 --output-dir "$lot/http-$cache"
    if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
}
foreach ($case in @(@{Cache=8192;Cap=12288},@{Cache=16;Cap=6144})) {
    $fallbackArgs = if ($case.Cache -eq 16) { @('--expect-cpu-routed') } else { @() }
    & $Python docs/benchmarks/2026-10-03-mla-full/validate_cache_streaming.py --config $Config --binary target/release/rbitnet.exe --library $library --mla-full --moe-cache $case.Cache --split-kv --device-mib $case.Cap --output-dir "$lot/streaming-$($case.Cache)" @fallbackArgs
    if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
}
