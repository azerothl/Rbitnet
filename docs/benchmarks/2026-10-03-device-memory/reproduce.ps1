param(
    [string]$Python='C:/Users/azero/anaconda3/python.exe',
    [string]$Config='docs/benchmarks/2026-10-03-parity-round2/manifest.json',
    [string]$LegacyLibrary='target/gpt-full/cuda/rbitnet_cuda_quant64.dll'
)
$ErrorActionPreference='Stop'
$lot='target/device-memory-reproduction'
$library="$lot/cuda/rbitnet_cuda_quant64.dll"
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir "$lot/cuda"
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
cargo build -p rbitnet-cli --release
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:RBITNET_CUDA_QUANT_LIB=(Resolve-Path -LiteralPath $library).Path
$env:RBITNET_CUDA_DEVICE_BUDGET_MB='12288'
$env:RBITNET_CUDA_DEVICE_MARGIN_MB='256'
$env:RBITNET_CUDA_QUANT_SMOKE='1'
$env:RBITNET_CUDA_MEMORY_TEST='1'
$env:RAYON_NUM_THREADS='16'
cargo test -p bitnet-core --release --lib backend::device_memory -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
cargo test -p bitnet-core --release --lib native -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:RBITNET_CUDA_QUANT_SMOKE='0'
$env:RBITNET_MOE_CACHE_TEST='1'
$env:RBITNET_TEST_GGUF='D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf'
cargo test -p bitnet-core --release --lib native::expert_cache -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
pwsh -NoProfile -File scripts/validate_gpt_full.ps1 -Library $library
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
& $Python scripts/benchmark_moe_budget.py --config $Config --binary target/release/rbitnet.exe --library $library --cycles 3 --output-dir "$lot/ablation"
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
foreach ($model in @('gpt-oss-20b','glm47-flash')) {
    & $Python scripts/validate_cache_streaming.py --config $Config --binary target/release/rbitnet.exe --library $library --moe-cache 8192 --moe-model $model --device-mib 12288 --output-dir "$lot/streaming-$model"
    if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
}
& $Python scripts/benchmark_moe_budget.py --config $Config --binary target/release/rbitnet.exe --library $library --cycles 2 --cache-mib 0 4096 --device-mib 6144 --output-dir "$lot/cap6144"
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
& $Python scripts/validate_device_memory.py --config $Config --binary target/release/rbitnet.exe --library $library --legacy-library $LegacyLibrary --output-dir "$lot/refusals"
exit $LASTEXITCODE
