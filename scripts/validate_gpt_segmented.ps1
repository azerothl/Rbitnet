param(
    [string]$Library='target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll',
    [string]$Gguf='D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',
    [string]$Tokenizer='target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json',
    [switch]$SyntheticOnly
)
$ErrorActionPreference='Stop'
$env:RBITNET_CUDA_QUANT_LIB=(Resolve-Path -LiteralPath $Library).Path
$env:RAYON_NUM_THREADS='16'
$env:RBITNET_CUDA_DEVICE_BUDGET_MB='12288'
$env:RBITNET_CUDA_DEVICE_MARGIN_MB='256'
$env:RBITNET_HYBRID_MAX_VRAM_MB='12288'
$env:RBITNET_CUDA_QUANT_SMOKE='1'
$env:RBITNET_GPT_FULL_TEST='0'
$env:RBITNET_GPT_SEGMENTED_TEST='0'
cargo test -p bitnet-core --release --lib native::graph::gpu_full::tests -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
if ($SyntheticOnly) {exit 0}
$env:RBITNET_CUDA_QUANT_SMOKE='0'
$env:RBITNET_GPT_SEGMENTED_TEST='1'
$env:RBITNET_GPT_TEST_GGUF=(Resolve-Path -LiteralPath $Gguf).Path
$env:RBITNET_GPT_TEST_TOKENIZER=(Resolve-Path -LiteralPath $Tokenizer).Path
# Native TLS scratch remains accounted after a model unload. Each case has a
# fresh process; raising a configured cap with live allocations is forbidden.
foreach ($case in @(@{Cache=0;Cap=12288},@{Cache=8192;Cap=12288},@{Cache=16;Cap=12288},@{Cache=0;Cap=6144})) {
    $env:RBITNET_GPT_TEST_CACHE=[string]$case.Cache
    $env:RBITNET_CUDA_DEVICE_BUDGET_MB=[string]$case.Cap
    cargo test -p bitnet-core --release --lib native::graph::gpt_segmented_runtime_tests::opt_in_gpt_segmented_real -- --nocapture --test-threads=1
    if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
}
exit 0
