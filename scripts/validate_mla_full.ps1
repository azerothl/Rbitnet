param(
    [string]$Library='target/mla-full/cuda/rbitnet_cuda_quant64.dll',
    [string]$Gguf='D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf',
    [string]$Tokenizer='target/engine-benchmark/tokenizers/GLM-4.7-Flash/tokenizer.json',
    [switch]$SyntheticOnly
)
$ErrorActionPreference='Stop'
$env:RBITNET_CUDA_QUANT_LIB=(Resolve-Path -LiteralPath $Library).Path
$env:RAYON_NUM_THREADS='16'
$env:RBITNET_CUDA_QUANT_SMOKE='1'
cargo test -p bitnet-core --release --lib native::graph::gpu_mla::tests -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
if ($SyntheticOnly) {exit 0}
$env:RBITNET_CUDA_QUANT_SMOKE='0'
$env:RBITNET_MLA_FULL_TEST='1'
$env:RBITNET_MLA_FALLBACK_TEST='1'
$env:RBITNET_MLA_TEST_GGUF=(Resolve-Path -LiteralPath $Gguf).Path
$env:RBITNET_MLA_TEST_TOKENIZER=(Resolve-Path -LiteralPath $Tokenizer).Path
$env:RBITNET_MAX_SEQ='2048'
$env:RBITNET_CUDA_DEVICE_BUDGET_MB='12288'
$env:RBITNET_CUDA_DEVICE_MARGIN_MB='256'
cargo test -p bitnet-core --release --lib native::graph::mla_runtime_tests -- --nocapture --test-threads=1
exit $LASTEXITCODE
