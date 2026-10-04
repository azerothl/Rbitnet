param(
    [string]$Library='target/gpt-full/cuda/rbitnet_cuda_quant64.dll',
    [string]$Gguf='D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',
    [string]$Tokenizer='target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json',
    [switch]$SyntheticOnly
)
$ErrorActionPreference='Stop'
$env:RBITNET_CUDA_QUANT_LIB=(Resolve-Path -LiteralPath $Library).Path
$env:RAYON_NUM_THREADS='16'
$env:RBITNET_CUDA_QUANT_SMOKE='1'
cargo test -p bitnet-core --release --lib native::graph::gpu_full::tests -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
if ($SyntheticOnly) {exit 0}
$env:RBITNET_CUDA_QUANT_SMOKE='0'
$env:RBITNET_GPT_FULL_TEST='1'
$env:RBITNET_GPT_TEST_GGUF=(Resolve-Path -LiteralPath $Gguf).Path
$env:RBITNET_GPT_TEST_TOKENIZER=(Resolve-Path -LiteralPath $Tokenizer).Path
cargo test -p bitnet-core --release --lib native::graph::tests::opt_in_gpt_full_real_teacher -- --nocapture --test-threads=1
exit $LASTEXITCODE
