param([string]$Library='target/qwen-block/cuda/rbitnet_cuda_quant64.dll')
$ErrorActionPreference='Stop'
$env:RBITNET_CUDA_QUANT_LIB=(Resolve-Path -LiteralPath $Library).Path
$env:RBITNET_CUDA_SPLIT_KV='1'
$env:RBITNET_CUDA_QUANT_SMOKE='1'
cargo test -p bitnet-core --release --lib native::qwen -- --nocapture --test-threads=1
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:RBITNET_CUDA_QUANT_SMOKE='0'
$env:RBITNET_QWEN_BLOCK_TEST='1'
$env:RBITNET_CUDA_QWEN_PREFILL='1'
pwsh -NoProfile -File scripts/validate_qwen_full.ps1 -Library $Library
exit $LASTEXITCODE
