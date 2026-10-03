param(
    [string]$Library = 'target/qwen-full/cuda/rbitnet_cuda_quant64.dll',
    [string]$Gguf = 'D:/Rbitnet-benchmark-models/qwen35-2b/Qwen3.5-2B-Q8_0.gguf',
    [string]$Tokenizer = 'target/engine-benchmark/tokenizers/Qwen3.5-2B/tokenizer.json',
    [switch]$Eager,
    [switch]$Legacy
)
$ErrorActionPreference = 'Stop'
$env:RBITNET_CUDA_QUANT_LIB = (Resolve-Path -LiteralPath $Library).Path
$env:RBITNET_CUDA_QWEN_FULL = '1'
$env:RBITNET_REQUIRE_QWEN_FULL = if ($Legacy) { '0' } else { '1' }
$env:RBITNET_CUDA_QWEN_FULL_GRAPH = if ($Eager) { '0' } else { '1' }
$env:RBITNET_PREFIX_KV = '1'
$env:RBITNET_CUDA_PREFIX_MB = '256'
$env:RBITNET_CUDA_PREFIX_ENTRIES = '8'
$env:RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS = '16'
$env:RBITNET_QWEN_PREFIX_TEST = if ($Legacy) { '0' } else { '1' }
$env:RBITNET_QWEN_TEST_GGUF = (Resolve-Path -LiteralPath $Gguf).Path
$env:RBITNET_QWEN_TEST_TOKENIZER = (Resolve-Path -LiteralPath $Tokenizer).Path
$env:RBITNET_QWEN_SEQUENCE_JSON = (Resolve-Path -LiteralPath 'tests/data/golden/qwen35-2b-q8_0.sequence.json').Path
$env:RBITNET_QWEN_SEQUENCE_BACKEND = 'cuda'
$env:RBITNET_QWEN_REQUIRE_RESIDENT = '1'
$env:RAYON_NUM_THREADS = '16'
cargo test -p bitnet-core --release --lib qwen35::runtime::sequence_tests -- --nocapture --test-threads=1
exit $LASTEXITCODE
