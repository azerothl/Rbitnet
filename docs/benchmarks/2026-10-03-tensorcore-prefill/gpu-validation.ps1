$ErrorActionPreference='Stop'
$env:RBITNET_CUDA_QUANT_LIB='E:\devs\Rbitnet\target\tensorcore\cuda\rbitnet_cuda_quant64.dll'
$env:RBITNET_PREFIX_KV='1'
$env:RBITNET_CUDA_PREFILL='1'
$env:RBITNET_CUDA_PREFILL_TF32X3='1'
$env:RBITNET_CUDA_SPLIT_KV='1'
$env:RBITNET_SPECULATIVE_PLD='1'
$env:RBITNET_SPECULATIVE='0'
$env:RBITNET_CUDA_PLD_TEST='1'
$env:RBITNET_CUDA_PREFIX_TEST='1'
$env:RBITNET_CUDA_PREFIX_ENTRIES='2'
$env:RBITNET_CUDA_PREFIX_MB='8'
$env:RBITNET_REQUIRE_RESIDENT='1'
$env:RBITNET_SEQUENCE_BACKEND='cuda'
$env:RBITNET_LLAMA_SEQUENCE_JSON='E:\devs\Rbitnet\tests\data\golden\llama32-1b-instruct-q4-k-m.sequence.json'
$env:RBITNET_TEST_GGUF='E:\devs\Rbitnet\models\exported-llama\model.gguf'
$env:RBITNET_TOKENIZER='E:\devs\Rbitnet\models\exported-llama\tokenizer.json'
$env:RAYON_NUM_THREADS='16'
$env:RBITNET_CUDA_QUANT_SMOKE='1'
cargo test -p bitnet-core --release opt_in_compensated_gemm --lib -- --nocapture *> target/tensorcore-oracle.log
if ($LASTEXITCODE -ne 0) {throw 'F64 GEMM oracle failed'}
$env:RBITNET_CUDA_VERIFY_TEST='1'
cargo test -p bitnet-core --release optional_real_verification --lib -- --nocapture *> target/tensorcore-verification.log
if ($LASTEXITCODE -ne 0) {throw 'Verification failed'}
$env:RBITNET_CUDA_VERIFY_TEST='0'
cargo test -p bitnet-core --release --test optional_llama_sequence -- --nocapture --test-threads=1 *> target/tensorcore-sequence.log
if ($LASTEXITCODE -ne 0) {throw 'Sequence failed'}
$env:RBITNET_CUDA_GEMM_BENCH_JSON='E:\devs\Rbitnet\target\tensorcore\real-gemm.json'
cargo test -p bitnet-core --release optional_real_gguf_gemm --lib -- --nocapture *> target/tensorcore-real-gemm.log
if ($LASTEXITCODE -ne 0) {throw 'GGUF GEMM microbench failed'}
