$ErrorActionPreference='Stop'
& C:/Users/azero/anaconda3/python.exe scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-split-kv/comparison-manifest.json --model qwen35-2b --backend gpu --binary target/release/rbitnet.exe --library target/qwen-block/cuda/rbitnet_cuda_quant64.dll --cycles 3 --qwen-full --qwen-prefill --tf32x3 --split-kv --output-dir target/qwen-block/http *> target/qwen-block-http.log
if ($LASTEXITCODE -ne 0) {throw 'Qwen block HTTP ablation failed'}
& C:/Users/azero/anaconda3/python.exe scripts/summarize_cache_benchmark.py target/qwen-block/http/results.json target/qwen-block/http/summary.json
if ($LASTEXITCODE -ne 0) {throw 'Summary failed'}
& C:/Users/azero/anaconda3/python.exe scripts/validate_cache_streaming.py --config docs/benchmarks/2026-10-03-split-kv/comparison-manifest.json --binary target/release/rbitnet.exe --library target/qwen-block/cuda/rbitnet_cuda_quant64.dll --qwen-full --qwen-prefill --tf32x3 --split-kv --output-dir target/qwen-block/streaming *> target/qwen-block-streaming.log
if ($LASTEXITCODE -ne 0) {throw 'Streaming failed'}
