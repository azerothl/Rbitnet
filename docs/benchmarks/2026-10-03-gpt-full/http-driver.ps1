$ErrorActionPreference='Stop'
& C:/Users/azero/anaconda3/python.exe scripts/benchmark_cache_stack.py --config target/gpt-full/ablation-manifest.json --model gpt-oss-20b --backend gpu --binary target/release/rbitnet.exe --library target/gpt-full/cuda/rbitnet_cuda_quant64.dll --cycles 3 --gpt-full --output-dir target/gpt-full/http *> target/gpt-full-http.log
if ($LASTEXITCODE -ne 0) {throw 'GPT-OSS HTTP ablation failed'}
& C:/Users/azero/anaconda3/python.exe scripts/summarize_cache_benchmark.py target/gpt-full/http/results.json target/gpt-full/http/summary.json
if ($LASTEXITCODE -ne 0) {throw 'Summary failed'}
& C:/Users/azero/anaconda3/python.exe scripts/validate_cache_streaming.py --config target/gpt-full/ablation-manifest.json --binary target/release/rbitnet.exe --library target/gpt-full/cuda/rbitnet_cuda_quant64.dll --gpt-full --split-kv --output-dir target/gpt-full/streaming *> target/gpt-full-streaming.log
if ($LASTEXITCODE -ne 0) {throw 'Streaming failed'}
