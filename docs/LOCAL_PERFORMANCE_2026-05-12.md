# Local Performance Notes - 2026-05-12

This document records the real Rbitnet model smoke tests run on the current
developer machine. These numbers are local engineering measurements, not frozen
release baselines. For release-quality rows, use the procedure in
[BENCHMARKS.md](BENCHMARKS.md).

## Machine

| Field | Value |
|-------|-------|
| Host | Dell Inc. XPS 15 9520 |
| OS | Microsoft Windows 11 Business `10.0.26200`, 64-bit |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700H |
| CPU topology | 14 physical cores / 20 logical processors |
| Reported max clock | 2300 MHz via WMI |
| RAM | 34,014,814,208 bytes, about 31.7 GiB |
| GPU present | Intel Iris Xe Graphics; NVIDIA GeForce RTX 3050 Ti Laptop GPU, about 4 GiB VRAM |
| GPU usage in these tests | CPU for the first pass; CUDA comparison pass used `RBITNET_BACKEND=cuda` |
| Rust | `rustc 1.93.1 (01f6ddf75 2026-02-11)` |
| Cargo | `cargo 1.93.1 (083ac5135 2025-12-15)` |
| Git SHA at capture | `de7417f` plus local Qwen3 loader changes |

## Method

- Server: `target\release\rbitnet-server.exe`.
- Real GGUF inference only. Stub and toy modes are excluded.
- Backends: CPU (`RBITNET_BACKEND` unset / CPU path) and CUDA
  (`RBITNET_BACKEND=cuda`) for the comparison pass.
- Prompt: short smoke prompt, usually `Say hello.`
- Chat format: `raw` for the recorded smokes.
- Timing source: wall-clock HTTP request time plus `/metrics` phase counters when
  the request completed.
- Goal: determine which model families currently load and generate on this
  machine, not produce statistically stable p50/p95 latency.
- CUDA detection: `nvidia-smi` reported an NVIDIA GeForce RTX 3050 Ti Laptop GPU,
  4096 MiB VRAM, driver `591.74`, CUDA `13.1`. `/v1/models` reported
  `backend="cuda"` and `backend_accelerated=true` for the CUDA-loaded smokes.

## Results

| Model | File / quant | CPU result | CUDA result | Comparison / notes |
|-------|--------------|------------|-------------|--------------------|
| TinyLlama 1.1B Chat | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Works: 4 tokens in about 11.65 s; prefill about 6698 ms; decode about 4791 ms; TPOT about 1.20 s/token | Works: 4 tokens in about 28.71 s; prefill about 16929 ms; decode about 11633 ms; TPOT about 2.91 s/token | CUDA MVP is about 2.5x slower here. The raw prompt/template still produced poor text (`act...`) on both backends. |
| Llama 3.2 3B Instruct | `llama-3.2-3b-instruct-q4_k_m.gguf` | Works after tied-output fallback: 2 tokens in about 44.03 s; prefill about 35425 ms; decode about 7518 ms; TPOT about 3.76 s/token | Works: 2 tokens in about 59.00 s; prefill about 42324 ms; decode about 15333 ms; TPOT about 7.67 s/token | Useful middle-weight Llama-compatible test. CUDA is slower than CPU in the current MVP backend. Raw prompt output was `..`, so quality still needs the Llama 3 chat template. |
| Mistral 7B Instruct v0.2 | `mistral-7b-instruct-v0.2.Q4_K_M.gguf` | Works, but too slow: 2 tokens in about 86.96 s; prefill about 69176 ms; decode about 16611 ms; TPOT about 8.31 s/token | Works: 2 tokens in about 86.64 s; prefill about 66491 ms; decode about 19409 ms; TPOT about 9.70 s/token | CUDA and CPU are effectively tied for this smoke. Prefill improved slightly, decode regressed, and overall latency stayed non-interactive. |
| Mistral 7B Instruct v0.2 | `mistral-7b-instruct-v0.2.Q2_K.gguf` | Fails / not usable: HTTP 504 after 120 s for 2 requested tokens | Fails / not usable: HTTP 504 after 120.82 s for 2 requested tokens | CUDA does not rescue this quant path. An earlier debug run also exposed a `dequant_q3_k` assertion path, so this quant is not currently a good target. |
| Qwen3 4B | `Qwen3-4B-Q4_K_M.gguf` | Works after the local dense Qwen3 loader integration: 1 token in about 37.53 s; prefill about 29140 ms; decode about 7018 ms | Loads as `backend="cuda"` / `backend_accelerated=true`, but generation fails immediately: `dense qwen3 MVP currently supports CPU backend only` | The current Qwen3 loader is a CPU correctness MVP. CUDA support still needs to be wired into the dense Qwen3 runtime. |
| Qwen3.6 35B A3B MoE | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | Startup failure: `qwen35moe` requires `RBITNET_BACKEND=cuda` | Works for a 1-token smoke: 1 token in about 62.40 s; prefill about 48533 ms; decode about 9121 ms; TPOT about 9.12 s/token | CUDA is mandatory and functional for this native MoE path, but the 35B A3B checkpoint is still far too slow for interactive local use on a 4 GiB laptop GPU. |

## Interpretation

TinyLlama 1.1B Q4 is the only real model in this set that is reasonable for
quick CPU smoke tests on the current machine. CUDA is not faster for this small
model in the current MVP backend because transfer/setup overhead dominates the
tiny workload. Output quality still needs a proper chat template; the raw prompt
path proves the runtime works but does not prove assistant quality.

Llama 3.2 3B fills the missing middle ground between TinyLlama and Mistral 7B.
It now runs after adding support for GGUFs that tie `output.weight` to
`token_embd.weight`, but it is still a correctness/performance smoke rather than
an interactive target. Mistral 7B Q4 runs on both CPU and CUDA, but the current
CUDA path does not materially improve end-to-end latency. Qwen3 4B runs on CPU
only today; the new loader is a correctness MVP, not an optimized CUDA inference
path.

Mistral Q2_K is not a safe optimization shortcut today. Although the file is
smaller than Q4_K_M, the tested path either timed out or hit quantization
handling issues, so it should not be promoted as a recommended CPU model until
the dequant kernels are fixed and benchmarked.

The local RTX 3050 Ti can run the native Qwen35MoE CUDA path, but a 1-token smoke
still takes about a minute. The model is useful to prove that the CUDA path loads
and generates, not as a practical local assistant target on this 4 GiB laptop GPU.

## Current CPU / CUDA Recommendations

Use `--release` builds for every real inference test. Debug builds are not
representative and can make even 1-2 generated tokens look unusable.

For local development on this machine:

- Prefer TinyLlama-class models, about 1B parameters in `Q4_K_M`, for fast smoke
  tests. Use CPU for this path unless CUDA kernels are redesigned to avoid the
  current overhead.
- Treat 3B-4B models as correctness tests, not interactive models, until matvec,
  dequantization, prompt prefill, and CUDA transfer overhead are optimized.
- Avoid 7B CPU inference for normal development unless the goal is stress
  testing load, cancellation, timeout, or memory behaviour.
- Do not assume `RBITNET_BACKEND=cuda` improves Llama/Mistral throughput yet. In
  these smokes it was slower for TinyLlama and neutral for Mistral 7B.
- Use CUDA for Qwen35MoE because CPU is unsupported, but keep `max_tokens` very
  small on this 4 GiB GPU.
- Keep `max_tokens` very small while validating loaders: 1-4 tokens is enough to
  prove load, prefill, decode, and response serialization.
- Keep GGUF as the native runtime format. Safetensors/PyTorch checkpoints need a
  conversion path before Rbitnet can run them natively.

## Reproduction Snippets

TinyLlama Q4:

```powershell
$env:RBITNET_BIND = "127.0.0.1:18101"
$env:RBITNET_MODEL = "models\tinyllama-direct\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
$env:RBITNET_TOKENIZER = "models\tinyllama-direct\tokenizer.json"
$env:RBITNET_CHAT_FORMAT = "raw"
$env:RBITNET_INFERENCE_TIMEOUT_SECS = "240"
target\release\rbitnet-server.exe
```

TinyLlama Q4, CUDA comparison:

```powershell
$env:RBITNET_BIND = "127.0.0.1:18201"
$env:RBITNET_MODEL = "models\tinyllama-direct\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
$env:RBITNET_TOKENIZER = "models\tinyllama-direct\tokenizer.json"
$env:RBITNET_CHAT_FORMAT = "raw"
$env:RBITNET_BACKEND = "cuda"
$env:RBITNET_INFERENCE_TIMEOUT_SECS = "240"
target\release\rbitnet-server.exe
```

Llama 3.2 3B Q4, CPU:

```powershell
$env:RBITNET_BIND = "127.0.0.1:18321"
$env:RBITNET_MODEL = "models\llama32-3b\llama-3.2-3b-instruct-q4_k_m.gguf"
$env:RBITNET_TOKENIZER = "models\llama32-3b\tokenizer.json"
$env:RBITNET_CHAT_FORMAT = "raw"
$env:RBITNET_INFERENCE_TIMEOUT_SECS = "240"
target\release\rbitnet-server.exe
```

Llama 3.2 3B Q4, CUDA comparison:

```powershell
$env:RBITNET_BIND = "127.0.0.1:18322"
$env:RBITNET_MODEL = "models\llama32-3b\llama-3.2-3b-instruct-q4_k_m.gguf"
$env:RBITNET_TOKENIZER = "models\llama32-3b\tokenizer.json"
$env:RBITNET_CHAT_FORMAT = "raw"
$env:RBITNET_BACKEND = "cuda"
$env:RBITNET_INFERENCE_TIMEOUT_SECS = "240"
target\release\rbitnet-server.exe
```

Qwen3 4B Q4:

```powershell
$env:RBITNET_BIND = "127.0.0.1:18121"
$env:RBITNET_MODEL = "models\qwen3-4b\Qwen3-4B-Q4_K_M.gguf"
$env:RBITNET_TOKENIZER = "models\qwen3-4b\tokenizer.json"
$env:RBITNET_CHAT_FORMAT = "raw"
$env:RBITNET_INFERENCE_TIMEOUT_SECS = "360"
target\release\rbitnet-server.exe
```

Qwen35MoE CUDA 1-token smoke:

```powershell
$env:RBITNET_BIND = "127.0.0.1:18206"
$env:RBITNET_MODEL = "models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
$env:RBITNET_TOKENIZER = "models\Qwen3.6-35B-A3B-GGUF\tokenizer.json"
$env:RBITNET_CHAT_FORMAT = "raw"
$env:RBITNET_BACKEND = "cuda"
$env:RBITNET_INFERENCE_TIMEOUT_SECS = "120"
target\release\rbitnet-server.exe
```

Example request:

```powershell
$body = @{
  model = "rbitnet-qwen3"
  messages = @(@{ role = "user"; content = "Say hello." })
  max_tokens = 1
  temperature = 0.1
} | ConvertTo-Json -Depth 8 -Compress

Invoke-WebRequest `
  -UseBasicParsing `
  "http://127.0.0.1:18121/v1/chat/completions" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## Next Measurements

For a cleaner performance baseline, rerun each working model through
`scripts/bench_backend_compare.py` with a fixed prompt, at least 10-15 runs, and
record p50/p95 plus peak RSS. The next useful CPU optimization pass should
measure:

- `RBITNET_LLAMA_WEIGHT_MODE=dense` vs mmap/quant mode where applicable.
- Release build with `RUSTFLAGS="-C target-cpu=native"` on the same machine.
- CUDA runs with peak VRAM and GPU utilization captured from `nvidia-smi`.
- Correct chat templates for TinyLlama, Mistral, and Qwen3.
- Longer prompts split into encode, prefill, and decode timing from `/metrics`.
