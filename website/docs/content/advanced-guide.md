# Advanced guide

This page is for people who run Rbitnet as a local server: GPU placement, several requests at once, prompt reuse, metrics, and deployment. Everyday install and chat are in the [user guide](user-guide.md).

Rbitnet is a Rust process. It memory-maps the GGUF, runs the model, and speaks HTTP. Python is only needed if you convert or train weights somewhere else, before you have a GGUF.

## Programs

| Binary | Role |
|--------|------|
| `rbitnet` | CLI: install models, write config, `serve`, chat TUI |
| `rbitnet-server` | HTTP server. `rbitnet serve` starts this |
| `rbitnet-runner` | One model process, used behind the proxy |
| `rbitnet-proxy` | One listener in front of several model runners |

Default listen address: `127.0.0.1:8080`. Chat API: `POST /v1/chat/completions`. Model list: `GET /v1/models`. Web chat: `/ui`.

## CPU and NVIDIA CUDA

`RBITNET_BACKEND` defaults to `auto`: CUDA, then ROCm, then CPU. Vulkan and Metal builds can probe the machine but do not run GGUF inference.

For an NVIDIA GPU, build the quant library and point the server at it:

```powershell
.\scripts\build_cuda_quant.ps1
$env:RBITNET_BACKEND = "cuda"
$env:RBITNET_CUDA_QUANT_LIB = "E:\path\to\rbitnet_cuda_quant64.dll"
$env:RBITNET_MODEL = "E:\path\to\model.gguf"
rbitnet serve
```

On Linux, `scripts/build_cuda_quant.sh` produces `librbitnet_cuda_quant.so`. Set `RBITNET_CUDA_QUANT_LIB` to that file.

`RBITNET_CUDA_ATTENTION=0` forces attention back onto the CPU, which is useful when you want a before/after comparison. Pin `RBITNET_BACKEND=cpu` when you need a reproducible CPU run.

A 16 GiB GPU cannot hold every large MoE model fully on device. `RBITNET_HYBRID_MAX_VRAM_MB` is the soft weight budget (CUDA Llama defaults toward 4096 MiB; native Qwen, GPT-OSS, and GLM default toward 12288 MiB). Weights that do not fit stay on the CPU.

## One server, several chats

Two different mechanisms exist. Pick the one that matches the model.

**Llama on CUDA, shared GPU work.** Set:

```bash
export RBITNET_BACKEND=cuda
export RBITNET_CONTINUOUS_BATCHING=1
export RBITNET_FUSED_MULTI_SEQ=1
export RBITNET_CUDA_KV_FORMAT=f32
export RBITNET_CUDA_KV_PAGE_LIMIT=128
```

Compatible `/v1/chat/completions` requests, including `stream: true`, are grouped for a short window. Prompt chunks and decode rows share GPU projections. Each request keeps its own KV cache, sampling, and stop behavior.

On a 16 GiB GPU, a dense KV cache sized to the model maximum can refuse a wave of four requests. Set `RBITNET_CUDA_KV_PAGE_LIMIT` (128 is a known working cap for Llama 3.2 1B). Leave `RBITNET_CUDA_CONTINUOUS` unset unless you explicitly want the older live worker below.

The default streaming bridge sends the text after that request's shared result is ready. It does not emit one network event per token.

**Live token streaming (opt-in, Llama CUDA only).** Add:

```bash
export RBITNET_CUDA_CONTINUOUS=1
export RBITNET_CUDA_LIVE_SSE_MUX=1
```

Each request then receives its own token deltas from a shared decode wave. A client disconnect or an HTTP stop string retires that request between waves. Other requests keep running.

`RBITNET_CUDA_CONTINUOUS_ADMISSION=adaptive` reserves decode rows and the current prefill chunk before admitting another request. The default is `fifo`. Adaptive admission is Llama-only.

CPU Llama, BitNet, Qwen, GPT-OSS, and GLM do not share one fused forward today. For those models, continuous batching still schedules work, but each sequence runs its own forward.

## Reusing a long prompt

`RBITNET_CONTEXT_TIERS=1` keeps compatible Llama and Qwen3.5 prefixes close to the GPU, then in RAM, then on disk.

```bash
export RBITNET_CONTEXT_TIERS=1
export RBITNET_CONTEXT_DIR=/var/lib/rbitnet/context
export RBITNET_CONTEXT_RAM_MB=512
export RBITNET_CONTEXT_DISK_MB=4096
```

Set `RBITNET_CONTEXT_DISK_MB=0` for RAM only. The client still sends the full conversation each time. Rbitnet skips recomputing a prefix it has already stored. A corrupted or incompatible checkpoint falls back to a normal prefill. GPT-OSS and GLM do not use this store.

On one Llama 3.2 1B CUDA measurement (676-token prompt, greedy), median server time-to-first-token was 2356 ms cold, 13 ms from RAM, and 51 ms from a fresh process reading SSD. Restored text matched the cold run. Those numbers are for that machine and that prompt, not a general speed claim.

`RBITNET_PREFIX_KV=1` is the older in-process prefix snapshot. `rbitnet tune interactive` turns it on and keeps continuous batching off, which is a reasonable laptop chat profile.

## Named profiles

```bash
rbitnet tune interactive
rbitnet tune throughput --export
rbitnet tune battery --export
```

| Profile | Effect |
|---------|--------|
| `interactive` | Prefix reuse on, low concurrency |
| `throughput` | Continuous batching, shared KV pool, Q8 KV |
| `battery` | One request at a time, caches off |
| `bitnet-cpu` | BitNet architecture, CPU backend |

## HTTP limits and auth

| Variable | Default | Meaning |
|----------|---------|---------|
| `RBITNET_BIND` | `127.0.0.1:8080` | Listen address. `--bind` wins when the variable is unset |
| `RBITNET_API_KEY` | empty | If set, `/v1/*` requires `Authorization: Bearer` or `X-API-Key` |
| `RBITNET_MAX_CONCURRENT` | `4` | Extra requests get HTTP 503 |
| `RBITNET_MAX_TOKENS_CAP` | `8192` | Client `max_tokens` above this is HTTP 400 |
| `RBITNET_MAX_PROMPT_TOKENS` | unset | Optional cap on encoded prompt length |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | `600` | HTTP 504 when a completion exceeds this |
| `RBITNET_IDLE_UNLOAD_SECS` | unset | After this much idle time, drop the loaded model. The next request loads it again when `RBITNET_MODEL` is set |
| `RBITNET_ADMIN_TOKEN` | unset | Unlocks `POST /v1/admin/unload` and `POST /v1/admin/reload` |

`RBITNET_CORS_ANY=1` allows any browser origin. Leave it off on a shared machine.

## Several models

Point `RBITNET_MODEL_REGISTRY` at a JSON file of model ids, and run `rbitnet-proxy`. Each id gets one `rbitnet-runner` child. `RBITNET_PROXY_STICKY=1` pins a session header, cookie, or body field to a model id.

External engines such as vLLM are not part of a normal build.

## Metrics

`GET /metrics` is Prometheus text. No API key is required. Useful series:

- `rbitnet_inference_ttft_ms_sum` — encode plus prefill
- `rbitnet_inference_decode_ms_sum`
- `rbitnet_completion_tokens_total`
- `rbitnet_core_gpu_llama_batch_rows_total` and `rbitnet_core_gpu_llama_batch_waves_total` — fused Llama waves. Rows greater than waves means several sequences shared a GPU step

`GET /health` is process liveness. `GET /ready` is 503 until a model is loaded.

## Put it on a trusted network

Bind only to localhost and put a reverse proxy in front if other machines must connect.

Example systemd unit:

```ini
[Service]
Environment=RBITNET_BIND=127.0.0.1:8080
Environment=RBITNET_API_KEY=replace-me
Environment=RBITNET_MODEL=/var/lib/rbitnet/model.gguf
ExecStart=/usr/local/bin/rbitnet serve
Restart=on-failure
```

Terminate TLS on the proxy. Do not expose an unlocked `/v1` on a public interface. Image and tool requests are refused with HTTP 501 before any token is generated.

## Limits you should expect

- Structured JSON and tool calls are refused (HTTP 501). There is no grammar-constrained decoder in this release.
- Vision files can be inspected, but image chat is refused.
- Qwen, GPT-OSS, and GLM do not share the Llama fused multi-request forward.
- ROCm, Vulkan, and Metal are not supported GGUF inference paths.
- Speculative decoding exists as an opt-in experiment. It is not faster by default.
- CPU throughput on a 7B model will not match a tuned llama.cpp or vLLM GPU server.

## Akasha

Akasha can call this server as an OpenAI-compatible backend. That contract (routes, metrics names, and the BitNet recipe id) is described on the [Akasha Infer](akasha-infer.md) page. Nothing on the Rbitnet home page depends on Akasha: the server is usable on its own with any OpenAI client pointed at `http://127.0.0.1:8080/v1`.
