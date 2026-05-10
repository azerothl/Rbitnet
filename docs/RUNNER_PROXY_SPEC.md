# Rbitnet multi-process runner proxy (Ollama-style)

This document specifies and tracks the implemented architecture: **one OS process per loaded model** (or per CUDA context family), with a thin **parent HTTP proxy** that routes `/v1/chat/completions` to the correct child. It mirrors the isolation model used by [Ollama’s `llm/server.go` runner subprocess](https://github.com/ollama/ollama/blob/main/llm/server.go) without vendoring their Go/C++ stack.

## Goals

- **VRAM accounting**: child exit releases GPU memory deterministically (no reliance on allocator behaviour in one big process).
- **Fault isolation**: a CUDA panic or abort in one model does not take down other models’ runners.
- **API**: keep a single OpenAI-compatible base URL; `model` selects the child (or a Unix socket / named pipe per runner).

## Components

1. **Proxy** (`rbitnet-proxy`): Axum router, model registry, no GGUF mmap. Responsibilities today: auth, request body cap, pick child, forward OpenAI-compatible requests, expose catalog, perform health checks.
2. **Runner** (`rbitnet-runner`, using the same server internals as `rbitnet-server`): one `RBITNET_MODEL`, one `Engine`, listens on a parent-selected `127.0.0.1:<port>`.
3. **Supervisor**: spawn runner on first use for `model_id`, restart after crash/health failure with exponential backoff, and kill children on graceful proxy shutdown.

## Protocol (minimal)

- Parent starts child with env: `RBITNET_MODEL`, `RBITNET_BIND=127.0.0.1:<port>`, `RBITNET_ACTIVE_MODEL_ID`, and `RBITNET_REQUIRE_MODEL_MATCH=1`.
- Parent stores `model_id -> { child_url, last_used }`.
- Health: `GET /ready` on child before routing; on failure, recycle child.

## Non-goals (for this spec)

- Sharing weights across children (each runner mmap’s its own file).
- Sub-second model hot-swap inside one process (use two children + drain instead).

## Relation to current code

Today’s **in-process** features (`RBITNET_MODEL_REGISTRY`, memory budget envs, idle unload to stub) are the lightweight subset. The runner proxy is the next step when you need **concurrent different large models** or **harder isolation** than `Arc<RwLock<Engine>>` provides.

## Phase 2 status

Done in-tree today:

- Single-process OpenAI-compatible serving with `/v1/models` and `/v1/chat/completions`.
- Compatibility bridge for `/v1/completions` through the chat path.
- Multi-model registry selection through `RBITNET_MODEL_REGISTRY`.
- Idle unload back to a stub engine through `RBITNET_IDLE_UNLOAD_SECS`.
- A local static UI at `/ui` for smoke testing the current process.
- A real `rbitnet-runner` worker binary that serves the existing Axum server for exactly one parent-selected model.
- A real `rbitnet-proxy` parent binary that reads `RBITNET_MODEL_REGISTRY`, supervises workers, routes `/v1/chat/completions` and `/v1/completions` by the JSON `model` field, and lists models from the registry.
- Native-first enforcement: default builds supervise only native workspace workers. External HTTP inference delegation is excluded unless compiled with the dev-only `experimental-external-backends` feature.

Still not implemented:

- Idle TTL/drain for child workers.
- Metrics aggregation from children.
- Request queueing or rate limiting beyond the existing body/auth checks.
- Full vLLM-class PagedAttention and continuous batching in native Rust kernels.

## Running the proxy

Create a registry:

```json
{
  "default": "tiny",
  "models": {
    "tiny": {
      "gguf": "C:/models/tiny.gguf",
      "tokenizer": "C:/models/tokenizer.json",
      "architecture": "llama"
    },
    "other": {
      "gguf": "C:/models/other.gguf"
    }
  }
}
```

PowerShell:

```powershell
$env:RBITNET_MODEL_REGISTRY="C:\path\to\rbitnet-registry.json"
$env:RBITNET_PROXY_BIND="127.0.0.1:8080"
cargo run -p rbitnet-proxy --release
```

Optional knobs:

- `RBITNET_RUNNER_BIN`: child executable path; defaults to `rbitnet-runner` beside `rbitnet-proxy`, then `PATH`.
- `RBITNET_RUNNER_READY_TIMEOUT_SECS`: child `/ready` deadline, default `60`.
- `RBITNET_PROXY_REQUEST_TIMEOUT_SECS`: upstream request timeout, default `600`.
- `RBITNET_API_KEY`: enforced by the proxy and forwarded to workers.

## External backend policy

`rbitnet-proxy` is native-first: the supported runtime path spawns `rbitnet-runner` children and forwards only to those local children. Any external HTTP backend must remain dev-only, behind `experimental-external-backends`, and must not be required for normal operation. See [NATIVE_FIRST.md](NATIVE_FIRST.md).
