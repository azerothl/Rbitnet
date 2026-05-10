# Rbitnet multi-process runner proxy (Ollama-style)

This document specifies an optional future architecture: **one OS process per loaded model** (or per CUDA context family), with a thin **parent HTTP proxy** that routes `/v1/chat/completions` to the correct child. It mirrors the isolation model used by [Ollama’s `llm/server.go` runner subprocess](https://github.com/ollama/ollama/blob/main/llm/server.go) without vendoring their Go/C++ stack.

## Goals

- **VRAM accounting**: child exit releases GPU memory deterministically (no reliance on allocator behaviour in one big process).
- **Fault isolation**: a CUDA panic or abort in one model does not take down other models’ runners.
- **API**: keep a single OpenAI-compatible base URL; `model` selects the child (or a Unix socket / named pipe per runner).

## Components

1. **Proxy** (new binary or `rbitnet serve --proxy`): Axum router, model registry, no GGUF mmap. Responsibilities: auth, rate limits, queue, pick child, forward request, merge metrics.
2. **Runner** (existing `rbitnet-server` or slimmed binary): one `RBITNET_MODEL`, one `Engine`, listens on `127.0.0.1:0` or inherits a Unix socket from parent.
3. **Supervisor**: spawn runner on first use for `model_id`, SIGTERM on idle TTL, restart on crash with backoff.

## Protocol (minimal)

- Parent starts child with env: `RBITNET_MODEL`, `RBITNET_BIND=127.0.0.1:<port>`, optional `RBITNET_ACTIVE_MODEL_ID` for validation.
- Parent stores `model_id -> { child_url, last_used }`.
- Health: `GET /ready` on child before routing; on failure, recycle child.

## Non-goals (for this spec)

- Sharing weights across children (each runner mmap’s its own file).
- Sub-second model hot-swap inside one process (use two children + drain instead).

## Relation to current code

Today’s **in-process** features (`RBITNET_MODEL_REGISTRY`, memory budget envs, idle unload to stub) are the lightweight subset. The runner proxy is the next step when you need **concurrent different large models** or **harder isolation** than `Arc<RwLock<Engine>>` provides.

## Phase 1 status

Done in-tree today:

- Single-process OpenAI-compatible serving with `/v1/models` and `/v1/chat/completions`.
- Compatibility bridge for `/v1/completions` through the chat path.
- Multi-model registry selection through `RBITNET_MODEL_REGISTRY`.
- Idle unload back to a stub engine through `RBITNET_IDLE_UNLOAD_SECS`.
- A local static UI at `/ui` for smoke testing the current process.
- A stub `rbitnet-runner` binary that exits with code `64` after printing the planned child-process contract.

Reserved proxy phase:

- Reserve `RBITNET_RUNNER_PROXY=1` for the future parent proxy mode.
- Reserve `rbitnet serve --proxy` for the future CLI entrypoint.

These proxy flags are not implemented yet. The runner stub is intentionally not a server; it documents the subprocess env contract and keeps release archives ready for the next step.

## Implemented vs Planned

Implemented:

- `crates/rbitnet-runner`: contract-only binary.
- Release archives include `rbitnet-runner` beside `rbitnet` and `rbitnet-server`.
- Parent process remains the existing single-process Axum server.

Planned:

- Parent proxy mode requiring `RBITNET_MODEL_REGISTRY`.
- Spawn one runner per model id with `RBITNET_MODEL`, `RBITNET_BIND=127.0.0.1:0`, tokenizer env/config and optional architecture override.
- Probe child `GET /ready`, then forward `/v1/chat/completions` and `/v1/completions`.
- Idle TTL, crash backoff, log capture, metrics merge and graceful drain.
- Tests for spawn, health probe, request forwarding, auth propagation and idle termination.
