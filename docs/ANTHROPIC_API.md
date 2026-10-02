# Anthropic Messages API (Rbitnet subset)

Spike inventory for [#43](https://github.com/azerothl/Rbitnet/issues/43). **OpenAI-compatible routes remain the primary Akasha contract** (`BitNetProvider` → `/v1/chat/completions`, `/v1/models`, `/v1/completions`). The Anthropic surface is an optional second door for clients that set `ANTHROPIC_BASE_URL` (for example Claude Code–style agents).

Implementation: [`crates/bitnet-server/src/anthropic.rs`](../crates/bitnet-server/src/anthropic.rs) → `POST /v1/messages`.

## What works today

| Capability | Status | Notes |
| ---------- | ------ | ----- |
| `POST /v1/messages` non-stream | **Shipped** | Text-only assistant reply; `id` / `type` / `role` / `content[{type,text}]` / `model` / `stop_reason` / `usage` |
| String message `content` | **Shipped** | Converted to `role: text` prompt lines |
| Array content blocks with `text` fields | **Partial** | Text parts are joined; non-text block types are ignored |
| `max_tokens`, `temperature`, `model` | **Shipped** | Same engine path as OpenAI completions (stub/toy/GGUF) |
| `stream: true` SSE | **Minimal stub** | Anthropic-shaped `message_*` / `content_block_*` events after a full completion; not token-by-token from the engine |
| Stub / toy smoke | **Covered** | `RBITNET_STUB=1` HTTP test in `crates/bitnet-server/tests/anthropic_compat.rs` |

### Example (non-stream)

```bash
RBITNET_STUB=1 rbitnet serve
curl -s http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "rbitnet-stub",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### Example (minimal stream)

```bash
curl -N http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rbitnet-stub",
    "max_tokens": 32,
    "stream": true,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Expect `text/event-stream` with `event:` lines (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`).

## OpenAI primary vs Anthropic optional

| Concern | OpenAI (`/v1/chat/completions`, …) | Anthropic (`/v1/messages`) |
| ------- | ---------------------------------- | -------------------------- |
| Akasha default | **Yes** — `BitNetProvider` | No — optional alternate base URL |
| Auth (`RBITNET_API_KEY`) | Enforced on chat/completions/models/admin | **Not wired** yet (gap) |
| Streaming | Real engine token deltas + `[DONE]` | Minimal post-complete SSE stub |
| Tools / function calling | Not a focus of this spike | **Missing** |
| Chat templates / stop / sampling knobs | Fuller OpenAI field set | Temperature + max_tokens only |
| Metrics / concurrency / timeouts | Wired on chat path | Shares engine; no Anthropic-specific metrics |

Use OpenAI for Akasha production paths. Point `ANTHROPIC_BASE_URL` at Rbitnet only when an Anthropic-shaped client is required and the gaps below are acceptable.

## Gaps (next slices)

1. **True streaming** — emit deltas from `Engine::stream` (or equivalent) instead of chunking a finished string; align cancel/timeout with OpenAI SSE.
2. **Tools / tool_use / tool_result** — multi-turn agent blocks Claude Code expects; today non-text content is dropped.
3. **Multi-block fidelity** — image, thinking, and mixed blocks; system prompt field; `stop_sequences`; `top_p` / `metadata`.
4. **Claude Code readiness** — `anthropic-version` header handling, error JSON shape (`type` / `error`), auth via `x-api-key`, and enough tool/stream parity to run a real session.
5. **Auth parity** — call the same `RBITNET_API_KEY` check used by OpenAI routes.
6. **Docs / website** — optional link from [INTEGRATIONS.md](INTEGRATIONS.md) once the subset is no longer spike-only.

## Acceptance vs #43

| Criterion | This spike |
| --------- | ---------- |
| Inventory of current Anthropic parity | **Done** (this doc) |
| OpenAI documented as primary Akasha contract | **Done** |
| Optional Anthropic smoke under stub | **Done** (non-stream + minimal stream tests) |
| Stream + tools useful for Claude Code | **Not done** — stream is a minimal stub; tools absent |

Follow-ups should keep OpenAI as the default contract and grow Anthropic only where agent DX needs it.
