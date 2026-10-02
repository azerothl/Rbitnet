# Anthropic Messages API (Rbitnet subset)

Inventory for [#43](https://github.com/azerothl/Rbitnet/issues/43). **OpenAI-compatible routes remain the primary Akasha contract** (`BitNetProvider` → `/v1/chat/completions`, `/v1/models`, `/v1/completions`). The Anthropic surface is an optional second door for clients that set `ANTHROPIC_BASE_URL` (for example Claude Code–style agents).

Implementation: [`crates/bitnet-server/src/anthropic.rs`](../crates/bitnet-server/src/anthropic.rs) → `POST /v1/messages`.

## What works today

| Capability | Status | Notes |
| ---------- | ------ | ----- |
| `POST /v1/messages` non-stream | **Shipped** | Text-only assistant reply; `id` / `type` / `role` / `content[{type,text}]` / `model` / `stop_reason` / `usage` |
| String message `content` | **Shipped** | Converted to `role: text` prompt lines |
| Array content blocks with `text` fields | **Partial** | Text parts are joined; non-text block types are ignored |
| `max_tokens`, `temperature`, `model` | **Shipped** | Same engine path as OpenAI completions (stub/toy/GGUF) |
| `stream: true` SSE | **Shipped** | Live deltas from `Engine::complete_streaming` (`message_*` / `content_block_*` events); timeout/cancel aligned with OpenAI SSE |
| Auth (`RBITNET_API_KEY`) | **Shipped** | Same check as OpenAI routes (`Authorization: Bearer` or `x-api-key`) |
| Stub / toy smoke | **Covered** | `RBITNET_STUB=1` HTTP test in `crates/bitnet-server/tests/anthropic_compat.rs` |

### Example (non-stream)

```bash
RBITNET_STUB=1 rbitnet serve
curl -s http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2026-01-01" \
  -d '{
    "model": "rbitnet-stub",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### Example (live stream)

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
| Auth (`RBITNET_API_KEY`) | Enforced | **Same check** (`Bearer` / `x-api-key`) |
| Streaming | Real engine token deltas + `[DONE]` | Real engine token deltas (Anthropic event names) |
| Tools / function calling | Not a focus | **Missing** (non-text blocks dropped) |
| Chat templates / stop / sampling knobs | Fuller OpenAI field set | Temperature + max_tokens only |
| Metrics / concurrency / timeouts | Wired on chat path | Shares engine + timeout/cancel; metrics on stream Done |

Use OpenAI for Akasha production paths. Point `ANTHROPIC_BASE_URL` at Rbitnet only when an Anthropic-shaped client is required and the gaps below are acceptable.

## Gaps (explicit, out of deferred #43 close)

1. **Tools / tool_use / tool_result** — multi-turn agent blocks Claude Code expects; today non-text content is dropped.
2. **Multi-block fidelity** — image, thinking, and mixed blocks; system prompt field; `stop_sequences`; `top_p` / `metadata`.
3. **Claude Code polish** — richer `anthropic-version` / error JSON edge cases beyond the shared auth path.
4. **Docs / website** — optional deeper link from [INTEGRATIONS.md](INTEGRATIONS.md).

## Acceptance vs #43

| Criterion | Status |
| --------- | ------ |
| Inventory of current Anthropic parity | **Done** (this doc) |
| OpenAI documented as primary Akasha contract | **Done** |
| Chat stream Anthropic documented + test | **Done** (live engine SSE + stub smoke) |
| Écarts listés explicitement | **Done** (gaps table) |
| Tools useful for Claude Code | **Deferred** — tracked as gap, not blocking close of deferred #43 |

Follow-ups should keep OpenAI as the default contract and grow Anthropic tools only where agent DX needs it.
