# Decision: ship bounded Llama/Qwen RAM/SSD prefix tiers; do not broaden the transport

**Status:** accepted and closed for [#94](https://github.com/azerothl/Rbitnet/issues/94)
**Date:** 2026-10-08
**Evidence:** [Llama tier ablation](benchmarks/2026-10-08-session-tiers-94/README.md), [global quota and serving validation](benchmarks/2026-10-04-context-global-quota-fresh/README.md), [tier contract](CONTEXT_TIERS.md)

## Verdict

Ship `RBITNET_CONTEXT_TIERS` as the opt-in, bounded prefix-checkpoint product
for **Native F32 Llama and Qwen3.5 only**:

- active prefixes remain in the existing VRAM snapshot path;
- evicted compatible checkpoints are retained in bounded RAM and may be sealed
  to bounded SSD storage;
- a standalone idle-unloaded server reloads its configured model on the first
  later request and reopens the compatible store; the client must resend the
  full rendered history;
- incompatibility, corruption, admission and I/O failures fall back to ordinary
  prompt replay without importing untrusted tensor data.

This is a prefix cache keyed by the rendered token history and compatibility
identity. It is not a proxy session snapshot or a promise to preserve opaque
session headers. No SSD operation occurs in the active output-token loop.

## Measured validation

The same-revision Llama 3.2 1B CUDA HTTP ablation measured a 676-token prompt
with three samples per path:

| path | median server TTFT |
| --- | ---: |
| cold recompute | 2356 ms |
| RAM restore | 13 ms |
| fresh-process SSD restore | 51 ms |

All restored greedy responses matched the cold output. The report preserves
the model/tokenizer hashes, request and metric captures, transfer timing,
sealed size, RSS, managed-CUDA accounting, warm-up and dispersion. Its TTFT is
the server metric after health/readiness; it excludes startup/model loading and
is not a client-observed streaming-TTFT claim.

The implementation and prior live validation cover both claimed transports:
Qwen checkpoints include its GDN and convolution state and are restored only
at an exact checkpoint boundary. Qwen did not receive a fresh TTFT rerun on
this host because the available GGUF has no colocated tokenizer bundle; no
Llama result is presented as a Qwen latency number.

## Explicit boundary and limitations

The following are intentionally **not** delivered by this decision and are
non-blocking follow-ups, not implied support:

1. GPT-OSS, GLM/MLA and encoded-KV have no context-tier export/import transport.
   They remain rejected rather than silently using an incompatible state.
2. The proxy has no session-to-checkpoint ownership contract or verified
   idle-runner restore. Standalone idle recovery is the shipped integration;
   proxy verification belongs to a dedicated proxy/session feature.
3. The Llama ablation is non-streaming HTTP. It does not establish HTTP/SSE
   cancellation or stop-string equivalence across every tier and architecture.
4. Linux tmpfs exercises physical ENOSPC; a Windows physical-full and timed
   write/rename-crash sweep remains unmeasured.
5. FlexGen-style weight streaming is a separate batch-throughput question, not
   evidence for interactive prefix-state caching.

These constraints match the rejected-scope close style used by the speculative
and hybrid-MoE decisions: unsupported mechanisms are documented as boundaries,
not left to ambiguous fallback behavior.

## Reopen threshold

Reopen #94 only to change the accepted Llama/Qwen prefix-tier product itself:
a compatibility violation, data-corruption path, loss of replay fallback, or a
regression that invalidates the measured Llama restore claim. New GPT/GLM
transports, proxy-session persistence, Qwen latency reruns, and expanded
failure-mode coverage require their own scoped acceptance and evidence.
