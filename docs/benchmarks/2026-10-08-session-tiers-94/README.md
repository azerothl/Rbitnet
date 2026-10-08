# Decision record: #94 cannot close yet

Date: 2026-10-08
Revision inspected: `origin/main` at `6373d6c` plus the current worktree.

This record corrects an earlier unverified progress draft. In particular, it
does **not** claim GPT-OSS or GLM/MLA portable context transport. Those ABI
symbols and their runtime wiring are absent from the inspected source.

## What is implemented and bounded

`RBITNET_CONTEXT_TIERS` is an opt-in **Llama Native F32 and Qwen3.5 Native
F32** prefix-checkpoint cache. It has:

- an independent VRAM prefix-snapshot path;
- a RAM host-checkpoint budget and LRU demotion;
- an optional SSD sealed-object budget, LRU eviction, retention TTL and entry
  cap;
- versioned, content-addressed envelopes that bind model, tokenizer, loaded
  CUDA library, device/runtime configuration and layout;
- bounded headers and payload geometry; checksums, finite-value checks,
  compatibility checks and replay on failure;
- temporary-file synchronization and atomic publication; recovery of generated
  interrupted writes; and a cooperative cross-namespace SSD quota.

The cache takes no SSD I/O in active token decode. A disk restore is read in
full before prefill resumes and is promoted to the charged RAM tier. These are
prefix checkpoints, not general conversational session snapshots: the key is
the token prefix plus compatibility identity, rather than a proxy session ID.

## Evidence inspected

- `crates/bitnet-core/src/context_tiers.rs` implements the RAM/disk budgets,
  retention, LRU, integrity failure fallback and Linux tmpfs ENOSPC fixture.
- `crates/bitnet-core/src/context_native.rs` only accepts the `llama` and
  `qwen` transport families and exports the cache/transfer counters.
- `native/cuda_quant/src/portable_state.cuh` only exports Llama and Qwen
  `portable_{bytes,export,import}` functions.
- `crates/bitnet-core/src/native/graph.rs` explicitly rejects context tiers in
  `NativeExecutor::load`; that is the GPT-OSS/GLM/MLA executor.
- `crates/bitnet-server/src/run.rs` swaps an idle server to a stub. On the
  next standalone request, `bitnet-server` now reloads its configured GGUF,
  reopens the compatible tier store, and records reload duration separately.
  A matching full rendered prompt can therefore restore its Llama/Qwen prefix;
  this has not yet been validated on a real model in this revision.
- `crates/rbitnet-proxy/src/lib.rs` still kills and respawns an idle runner
  without a proxy-level restore verification.

The existing two-model Llama/Qwen evidence is
[`2026-10-04-context-global-quota-fresh`](../2026-10-04-context-global-quota-fresh/README.md).
It proves bounded persistence and recovery behavior, not a measured resume
TTFT-versus-recompute result.

## Acceptance gaps that prevent closure

1. GPT-OSS and GLM/MLA have no RAM/SSD transport, no export/import ABI, and no
   restore/capture integration. Encoded-KV remains unsupported as documented.
2. There is no verified proxy/session-to-checkpoint ownership. Standalone idle
   unload now reloads and reopens compatible checkpoints, but real-model
   restore correctness and proxy idle-runner restoration remain unverified.
3. No reproducible same-revision ablation publishes cold/recompute, RAM and SSD
   resume TTFT; transfer/read/write time; disk size; RSS; VRAM; warm-up;
   repetitions and dispersion. The expected local benchmark directory
   `D:\Rbitnet-benchmark-models` was unavailable on this host.
4. HTTP/SSE cancellation and stop-string equivalence have not been run across
   the hot/warm/cold restore paths for each supported architecture.
5. The FlexGen-style weight-streaming decision is not separately recorded.
   It must remain distinct from session-state caching and be evaluated for
   throughput-oriented batches, not inferred to benefit interactive decode.
6. Physical ENOSPC is covered only by the Linux tmpfs fixture; a Windows
   physical-full and timed write/rename-crash sweep remains absent.

## Closure gate

Do not use `Closes #94` until all of the following are attached to a PR:

1. validated snapshot transports for the claimed model families, including
   position and recurrent/convolution state where applicable;
2. session-safe promotion/eviction and idle runner recovery that preserves
   isolation between proxy sessions;
3. hot (VRAM), warm (RAM), cold (SSD) and recompute measurements on the same
   GGUF, tokenizer, device and prompts, with raw artifacts; and
4. real HTTP/SSE correctness and failure-path evidence.

The correct current disposition is to keep #94 open. This document is a
decision and evidence boundary, not a completion claim.
