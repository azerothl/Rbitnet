# Prefix checkpoints in RAM and on disk

This opt-in implementation saves Native CUDA F32 prompt prefixes for Llama and
Qwen3.5. A matching checkpoint avoids recomputing that prefix. It can reduce
latency before the first output token on repeated prompts; it does not avoid
reading model weights during decoding or establish an increase in decode tokens/s.

Implementation and serving validation are in progress. Do not treat the presence
of this document as a completed benchmark or support for other architectures.

Compatibility also records prefill partition sizes, ordered Qwen block mode,
prefix checkpoint boundaries and Native page geometry. Changing these options
selects another namespace rather than importing state produced by a different
calculation configuration.

The engine-created `rbitnet-state-v1` directory and its identity namespace are
checked before creating anything below them. Links and Windows reparse points,
including directory junctions, are refused. Windows keeps directory handles
without delete sharing for the store lifetime, preventing those managed
components from being renamed while the cache uses them. Lock inspection also
covers dangling links before opening the file. The configured parent remains
the user's chosen directory.

The namespace-guard follow-up is prepared; Windows junction/rename regression
tests and fresh serving validation have not run yet. These changes do not add
a global quota across old namespaces or validate other operating systems.

## Configuration

Set `RBITNET_CONTEXT_TIERS=1` and `RBITNET_CONTEXT_DIR` to a local directory.
The default payload budget is 256 MiB in RAM and 2048 MiB on disk, with 64 entries
and a 1800-second retention period. Override these using
`RBITNET_CONTEXT_RAM_MB`, `RBITNET_CONTEXT_DISK_MB`, `RBITNET_CONTEXT_ENTRIES`
and `RBITNET_CONTEXT_TTL_SECS`. Setting the disk budget to zero selects RAM only.

Budgets apply to one model runtime and its compatibility namespace. They are not
a global quota across model versions, binary versions or several models. Old
compatibility namespaces remain on disk until removed by their owner. Payload
accounting includes borrowed snapshots until their last owner is dropped. It
does not measure process RSS, allocator overhead or the smaller indexing metadata.
Disk accounting includes the sealed file and the temporary write needed to
create it; arbitrary user files are outside the derived cache quota.

Llama requires Native CUDA F32 residency. Qwen requires the full Native CUDA
runtime, including recurrent state and convolution history. Both have a context
capacity limit of 8192 tokens for this transport. TF32 and speculative decoding are refused
with this option. CPU, encoded KV, GPT-OSS and GLM state transports are not provided.

## Restore and persistence

The runtime captures before output decoding. It first tries an existing VRAM
prefix when that independent feature is enabled, then the longest exact saved
prefix. The RAM tier uses owned leases. The disk tier reads a complete checkpoint
before the last prompt token is evaluated. There is no SSD read or write in the
output-token loop. Captures are persisted eagerly so a terminated runner can
release its weights without losing already sealed prompt prefixes.

A checkpoint binds the actual GGUF bytes, tokenizer bytes, loaded Native DLL,
executable, GPU UUID, compute capability, CUDA driver API version, runtime shape
and selected arithmetic settings. The driver API version is not the NVIDIA driver
build number. Binary or configuration changes can cause intentional cache misses.
Hashes are computed when initializing the store and add startup work.

With `RBITNET_IDLE_UNLOAD_SECS`, a standalone (non-registry) server reloads its
configured GGUF on the first later inference request. That reload reopens the
compatible context-tier store, so an identical conversation prefix can restore
from RAM or SSD before prefill. `rbitnet_idle_restores_total` and
`rbitnet_idle_restore_ms_sum` distinguish model reload cost from generation
TTFT. This is prefix reuse based on the full rendered prompt, not a server-held
conversation continuation: clients must resend their history, and no generated
state is shared merely because two requests use the same session header.

The format checks the version, exact byte size, bounded header, expected payload
geometry, compatibility fields, finite values, SHA-256 checksum and content
address before Native import. Qwen state is captured only at an exact prefill
boundary; its recurrent history cannot be truncated like an attention KV plane.

An OS lock admits one cooperative owner per namespace. Writes use a new temporary
file, flush and synchronize it, then rename it into a sealed content-addressed
object. Opening a namespace reclaims generated interrupted-write files and
malformed derived objects. Unrecognized user files are retained. A bounded scan
refuses an oversized namespace and falls back to normal prompt replay.

Store or transfer failures retain ordinary prompt replay. A failed persistence
attempt can retain a valid RAM checkpoint. Corruption cannot supply tensor data
to CUDA. No cache operation is permitted to make another active lease exceed the
host payload budget.

## Evidence required before adoption

### Cooperative global SSD admission (Windows validation passed)

`RBITNET_CONTEXT_DISK_GLOBAL_MB` opts into a cooperative global disk cap below
the configured cache root. RAM and ordinary per-namespace disk limits remain
unchanged. A shared OS lock protects admission through the temporary write and
rename. The scan counts sealed objects and recognized interrupted temporaries
across compatibility namespaces. Inactive namespaces can lose their oldest
derived files; active foreign namespaces remain protected. The current runtime
can evict its own cold files while external RAM leases remain valid.

Lock contention, an active-namespace capacity shortage or an I/O refusal retains
the RAM checkpoint and ordinary replay. The global cap is registered once in
`.disk-quota.lock`; inconsistent caps are refused. To change the registered cap,
stop writers and choose a fresh root. Legacy writers without this option do not
participate in global admission and must not share the root with enabled writers.
Existing active data above a newly introduced cap cannot be reclaimed while
its namespace is locked; new disk admission is refused rather than deleting it.
Only recognized managed objects count, not arbitrary user files in the root.

Eight new fixtures pass for cross-model accounting, active
namespace refusal, own LRU reclamation with live RAM leases, contention, differing
caps, I/O refusal, interrupted temporaries and redirected foreign namespaces.
The process fixture holds a sealed checkpoint in a second process, attempts
admission while its namespace is locked, terminates that process forcibly, then
attempts reclamation and checks physical bytes and a surviving RAM lease.
Fresh compilation, all 19 store tests and actual Llama/Qwen HTTP/proxy execution pass.
346 workspace tests pass with one ignored. See the [raw validation report](benchmarks/2026-10-04-context-global-quota-fresh/README.md).
This does not claim a physical ENOSPC or timed process-crash recovery sweep.

The validation must cover actual Llama and Qwen prompt outputs, seeded sampling,
RAM hits, RAM eviction to disk, a separate server-process restart, corruption
replay, stop strings and disconnected streams. It must also cover two model
runners through the proxy with session affinity and idle runner recycling.
Report startup hashing, transfers, sealed storage, RSS, VRAM and first-token
latency separately. A successful unit test is not evidence of faster serving.

Simulated interrupted files and an insufficient configured disk quota do not
demonstrate recovery from a physical disk-full condition. Crash timing, physical
disk-full recovery and combined cache performance remain unverified.
Cross-process ownership/restart and real two-model admission are now covered
by the published captures; this does not cover a timed write/rename crash sweep.
