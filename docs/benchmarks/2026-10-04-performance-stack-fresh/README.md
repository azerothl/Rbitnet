# Fresh combined performance-stack validation

The Native CUDA DLL, CLI, runner and proxy were rebuilt from combined commit
`f1257e5bd564543fa3b46a31acf351260d188f9c`. Workspace check, Clippy and tests passed
(335 passed, 1 ignored).

## Cross-feature validation

- Eight actual encoded-KV Native consumer refusals preserve state and counters.
- Eight actual Qwen import/nonce cases cover valid replacement, malformed refusal
  and stale speculative ownership, with split KV both off and on.
- The actual continuous Llama driver passes 24 configurations and two thread
  ownership fixtures. Portable Llama/Qwen state passes both split modes.
- Actual Qwen draft decoding retains its seeded sampling, penalties, rollback,
  cancellation and rejection fixtures.
- CPU direct-row arithmetic matches original bits in 240 synthetic and 1200
  actual-model row configurations: four GGUF models, five thread/SIMD settings.
- Actual ordinary four-model finish, CPU JSON/SSE/stop, RAM/SSD context and
  two-model proxy alias/session/restart suites pass on the combined executables.
- HTTP structured/tool refusal is tested with an active CUDA continuous worker
  and active Qwen draft verifier. Both paths are positively demonstrated by
  their counters before eight refusals; protected forward/ownership counters
  remain unchanged, and normal continuation outputs remain identical.

The first combined attempt correctly failed its positive worker assertion:
the HTTP fixture used generic `RBITNET_CONTINUOUS_BATCHING`, rather than the
Native worker's `RBITNET_CUDA_CONTINUOUS` option. That failed attempt remains
preserved locally with its exact helpers and artifacts. The corrected fixture
keeps the active-path and refusal-preservation assertions; this directory
contains a full fresh rerun, not a relabelled failed attempt.

## Scope and provenance

Defaults remain unchanged. Encoded KV is a capacity option; Qwen draft and
expert arena remain opt-in. Context tiers are refused with the continuous
worker, and Qwen draft bypasses ordinary context-tier capture. No unsupported
combination is presented as implemented merely because its refusal passes.

`receipt.json` indexes 124 original gzip captures and
14 exact helpers. `raw/manifest.json.gz` binds all
compiled sources, executed commands, three executable identities and DLL identity.
Binary models/executables and physical cache-state files are omitted. Source
working bytes are distinguished from Git line-ending normalization.
The orchestration checker requires local predecessor proof and model paths.

Individual-branch timing comparisons are separate evidence. This validation
does not establish the combined stack's throughput or Ollama/llama.cpp parity;
that four-model CPU/GPU comparison is queued separately. Namespace hardening,
Q8 speculative weight reuse and adaptive nucleus selection remain separate
candidates and are not included in this build. KIVI, true subword grammar,
trained expert prediction, dynamic compaction, MoE continuous serving and
non-NVIDIA hardware validation remain outside this delivery.
