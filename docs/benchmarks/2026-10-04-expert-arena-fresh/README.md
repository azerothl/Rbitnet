# Fresh expert-arena public-checkout validation

The CLI and Native CUDA DLL were rebuilt from the published finish-reason
checkout containing the seven unchanged arena implementation files. Workspace
check, Clippy and tests passed: 299 passed, one ignored.

Actual CUDA validation covers an independent F64 routed expert fixture;
one physical allocation, four groups and twelve leased views; mixed-format
cache refill, poison and final-lease ownership; and complete GPT-OSS/GLM
runtime checks at 512 and 8192 MiB expert budgets. Full-model logits,
generations, prefix reset and model lifetime are compared against the existing
reference path, with their numerical tolerances retained.

Actual arena-enabled HTTP validation passed for both MoE models, including
nonempty EOS, token-budget boundaries, zero-token requests, JSON/SSE equality
and explicit stop strings. The ordinary four-model finish suite also passed
with arena disabled.

The earlier [same-revision capacity and timing comparison](../2026-10-04-expert-arena/README.md)
is retained separately. This fresh run validates numerical behavior, ownership
and serving; it does not repeat that timing comparison. Earlier global-device
GPU peaks are not process VRAM. The option remains disabled by default, and
there is no new throughput, hardware-backend or independent-engine parity claim.
Fixed maximum-format slots do not implement dynamic compaction or a trained
expert predictor. Issues #83 and #84 remain open.

`receipt.json` indexes 43 gzip captures, the exact checker/HTTP helpers,
publication head and workspace test counts. `raw/manifest.json.gz` binds
compiled source bytes, build/test commands and executable/DLL SHA-256 identities.
All original capture bytes are retained after decompression. Models and binaries
are omitted. Historical preparation entries describe the earlier untested phase;
the completed commands and logs describe this fresh successful validation.
The orchestration checker depends on local predecessor proof directories and
model paths that require adaptation for reproduction.
