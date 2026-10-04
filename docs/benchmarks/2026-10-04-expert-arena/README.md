# Expert arena: measured capacity, limited throughput effect

See [the implementation and scope](../../MOE_EXPERT_ARENA.md).

`summary.json` is computed from four completed `results.json` captures, each
with 27 JSON rows, nine SSE rows and three stop probes. `raw/` contains gzip
copies of the actual logs and JSON captures; decompressed SHA-256 values are
listed in `SHA256.json`. The model files, executable and DLL are not included.

`receipt.json` binds the isolated measurement manifest, the exact seven public
implementation files, Native source identity, original capture hashes and
reproducibility scripts. The preparation manifest's "prepared only" limitations
describe its historical creation phase; the completed logs are the evidence of
later execution. Fresh public-checkout validation is still pending separately.

The summary uses generated-token counts checked against response usage and
decode time from request metrics. Timing excludes the warm cycle. GPU memory is
sampled global device usage, not process VRAM; managed memory is the Native
requested-byte accounting. The difference between them does not prove a driver
granularity or memory compaction mechanism. Async demand-only is the direct
baseline for the incremental arena effect.

No model/expert quantization changed. No speed guarantee, dynamic compaction,
trained predictor or independent-engine parity is claimed.
