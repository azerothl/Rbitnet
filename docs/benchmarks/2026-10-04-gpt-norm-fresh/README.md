# Fresh GPT ordered-normalization proof

The selected normalization-only change passed fresh validation on 2026-10-04 at 15:21:00 UTC, using GPT-OSS-20B on the RTX 4080 SUPER. The original router remains unchanged. The CLI is identical across each baseline/new DLL comparison.

Validation includes cargo check/Clippy, 298 passing workspace tests (one ignored), a native build, 108 independent normalization cases with exact original-kernel bits and F64 checks, fixed-bank and segmented F64 graph fixtures, block positions/output modes/tails, and actual-model output comparisons. The 145 full-vector/output captures per DLL have identical SHA-256 maps. This complete-vector archive covers fixed-bank configurations; segmented behavior is covered by independent F64 fixtures and matching actual HTTP/SSE responses, not a segmented corpus-logit archive.

Four quiet captures each contain 27 JSON responses, nine SSE responses and three stop probes. Responses and usages match across DLLs. Network streaming and four-model finish-reason checks also passed.

## Measured decode

Context 2048, 24 system notes, maximum 128 output tokens, one warmup plus two measured cycles. Rates are medians derived from completion-token and decode-duration counter deltas. The two sustained prompts are library and museum stories, respectively; the capital-of-France short response is excluded from this throughput table.

| Placement / mode | Library: original → staged, tokens/s | Gain | Museum: original → staged, tokens/s | Gain |
| --- | ---: | ---: | ---: | ---: |
| Fixed / serial | 92.22 → 118.03 | 28.0% | 91.60 → 118.08 | 28.9% |
| Fixed / block16 | 91.66 → 117.54 | 28.2% | 91.82 → 117.38 | 27.8% |
| Fixed / block16-prefix | 92.22 → 118.35 | 28.3% | 92.25 → 118.41 | 28.4% |
| Segmented / full | 63.37 → 70.74 | 11.6% | 60.80 → 70.51 | 16.0% |
| Segmented / full-split | 71.36 → 89.58 | 25.5% | 71.73 → 89.52 | 24.8% |
| Segmented / full-split-prefix | 79.18 → 95.06 | 20.1% | 78.65 → 93.81 | 19.3% |

Segmented runs use an 8192 MiB expert cache, with asynchronous loading and prefetch disabled. Modes run in recorded order; two measured repeats provide no confidence interval. These figures establish neither general model speedup nor parity with other engines.

## Source and capture identities

`receipt.json` binds every compressed original log/JSON capture and copied harness. `raw/manifest.json.gz` binds the compiled working-file source bytes, CLI/DLL hashes, command results and all full-vector SHA-256 values. The local float archives total about 232 MB and are not committed, along with models and binaries. Both archives were rehashed and compared to the manifest before packaging. Diagnostic decoding sidecars preserve original console-byte hashes.

Changed source paths were checked against actual indexed Git blobs with only CRLF/LF normalization; raw working-file hashes may differ from Git checkout bytes on another platform. Capture/helper paths use `-text` and their actual Git blobs were verified byte for byte. Saved harnesses have local orchestration and model paths requiring adaptation; hardware runs must remain serial. Preparation metadata retains its original pending wording as historical data; the sealed journal and final successful commands establish completion.
