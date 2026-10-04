# Fresh persistent prompt-state proof

This opt-in implementation transports Native CUDA F32 prompt state for Llama and dense Qwen3.5 to RAM and sealed disk files. Qwen includes KV, recurrent GDN state and convolution history. A compatible saved prefix avoids its recomputation before decoding; this change does not stream model weights or establish faster output tokens/s.

Fresh validation completed on 2026-10-04 at 15:34:20 UTC: cargo check/Clippy, 309 passing workspace tests (one ignored), native/CLI/proxy/runner builds, actual Llama/Qwen portable-state fixtures with split attention off/on, independent Qwen attention/recurrent F64 fixtures, standalone HTTP/SSE and two-model proxy serving. Default finish reasons were also checked on all four reference models.

The standalone captures exercise seeded responses, RAM hits, RAM eviction followed by disk restoration, a separate server-process restart, corruption replay, stop strings and disconnected streams. The proxy captures verify explicit selection, omitted-model session affinity, actual runner PID replacement after idle expiry, disk hits after runner replacement and after proxy-process restart. Content, finish reasons and completion counts match the uncached references. These are state-equivalence tests, not a semantic-quality evaluation.

## Proxy identity correction

The first proxy run returned HTTP 400: the proxy forwarded `llama32-1b`, while its scoped runner required the architecture ID `rbitnet-llama`. Strict standalone runners now honor the parent's `RBITNET_ACTIVE_MODEL_ID` alias at startup and reload, expose it in `/v1/models`, and still reject another model ID. Empty/unset aliases retain the engine ID. The successful proxy harness also checks each runner's catalog and response model ID; error bodies are preserved before any assertion.

The failed first attempt and exact HTTP diagnostic are retained locally under `target/performance-cache/context-proxy-attempt-20261004T152508Z/` and `context-proxy-diagnosis/`. They are separate from the fresh successful captures packaged here.

## Limits and follow-up

Use `RBITNET_CONTEXT_TIERS=1` and `RBITNET_CONTEXT_DIR`; see [configuration and format](../../CONTEXT_TIERS.md). Budgets apply per runtime/compatibility namespace, including snapshot leases and temporary writes. Old namespaces are not reclaimed globally. Compatibility binds model/tokenizer/native library/executable/device and arithmetic configuration. CPU, encoded KV, GPT-OSS and GLM transport are unsupported. SSD I/O occurs around prompt restoration/capture, outside the output-token loop.

Physical disk-full recovery, an adversarial crash-timing sweep, global quota reclamation, and a same-protocol comparison with reference engines remain unverified. Follow-up managed-directory/junction guards and combined stack tests are awaiting their own fresh proof. This is a draft foundation delivery for issue #94, with no general cache throughput claim.

## Traceability

`receipt.json` records the sealed journal, 52 original compressed log/JSON captures and exact copied helpers. `raw/manifest.json.gz` binds all compiled Rust/native/Cargo source bytes, the original configuration document, command results and CLI/proxy/runner/DLL identities. Models, binaries and checkpoint payload files are omitted. Captured JSON includes checkpoint hashes, requests/responses, counters and memory observations; sampled device VRAM is global, not isolated process VRAM.

Source bytes were checked against actual indexed Git blobs with CRLF/LF normalization. Raw working-file hashes can differ from another platform's checkout bytes. Capture/helper paths have `-text` and their Git blobs were checked byte for byte. Diagnostic logs retain original-byte hashes and decoding sidecars. The configuration document's original pending wording is preserved because it is included in the source-bound manifest; this report records the validation now completed within the scope above. Local harness paths and serial ownership journals need adaptation for reproduction.
