# Mutable TF32 guard for encoded Llama KV

The C ABI now refuses enabling TF32 prefill on an owner using F16 or Q8 KV,
before changing its arithmetic mode or shared paged-pool state. Rust already
refuses this combination at construction; the C ABI needed the same guard.
F32 remains the default and a fresh F32 owner still accepts configuration.

Fresh workspace check, Clippy and tests passed (298 passed,
1 ignored), followed by a rebuilt Native DLL and 64 independent
F64 encoded-attention cases. With split KV both off and on, actual Llama tests
each passed eight F16/Q8 dense/paged, graph-off/on refusal configurations.
Continuation logits match untouched owners bit for bit; shared-pool creation
and filled-owner refusal are checked. Canonical prefix/state suites also passed
with both split modes.

This is a guard-only correction. Encoded storage/attention kernels are unchanged.
Full serving, long-context and capacity evidence remains in
[the original production proof](../2026-10-04-kv-formats/production/).
Encoded KV still trades decoding speed for capacity; KIVI is not implemented.
This validation makes no additional quality, speed or engine-parity claim.

`receipt.json` indexes 24 original gzip captures and exact helpers;
`raw/manifest.json.gz` binds all compiled sources and the executed DLL identity.
The three correction/test files copied into this PR match the tested bytes.
Original checker paths require local predecessor evidence and model adaptation.
Models and the Native DLL are omitted from Git.
