# GPT ordered normalization

The resident GPT-OSS pipeline now stages independent input loads and squared values across a CUDA block before retaining the original serial round-to-nearest sum. Normalized outputs are written in parallel. Residual additions, multiplication order and the scalar fold remain unchanged, preserving the tested output bits rather than changing the reduction tree.

Ordered widths up to 8192 use at most 32 KiB of shared memory. Larger widths retain the original scalar kernel. This applies to token and fixed-bank block normalization; the router and other ordered projections are unchanged. There are no additional persistent allocations or feature flags.

The fresh [numerical, serving and performance proof](benchmarks/2026-10-04-gpt-norm-fresh/README.md) records 108 independent normalization cases, F64 fixed/segmented/block fixtures, 145 exact complete-output captures per baseline/new DLL, and HTTP/SSE behavior. On this machine, two measured repeats show approximately 28% faster fixed-bank decode and 12–26% faster segmented decode. These are bounded same-engine measurements, with no cross-engine parity claim.
