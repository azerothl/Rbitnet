# GPT-OSS segmented block prefill

Opt-in `RBITNET_CUDA_GPT_PREFILL=1` on segmented/cache GPT resident paths batches attention and router projections over 1–32 tokens per block. Routed FFN admission stays on the host (cache, adaptive, or CPU fallback), matching the serial segmented pipeline.

## Validate

With CUDA and a rebuilt `rbitnet_cuda_quant` DLL:

```powershell
cargo test -p bitnet-core gpt_segmented_block_prefill_matches_serial_segmented -- --nocapture
```

Optional end-to-end (real GGUF): set `RBITNET_GPT_SEGMENTED_TEST=1`, `RBITNET_CUDA_GPT_FULL=1`, `RBITNET_CUDA_GPT_PREFILL=1`, and a positive `RBITNET_MOE_CACHE_MB` or partial placement, then run the existing segmented runtime test harness.

## Limits

- MLA block prefill remains in [PR #154](https://github.com/azerothl/Rbitnet/pull/154).
- No published tok/s ablation in this note; compare serial vs block on the same revision with `scripts/benchmark_cache_stack.py` once a GPT-OSS cache fixture is configured.
