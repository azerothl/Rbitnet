# MLA block prefill validation (2026-10-08)

Opt-in GLM/MLA causal block prefill for fixed expert banks (`RBITNET_CUDA_MLA_PREFILL=1`).

## Build CUDA quant DLL

```powershell
cd E:\devs\Rbitnet\native\cuda_quant
# Follow native/cuda_quant/README.md for your toolchain; then copy the DLL where bitnet-core expects it.
```

## Synthetic parity (no GGUF)

```powershell
$env:RBITNET_CUDA_QUANT_SMOKE = "1"
cargo test -p bitnet-core mla_block_prefill_matches_serial_logits_on_fixed_banks
```

## Real GLM (RTX 4080 SUPER)

Models under `D:/Rbitnet-benchmark-models/` (same as prior MLA evidence):

```powershell
$env:RBITNET_MLA_FULL_TEST = "1"
$env:RBITNET_MLA_TEST_GGUF = "D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/..."
$env:RBITNET_MLA_TEST_TOKENIZER = "..."
$env:RBITNET_CUDA_MLA_FULL = "1"
$env:RBITNET_CUDA_MLA_PREFILL = "1"
$env:RBITNET_CUDA_MLA_PREFILL_TOKENS = "16"
.\scripts\validate_mla_full.ps1
```

Compare `rbitnet_core_gpu_prefill_blocks_total` and prefill phase timings with block prefill off.

## Limits

- Block path requires non-dynamic fixed expert banks; segmented admission and per-layer CPU MoE fallback stay serial.
- GPT segmented/cache block prefill remains open for #95.
