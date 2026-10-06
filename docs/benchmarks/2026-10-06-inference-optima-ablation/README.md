# Ablation: TinyLlama CPU — baseline vs PLD n-gram speculative

Part of [#144](https://github.com/azerothl/Rbitnet/issues/144) research note
[`docs/research/2026-10-06-inference-optima.md`](../../research/2026-10-06-inference-optima.md).

## Command

```bash
export RBITNET_MODEL=/path/to/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
export RBITNET_TOKENIZER=/path/to/tokenizer.json
export RBITNET_BACKEND=cpu
export RBITNET_MAX_SEQ=512
export RBITNET_SMOKE_MAX_TOKENS=32
# baseline
unset RBITNET_SPECULATIVE RBITNET_DRAFT_PATH
cargo run -p bitnet-core --example engine_smoke --release
# treatment
export RBITNET_SPECULATIVE=1 RBITNET_DRAFT_PATH=ngram
cargo run -p bitnet-core --example engine_smoke --release
```

See `results.txt` for captured timings (2026-10-06).
