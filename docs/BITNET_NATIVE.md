# BitNet Native GGUF

Rbitnet includes a native Rust inference path for BitNet b1.58 / ternary GGUF files that follow the Microsoft BitNet + llama.cpp layout. It does not shell out to `bitnet.cpp`, Python, or another inference binary.

## Supported Models

Primary target:

- `microsoft-bitnet-b1.58-2b-4t`
- GGUF repo: `microsoft/bitnet-b1.58-2B-4T-gguf`
- Tokenizer repo: `microsoft/bitnet-b1.58-2B-4T`
- Expected architecture metadata: `general.architecture=bitnet`
- Expected tensor layout: Llama-shaped GGUF names such as `token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`, `blk.N.ffn_down.weight`, `output.weight`

The native path supports ternary GGML matrix types used by these exports:

- `TQ1_0` (`ggml_type=34`)
- `TQ2_0` (`ggml_type=35`)

Other standard GGML types already supported by the mmap GEMV path remain available for embeddings, output heads, or mixed exports.

## How To Run

Install the curated bundle:

```powershell
cargo run -p rbitnet-cli -- models install microsoft-bitnet-b1.58-2b-4t --dir .\models
```

Then point the runtime at the GGUF and tokenizer. If `rbitnet.manifest.json` was written, use the paths printed by the installer.

```powershell
$env:RBITNET_MODEL="C:\path\to\ggml-model-i2_s.gguf"
$env:RBITNET_TOKENIZER="C:\path\to\tokenizer.json"
$env:RBITNET_BACKEND="cpu"
cargo run -p bitnet-server --bin rbitnet-server --release
```

Validation smoke:

```powershell
cargo run -p bitnet-core --example inspect_gguf -- $env:RBITNET_MODEL
cargo run -p bitnet-core --example engine_smoke
```

## Environment Variables

- `RBITNET_MODEL`: absolute path to one `.gguf` file.
- `RBITNET_TOKENIZER`: tokenizer file from the paired model repo, usually `tokenizer.json`.
- `RBITNET_BACKEND`: `cpu` is the supported baseline. Other backend labels currently share the portable runtime unless a family-specific CUDA path says otherwise.
- `RBITNET_LLAMA_WEIGHT_MODE`: defaults to `auto`; BitNet uses the same mmap-quant machinery and should stay in `auto` unless debugging.
- `RBITNET_PREFILL_CHUNK_TOKENS`: optional prefill chunk size for generation.

## Notes Et Limites

Le chemin BitNet natif reutilise le runtime transformeur Llama-shaped de `bitnet-core`, mais selectionne un executor `bitnet` lorsque `general.architecture=bitnet`. Les produits matrice-vecteur ternaires `TQ1_0` et `TQ2_0` sont executes directement depuis le mmap GGUF.

Ce qui reste a valider pour une parite complete:

- comparaison logits couche par couche avec l'implementation Microsoft / llama.cpp sur le modele complet;
- variantes GGUF dont les noms de tenseurs different des conventions Microsoft;
- kernels optimises SIMD/GPU pour les produits ternaires, au-dela du chemin CPU portable actuel.
