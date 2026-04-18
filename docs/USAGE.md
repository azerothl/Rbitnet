# Using Rbitnet — models, tokenizer, and runtime

## Do you need Python?

**No — not for running Rbitnet.** Inference is implemented in **Rust** (`bitnet-core`): GGUF is memory-mapped, weights are dequantized in-process, and text is generated via the [`Engine`](../crates/bitnet-core/src/inference.rs) or the HTTP server.

You **only need Python (or another stack)** if you are **converting** checkpoints from Hugging Face Safetensors into **GGUF** using upstream tools (for example [microsoft/BitNet](https://github.com/microsoft/BitNet) scripts). That is a one-time **export** step on the machine where you build the file, not a runtime dependency of `rbitnet-server`.

## Requirements to run a real model

1. A **`.gguf`** file with **Llama-compatible** layout (see [BITNET_SPEC.md](BITNET_SPEC.md) and [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md)).
2. A **tokenizer** file that the Hugging Face `tokenizers` crate can load:
   - Prefer **`tokenizer.json`** next to the GGUF, **or**
   - **`tokenizer.model`** (SentencePiece) in the same directory, **or**
   - Set **`RBITNET_TOKENIZER`** to the absolute path of either file.

Without a tokenizer, the engine returns `TokenizerMissing` when you try to generate text.

## Quick start — HTTP server with a GGUF

```bash
# Linux / macOS
export RBITNET_MODEL=/absolute/path/to/model.gguf
# Optional if tokenizer.json is not beside the GGUF:
# export RBITNET_TOKENIZER=/absolute/path/to/tokenizer.json
export RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server --release
```

```powershell
# Windows PowerShell
$env:RBITNET_MODEL="C:\path\to\model.gguf"
$env:RBITNET_BIND="127.0.0.1:8080"
cargo run -p bitnet-server --bin rbitnet-server --release
```

**Check models list:**

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Use the reported `id` (for example `rbitnet-llama` when `general.architecture` is `llama`) as the `model` field in chat requests.

**Chat completion (non-streaming):**

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rbitnet-llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0.8}'
```

## Modes without a full GGUF

| Mode | Purpose |
|------|---------|
| **`RBITNET_STUB=1`** | Integration smoke tests; no weights; canned reply. |
| **`RBITNET_TOY=1`** | Tiny built-in toy LM; no file; useful for CI and API checks. |

Do **not** set stub/toy if you want real generation from `RBITNET_MODEL`.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `RBITNET_BIND` | Listen address (default `127.0.0.1:8080`). |
| `RBITNET_MODEL` | Path to a `.gguf` file. |
| `RBITNET_TOKENIZER` | Path to `tokenizer.json` or `tokenizer.model` if not next to the GGUF. |
| `RBITNET_STUB` | `1` / `true` / `yes` — stub responses (overrides real inference when set). |
| `RBITNET_TOY` | `1` — toy LM instead of GGUF. |
| `RBITNET_TOY_SEED` | Integer seed for the toy LM (default `42`). |
| `RBITNET_TEST_GGUF` | Used only by the `optional_gguf_from_env_smoke` test in `bitnet-core`. |

## Inspect a GGUF (no server)

```bash
cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
```

## Programmatic use (Rust)

Depend on `bitnet-core` and build an [`Engine`](../crates/bitnet-core/src/inference.rs) from the environment or from a path:

```rust
use bitnet_core::Engine;

let engine = Engine::load_path(std::path::Path::new("/path/to/model.gguf"))?;
let text = engine.complete("Hello", 64, 0.8)?;
```

The first call that needs generation will load the tokenizer (same rules as above). Ensure `RBITNET_TOKENIZER` or a tokenizer beside the GGUF is available at runtime.

## Limitations and troubleshooting

- **Quantization types:** Some rare GGML types may not be implemented yet; loading can fail with `UnsupportedGgmlType`. Prefer widely used types (for example Q4_K, Q8_0, or F16 layers) or re-export with a supported layout.
- **Tensor names:** The loader expects **llama.cpp-style** names (`token_embd.weight`, `blk.N.attn_q.weight`, …). Custom exports may need renaming or a small fork of the loader.
- **Context length:** Generation is bounded by `llama.context_length` (capped internally for safety). Very long prompts can hit limits or run slowly on CPU.
- **Performance:** Pure Rust + dequantized matmuls is correct but not as fast as highly optimized C++/GPU stacks; for production throughput, profile on your hardware.

For golden tests and regression expectations, see [GOLDEN_TESTS.md](GOLDEN_TESTS.md). For HF → GGUF conversion workflows, see [MODEL_TESTING.md](MODEL_TESTING.md).
