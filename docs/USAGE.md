# Using Rbitnet — models, tokenizer, and runtime

## Do you need Python?

**No — not for running Rbitnet.** Inference is implemented in **Rust** (`bitnet-core`): GGUF is memory-mapped, weights are dequantized in-process, and text is generated via the [`Engine`](../crates/bitnet-core/src/inference.rs) or the HTTP server.

You **only need Python (or another stack)** if you are **converting** checkpoints from Hugging Face Safetensors into **GGUF** using upstream tools (for example [microsoft/BitNet](https://github.com/microsoft/BitNet) scripts). That is a one-time **export** step on the machine where you build the file, not a runtime dependency of `rbitnet-server`.

## Hugging Face: curated list, search, and download (no Python)

The **`rbitnet`** binary (crate `rbitnet-cli`) lists a **curated** model index, can **search** the Hugging Face Hub for repos that expose **`.gguf`** files, and **downloads** files into a directory using the same cache layout as the Python hub (`HF_TOKEN` / `--token` for gated models).

| Command | Purpose |
|--------|---------|
| `rbitnet models list` | Print the curated catalog (default: raw `data/compatible_models.json` on GitHub). Override with `RBITNET_MODELS_INDEX_URL`. |
| `rbitnet models search <query>` | Query the Hub API and show repos that have at least one `.gguf` (not project-tested — see stderr warning). |
| `rbitnet models download <repo_id> [--dir DIR] [--file NAME]...` | Download files (repeat `--file`; if omitted, all `.gguf` plus tokenizer files when present). |
| `rbitnet serve` | Same HTTP server as `rbitnet-server` (same `RBITNET_*` env vars). |

**Compatibility:** Only entries in the **curated** list are maintained for Rbitnet testing. Search hits are **best-effort** Hub results; repos that ship **only Safetensors** are naturally excluded when no `.gguf` exists in the tree.

```bash
cargo build -p rbitnet-cli --release
./target/release/rbitnet models list
./target/release/rbitnet models search llama
./target/release/rbitnet models download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --dir ./models --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --file tokenizer.json
```

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

**Health and metrics (operations):**

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
curl -s http://127.0.0.1:8080/metrics
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

### Engine

| Variable | Meaning |
|----------|---------|
| `RBITNET_BIND` | Listen address (default `127.0.0.1:8080`). |
| `RBITNET_MODEL` | Path to a `.gguf` file (must not contain `..` path components). |
| `RBITNET_TOKENIZER` | Path to `tokenizer.json` if not next to the GGUF (must not contain `..`). |
| `RBITNET_STUB` | `1` / `true` / `yes` — stub responses (overrides real inference when set). |
| `RBITNET_TOY` | `1` — toy LM instead of GGUF. |
| `RBITNET_TOY_SEED` | Integer seed for the toy LM (default `42`). |
| `RBITNET_TEST_GGUF` | Used only by the `optional_gguf_from_env_smoke` test in `bitnet-core`. |

### Server limits and security (`rbitnet-server`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `RBITNET_MAX_BODY_BYTES` | `1048576` | Max JSON body size for `/v1/chat/completions` (HTTP 413 if exceeded). |
| `RBITNET_MAX_PROMPT_CHARS` | `256000` | Max UTF-8 characters in the built prompt (HTTP 400). |
| `RBITNET_MAX_TOKENS_CAP` | `8192` | Hard ceiling on client `max_tokens` (HTTP 400 if higher). |
| `RBITNET_MAX_CONCURRENT` | `4` | Simultaneous blocking inference tasks (HTTP 503 when saturated). |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | `600` | Wall-clock limit per completion (HTTP 504). |
| `RBITNET_API_KEY` | unset | If set, `Authorization: Bearer <key>` or `X-API-Key: <key>` is required on `/`, `/v1/models`, and `/v1/chat/completions` (not on `/health`, `/ready`, `/metrics`). |
| `RBITNET_CORS_ANY` | unset | Set to `1` only for dev to allow any CORS origin. |

Binding to `0.0.0.0` or `[::]` logs a warning: use a reverse proxy and TLS for untrusted networks ([DEPLOYMENT.md](DEPLOYMENT.md)).

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
- **Tensor names:** The loader expects **llama.cpp-style** names, with a few **aliases** (for example `lm_head.weight` vs `output.weight`). See [BITNET_SPEC.md](BITNET_SPEC.md). Odd exports may still need renaming or loader tweaks.
- **Context length:** Generation is bounded by `llama.context_length` (capped internally for safety). Very long prompts can hit limits or run slowly on CPU.
- **Performance:** Pure Rust + dequantized matmuls is correct but not as fast as highly optimized C++/GPU stacks; for production throughput, profile on your hardware.

For golden tests and regression expectations, see [GOLDEN_TESTS.md](GOLDEN_TESTS.md). For HF → GGUF conversion workflows, see [MODEL_TESTING.md](MODEL_TESTING.md). For **what is implemented vs still planned** (production roadmap), see [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md).
