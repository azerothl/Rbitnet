# Using Rbitnet — models, tokenizer, and runtime

## Do you need Python?

**No — not for running Rbitnet.** Inference is implemented in **Rust** (`bitnet-core`): GGUF is memory-mapped, weights are dequantized in-process, and text is generated via the [`Engine`](../crates/bitnet-core/src/inference.rs) or the HTTP server.

You **only need Python (or another stack)** if you are **converting** checkpoints from Hugging Face Safetensors into **GGUF** using upstream tools (for example [microsoft/BitNet](https://github.com/microsoft/BitNet) scripts). That is a one-time **export** step on the machine where you build the file, not a runtime dependency of `rbitnet-server`.

## Hugging Face: curated list, search, and download (no Python)

The **`rbitnet`** binary (crate `rbitnet-cli`) lists a **curated** model index, can **search** the Hugging Face Hub for repos that expose **`.gguf`** files, and **downloads** files into a directory using the same cache layout as the Python hub (`HF_TOKEN` / `--token` for gated models).

**Why many HF BitNet repos do not “just work”:** Rbitnet loads **GGUF + Llama-shaped** graphs and a **tokenizer file** on disk; Hugging Face often splits **Safetensors vs GGUF** across repos, or documents **AutoTokenizer** from another (sometimes **gated**) repository. See **[HF_BITNET_RBITNET_GAP.md](HF_BITNET_RBITNET_GAP.md)** for the full gap table, readiness labels, and `models install` bundles.

| Command | Purpose |
|--------|---------|
| `rbitnet models list` | Print the curated catalog (default: raw `data/compatible_models.json` on GitHub). Override with `RBITNET_MODELS_INDEX_URL`. |
| `rbitnet models list --interactive` (`-i`) | Same catalog in a **terminal UI** (table + detail panel + download with `d`). Target directory: `--download-dir` or `RBITNET_DOWNLOAD_DIR` (default `models`); optional `HF_TOKEN` for gated downloads. |
| `rbitnet models search <query>` | Query the Hub API and show repos that have at least one `.gguf` (not project-tested — see stderr warning). Includes a heuristic `confidence` label for BitNet likelihood and an **`rbitnet=`** readiness hint (`ready`, `needs_tokenizer`, `needs_external_tokenizer`, `unsupported_arch_likely`, `experimental_gguf` — see [HF_BITNET_RBITNET_GAP.md](HF_BITNET_RBITNET_GAP.md)). **Default mode is strict BitNet filtering** (`likely`/`possible` only). |
| `rbitnet models search <query> --all-gguf` | Disable strict filtering and show all GGUF repos, including `generic-gguf`. |
| `rbitnet models search <query> -i` | Same search as an **interactive** table (detail + `d` download like `models download` without `--file`). Press `f` to toggle between the default strict BitNet filter and `all-gguf`. Readiness appears in the **rbitnet** column. |
| `rbitnet models install --list` | Print curated **bundle** ids (paired GGUF repo + tokenizer source). |
| `rbitnet models install <bundle-id> --dir DIR` | Download the bundle into `DIR` and write **`rbitnet.manifest.json`** with suggested `RBITNET_MODEL` / `RBITNET_TOKENIZER` paths (relative). Uses `HF_TOKEN` when the Hub requires it. |
| `rbitnet models generate-catalog` | Build a `compatible_models.json` **draft** from Hub search (one GGUF + tokenizer per repo when found). Review before commit — see below. |
| `rbitnet models download <repo_id> [--dir DIR] [--file NAME]...` | Download files (repeat `--file`; if omitted, all `.gguf` plus tokenizer files when present). |
| `rbitnet serve` | Same HTTP server as `rbitnet-server` (same `RBITNET_*` env vars). |

**Compatibility:** Only entries in the **curated** list are maintained for Rbitnet testing. Search hits are **best-effort** Hub results based on `.gguf` file presence only. **Important:** `.gguf` does **not** imply BitNet 1-bit weights nor guaranteed Rbitnet compatibility.

**TLS / networking (Windows):** Hub calls and `raw.githubusercontent.com` use **native-tls** (the OS certificate store) and honor **`HTTPS_PROXY` / `HTTP_PROXY`** when set. If you see errors such as "connection closed by remote host" (10054) or TLS initialization failures, check your proxy, antivirus, or corporate HTTPS inspection settings.

**Regenerate the catalog file without filling it in by hand:** `models generate-catalog` queries the Hugging Face API (like `search`), selects **one** `.gguf` file per repo (using a `Q4_K_M`-style heuristic when present), and adds `tokenizer.json` / `tokenizer.model` if they are listed in the repo. The default query is **`gguf`** (`llama`-style searches often return Safetensors repos first, without `.gguf` files to inspect). The output is a **draft** to review, then commit to `data/compatible_models.json`.

```bash
cargo build -p rbitnet-cli --release
./target/release/rbitnet models list
# Interactive table: ↑/↓ or j/k (row), PgUp/PgDn (detail), d (download), q or Esc (quit)
./target/release/rbitnet models list -i --download-dir ./models
./target/release/rbitnet models search llama
./target/release/rbitnet models search llama --all-gguf
./target/release/rbitnet models search gguf -i
./target/release/rbitnet models search bitnet -i
./target/release/rbitnet models install --list
./target/release/rbitnet models install microsoft-bitnet-b1.58-2b-4t --dir ./models
# JSON draft to stdout (or --output data/compatible_models.json)
./target/release/rbitnet models generate-catalog --max-entries 40 --output data/compatible_models.json
# (default: `--query gguf`; for a specific family: `--query llama` + `--max-inspect 400`)
./target/release/rbitnet models download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --dir ./models --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --file tokenizer.json
```

**Interactive mode:** The catalog or search results are displayed in a **table** (ratatui). The selected row fills the **Detail** panel; **`d`** downloads the listed files (curated catalog: JSON file list; search: `.gguf` + `tokenizer.json` / `tokenizer.model` **if** present in Hub siblings). In interactive search mode, **`f`** toggles the filter between strict BitNet and `all-gguf`. Target directory: **`--download-dir`** or environment variable **`RBITNET_DOWNLOAD_DIR`** (default `models`). Optional Hub token: **`HF_TOKEN`** / **`--token`** (for search and private repos).

**BitNet heuristic (`confidence`):** `likely-bitnet`, `possible-bitnet`, and `generic-gguf` are **textual hints** (repo/file name matching) and not a formal validation. Strict mode (the default) combines the Hub `other=bitnet` filter with this heuristic to reduce noise; entries should still be manually validated before production use.

**Tokenizers on the Hub:** Many BitNet / Transformers repos document `AutoTokenizer.from_pretrained(...)` without publishing a `tokenizer.json` or `tokenizer.model` in the same repo as the GGUF (e.g., tokenizer loaded from a different repo, or only Safetensors weights). Rbitnet requires a **file** `tokenizer.json` or `tokenizer.model` alongside the GGUF, or **`RBITNET_TOKENIZER`** pointing to one of those files. The `models search` command can only list what the `siblings` API exposes; if the model card points to a different Hugging Face id for the tokenizer, download that file from that repo or set `RBITNET_TOKENIZER` accordingly.
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
