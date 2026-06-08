# Rbitnet

Pure Rust **Llama-compatible GGUF inference** and an **OpenAI-compatible HTTP server** for [Akasha](https://github.com/loicpeaudecerf/Akasha) (`BitNetProvider`).

## Do I need Python?

**Not to run Rbitnet.** The server and `bitnet-core` are **self-sufficient in Rust**: mmap the GGUF, dequantize weights, run the transformer, sample tokens. No external inference engine is required.

You only need **Python (or other tools)** if you are **converting** a Hugging Face / Safetensors checkpoint into **GGUF** upstream (for example Microsoft BitNet or `llama.cpp` converters). That is export-time, not a runtime dependency.

Optional helper: `[scripts/setup_env.py](scripts/setup_env.py)` — download HF weights (`huggingface_hub`), print `RBITNET_`* lines; **calling Microsoft BitNet is optional** — see [docs/MODEL_TESTING.md](docs/MODEL_TESTING.md).

**Start here:** [docs/USAGE.md](docs/USAGE.md) (models, tokenizer, env vars, hot reload, TUI chat, curl examples). **Quick start (FR):** [docs/DEMARRAGE_5MIN.md](docs/DEMARRAGE_5MIN.md).

## Installation

### Windows

Build from a checkout:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Tagged releases publish Windows zip assets named like `rbitnet-server-vX.Y.Z-windows-x86_64.zip` on [GitHub Releases](https://github.com/azerothl/Rbitnet/releases). The zip contains both `rbitnet.exe` and `rbitnet-server.exe`.

WinGet is prepared as a submission template at `[packaging/winget/Rbitnet.Rbitnet.yaml](packaging/winget/Rbitnet.Rbitnet.yaml)`. After replacing `PackageVersion`, `InstallerUrl`, and `InstallerSha256` for a tagged release, install/test locally with WinGet tooling or submit it to `microsoft/winget-pkgs`:

```powershell
winget install --manifest .\packaging\winget\Rbitnet.Rbitnet.yaml
```

### macOS / Linux

Build from a checkout:

```bash
./scripts/install.sh
```

Or install the CLI from the default branch with a curl script (requires Rust/Cargo and git):

```bash
curl -fsSL https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.sh | sh
```

Tagged releases publish tarballs named like `rbitnet-server-vX.Y.Z-linux-x86_64.tar.gz` and `rbitnet-server-vX.Y.Z-macos-arm64.tar.gz`. A documented Homebrew tap formula template lives at `[packaging/homebrew/rbitnet.rb](packaging/homebrew/rbitnet.rb)`; it is not a homebrew-core formula. After replacing the release URLs and `sha256` values:

```bash
brew install --formula ./packaging/homebrew/rbitnet.rb
```

For a future tap, copy the formula into a tap repo and use:

```bash
brew tap <owner>/rbitnet
brew install rbitnet
```

Direct install from the working tree remains:

```bash
cargo install --path crates/rbitnet-cli --locked
```

### Docker

The included `[Dockerfile](Dockerfile)` builds `rbitnet-server` and is usable for server-only deployments:

```bash
docker build -t rbitnet:local .
docker run --rm -e RBITNET_MODEL=/model/model.gguf -e RBITNET_TOKENIZER=/model/tokenizer.json \
  -v /path/on/host:/model:ro -p 8080:8080 rbitnet:local
```

## Status

- **bitnet-core**: GGUF parse, GGML dequantization, Llama/Qwen3-shaped forward paths, native BitNet GGUF path, mmap quantized matvec, CPU parallel quant kernels, scratch arena reuse, paged/quantized KV cache hooks, hybrid CPU/GPU offload planning, structured-output sampling mask, and optional toy LM.
- **bitnet-server** (`rbitnet-server`): OpenAI-compatible API; `GET /health`, `GET /ready`, `GET /metrics`; `GET /`, `GET /ui`, `GET /v1/models`, `POST /v1/chat/completions` (JSON + SSE), `POST /v1/admin/unload`, and `POST /v1/admin/reload`. Supports runtime model reload without stopping the HTTP process.
- **rbitnet CLI** (`rbitnet`): curated model list/search/download/install, `quickstart`, `up`, `serve`, local web UI launcher, and terminal chatbot TUI via `rbitnet chat`.
- **Docs (English)**:
  - **[docs/USAGE.md](docs/USAGE.md)** — how to run a model (no Python at runtime)
  - **[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)** — curl, Python OpenAI, Node OpenAI, LiteLLM, Akasha
  - **[docs/CURATED_MODELS.md](docs/CURATED_MODELS.md)** — catalog schema and verification policy
  - **[docs/TRAINING_AND_COMPATIBILITY.md](docs/TRAINING_AND_COMPATIBILITY.md)** — training elsewhere, export to GGUF, compatibility rules
  - **[training/README.md](training/README.md)** — optional Python LoRA/SFT recipe; `rbitnet train` / `rbitnet export-gguf`
  - **[docs/PLAN_PRODUCTION.md](docs/PLAN_PRODUCTION.md)** — roadmap and exit criteria for a production-ready release
  - **[CHANGELOG.md](CHANGELOG.md)** — release-facing changes (Keep a Changelog style)
  - **[docs/STATUS_AND_ROADMAP.md](docs/STATUS_AND_ROADMAP.md)** — what is implemented vs missing, next todos
  - **[docs/ENV_REFERENCE.md](docs/ENV_REFERENCE.md)** — consolidated `RBITNET_`* variables
  - **[docs/LOCAL_PERFORMANCE_2026-05-12.md](docs/LOCAL_PERFORMANCE_2026-05-12.md)** — local CPU/CUDA/hybrid performance notes and post-optimization comparison
  - **[docs/NATIVE_FIRST.md](docs/NATIVE_FIRST.md)** — politique native-first: aucun moteur d'inference externe requis
  - **[docs/GPU_NATIVE_ROADMAP.md](docs/GPU_NATIVE_ROADMAP.md)** — feuille de route GPU native dans `bitnet-core`
  - [docs/BITNET_SPEC.md](docs/BITNET_SPEC.md) — format / metadata expectations
  - [docs/GOLDEN_TESTS.md](docs/GOLDEN_TESTS.md) — golden / regression testing
  - [docs/MODEL_TESTING.md](docs/MODEL_TESTING.md) — HF `bitnet_b1_58-large` and GGUF conversion
  - [docs/MODEL_MATRIX.md](docs/MODEL_MATRIX.md) — curated-model RAM / tok/s placeholders and reproduction commands
  - [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — how to record kernel and HTTP benchmarks
  - [docs/BENCHMARKS_RESULTS.md](docs/BENCHMARKS_RESULTS.md) — append-only local benchmark output
  - [docs/RELEASE_PACKAGING.md](docs/RELEASE_PACKAGING.md) — WinGet/Homebrew release checklist
  - [docs/FUTURE_DIFFERENTIATION.md](docs/FUTURE_DIFFERENTIATION.md) — French future differentiation backlog
  - [docs/PROFILING.md](docs/PROFILING.md) — CPU profiling checklist (Phase 2)
  - [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — systemd / reverse proxy / health checks
  - [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — performance and format constraints
  - [docs/RELEASE.md](docs/RELEASE.md) — versioning and release checklist
  - [docs/INFERENCE_STACK_V2.md](docs/INFERENCE_STACK_V2.md) — long-term inference backlog (PagedAttention-class epic)

## Works Today

Rbitnet runs **Llama-architecture GGUF** models and now has a native BitNet GGUF path for Llama-shaped BitNet b1.58 / ternary exports tagged with `general.architecture=bitnet`. The first curated target is the Microsoft `microsoft-bitnet-b1.58-2b-4t` bundle; see [docs/BITNET_NATIVE.md](docs/BITNET_NATIVE.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

Recent inference-engine features include:

- `RBITNET_QUANT_KERNEL=auto` for the shared CPU-parallel quantized matvec path.
- `RBITNET_BACKEND=hybrid` plus `RBITNET_HYBRID_POLICY={layers,hotcold,auto}` for CPU/GPU offload planning.
- `RBITNET_LLAMA_PAGED_KV=1` and `RBITNET_KV_QUANT={off,q8,q4}` for paged Llama KV experiments.
- `RBITNET_PREFILL_CHUNK_TOKENS`, speculative draft hooks (`RBITNET_DRAFT_PATH`), and JSON/tool structured-output masking (`RBITNET_STRUCTURED_OUTPUT`).
- Runtime admin reload through `POST /v1/admin/reload`, useful for switching models/configs while the server stays up.

Concrete public GGUF repos verified through the Hugging Face model API as `gguf.architecture=llama`:

- `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` with `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (small CPU smoke model).
- `unsloth/Llama-3.2-1B-Instruct-GGUF` with `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.
- `NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF` with `Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf` (`RBITNET_CHAT_FORMAT=chatml` recommended).

Native BitNet curated install:

```bash
rbitnet models install microsoft-bitnet-b1.58-2b-4t --dir ./models
export RBITNET_MODEL=/absolute/path/to/ggml-model-i2_s.gguf
export RBITNET_TOKENIZER=/absolute/path/to/tokenizer.json
cargo run -p bitnet-server --bin rbitnet-server --release
```

Install the CLI, download a GGUF, set a tokenizer file, then serve:

```bash
cargo install --path crates/rbitnet-cli
rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# Or write rbitnet.toml so serve needs no model env vars:
rbitnet up TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# If the GGUF repo did not include tokenizer.json/tokenizer.model:
rbitnet models download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dir models/tinyllama-tokenizer --file tokenizer.json
export RBITNET_TOKENIZER="$PWD/models/tinyllama-tokenizer/tokenizer.json"
rbitnet serve
```

For Windows quick install:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Run the server (real GGUF)

Place `**tokenizer.json**` (or `tokenizer.model`) beside the `.gguf`, or set `RBITNET_TOKENIZER`.
`rbitnet serve` and `rbitnet-server` also read `rbitnet.toml` from the current directory, `RBITNET_CONFIG`, or the user config directory; explicit env vars still win.

```bash
export RBITNET_MODEL=/absolute/path/to/model.gguf
export RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server --release
```

Stub (no weights, integration text):

```bash
export RBITNET_STUB=1
export RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server
```

Toy LM (no GGUF):

```bash
export RBITNET_TOY=1
cargo run -p bitnet-server --bin rbitnet-server
```

Open the local static UI from the same server:

```bash
RBITNET_STUB=1 rbitnet serve --open-ui
# or visit http://127.0.0.1:8080/ui
```

## Quick terminal testing

Use the terminal chatbot TUI to test a running server:

```bash
rbitnet chat --base-url http://127.0.0.1:8080/v1 --model rbitnet-llama
```

Or let the TUI launch and manage a local server:

```bash
rbitnet chat --serve \
  --model-path /absolute/path/to/model.gguf \
  --tokenizer /absolute/path/to/tokenizer.json \
  --chat-format raw
```

Useful keys:

| Key | Action |
|-----|--------|
| `Enter` | Send the prompt. |
| `F2` / `F3` | Decrease / increase `max_tokens`. |
| `-` / `+` | Decrease / increase `temperature`. |
| `F4` | Fetch `/v1/models`. |
| `Ctrl+R` | Call `POST /v1/admin/reload` when `RBITNET_ADMIN_TOKEN` is set. |
| `Ctrl+U` | Call `POST /v1/admin/unload` when `RBITNET_ADMIN_TOKEN` is set. |
| `Esc` or `Ctrl+C` | Quit. |

Add `--transcript chat.jsonl` to append prompt/reply rows during quick regression tests.

## Hot reload

Set an admin token before starting the server:

```bash
export RBITNET_ADMIN_TOKEN=dev-secret
```

Reload the current env/config model without restarting the HTTP process:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/admin/reload \
  -H 'X-Rbitnet-Admin-Token: dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Reload a registry model:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/admin/reload \
  -H 'X-Rbitnet-Admin-Token: dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{"active_model_id":"tiny"}'
```

Reload a specific GGUF:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/admin/reload \
  -H 'X-Rbitnet-Admin-Token: dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/tiny.gguf","tokenizer":"/models/tokenizer.json","architecture":"llama"}'
```

Reload metrics are exposed as `rbitnet_model_reloads_total`, `rbitnet_model_reload_failures_total`, and `rbitnet_model_reload_ms_sum`.

## Benchmarks

Use [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the procedure and [docs/BENCHMARKS_RESULTS.md](docs/BENCHMARKS_RESULTS.md) for appended rows. The helper scripts start a stub server by default for API-overhead smoke checks; for real numbers, start a model yourself and run `scripts/bench_matrix.*` with `NO_START_SERVER=1` / `-NoStartServer`.

## Inspect a GGUF

```bash
cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
```

## Test BitNet / Ternary GGUF

Use the curated `microsoft-bitnet-b1.58-2b-4t` bundle for native GGUF inference, or convert Safetensors with **Microsoft BitNet** tooling, then add the tokenizer and point `RBITNET_MODEL` at the `.gguf`. Walkthrough: **[docs/BITNET_NATIVE.md](docs/BITNET_NATIVE.md)** and **[docs/MODEL_TESTING.md](docs/MODEL_TESTING.md)**.

Optional automated parse check (local only):

```bash
export RBITNET_TEST_GGUF=/path/to/model.gguf
cargo test -p bitnet-core optional_gguf_from_env_smoke -- --nocapture
```

## Akasha

1. Run `rbitnet-server` (see [docs/USAGE.md](docs/USAGE.md)).
2. In your data directory, edit `llm_router.yaml`:

```yaml
providers:
  bitnet:
    base_url: "http://127.0.0.1:8080"

task_types:
  conversation:
    primary:
      provider: bitnet
      model: rbitnet-llama
    fallback:
      - provider: akasha_core
        model: core
```

The `model` field must match an `id` from `GET /v1/models` (`rbitnet-stub`, `rbitnet-toy`, or `rbitnet-<architecture>` when a GGUF is loaded).

See the Akasha repo: `spec/llm_router.example.yaml`.

### Hermes / Akasha ecosystem (ops)

Rbitnet is the **local OpenAI-compatible** backend in the Akasha multi-reference parity story. For self-hosted SLO alignment with the daemon router, scrape `**GET /metrics`** on Rbitnet and compare with Akasha’s `**GET /api/router/metrics**` (see [docs/USAGE.md](docs/USAGE.md) § correlation). Product parity tracking: [Akasha `spec/dev/roadmap/reference-products-parity-matrix.md](https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/reference-products-parity-matrix.md)` and [Akasha `spec/dev/roadmap/hermes-integration-remainder.md](https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/hermes-integration-remainder.md)`.

## Environment


| Variable            | Meaning                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `RBITNET_BIND` | Host:port (default `127.0.0.1:8080`) |
| `RBITNET_MODEL` | Path to `.gguf` for real inference |
| `RBITNET_TOKENIZER` | Path to `tokenizer.json` or `tokenizer.model` if not beside the GGUF |
| `RBITNET_STUB` | `1` = stub text (no inference) |
| `RBITNET_TOY` | `1` = tiny in-process F32 toy LM (no GGUF) |
| `RBITNET_BACKEND` | `cpu`, `cuda`, or `hybrid` |
| `RBITNET_QUANT_KERNEL` | `auto`, `scalar`, or `cuda` quantized matvec backend |
| `RBITNET_HYBRID_POLICY` | `layers`, `hotcold`, or `auto` layer offload policy |
| `RBITNET_LLAMA_PAGED_KV` / `RBITNET_KV_QUANT` | Enable paged KV and choose `off`, `q8`, or `q4` KV format |
| `RBITNET_ADMIN_TOKEN` | Enables `POST /v1/admin/unload` and `POST /v1/admin/reload` |
| `RBITNET_CHAT_BASE_URL` | Default base URL for `rbitnet chat` |
| `RBITNET_TOY_SEED` | Seed for toy weights (default `42`) |
| `RBITNET_TEST_GGUF` | Optional path for `optional_gguf_from_env_smoke` test only |


Server tuning (`rbitnet-server`): `RBITNET_MAX_BODY_BYTES`, `RBITNET_MAX_PROMPT_CHARS`, `RBITNET_MAX_TOKENS_CAP`, `RBITNET_MAX_CONCURRENT`, `RBITNET_INFERENCE_TIMEOUT_SECS`, `RBITNET_API_KEY` — see [docs/USAGE.md](docs/USAGE.md).

## Benchmarks

```bash
cargo bench -p bitnet-core
```

## License

MIT — see [LICENSE](LICENSE).