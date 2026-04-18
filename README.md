# Rbitnet

Pure Rust **Llama-compatible GGUF inference** and an **OpenAI-compatible HTTP server** for [Akasha](https://github.com/loicpeaudecerf/Akasha) (`BitNetProvider`).

## Do I need Python?

**Not to run Rbitnet.** The server and `bitnet-core` are **self-sufficient in Rust**: mmap the GGUF, dequantize weights, run the transformer, sample tokens.

You only need **Python (or other tools)** if you are **converting** a Hugging Face / Safetensors checkpoint into **GGUF** upstream (for example Microsoft BitNet or `llama.cpp` converters). That is export-time, not a runtime dependency.

**Start here:** [docs/USAGE.md](docs/USAGE.md) (models, tokenizer, env vars, curl examples).

## Status

- **bitnet-core**: GGUF parse, GGML dequantization, Llama-shaped forward (RMSNorm, RoPE, GQA, KV cache, SiLU FFN), [`Engine`](crates/bitnet-core/src/inference.rs), optional toy LM.
- **bitnet-server** (`rbitnet-server`): `GET /`, `GET /v1/models`, `POST /v1/chat/completions` (JSON + SSE). Integration tests for stub mode.
- **Docs (English)**:
  - **[docs/USAGE.md](docs/USAGE.md)** — how to run a model (no Python at runtime)
  - **[docs/TRAINING_AND_COMPATIBILITY.md](docs/TRAINING_AND_COMPATIBILITY.md)** — training elsewhere, export to GGUF, compatibility rules
  - **[docs/PLAN_PRODUCTION.md](docs/PLAN_PRODUCTION.md)** — plan de feuille de route pour une version « prod ready »
  - [docs/BITNET_SPEC.md](docs/BITNET_SPEC.md) — format / metadata expectations
  - [docs/GOLDEN_TESTS.md](docs/GOLDEN_TESTS.md) — golden / regression testing
  - [docs/MODEL_TESTING.md](docs/MODEL_TESTING.md) — HF `bitnet_b1_58-large` and GGUF conversion

## Run the server (real GGUF)

Place **`tokenizer.json`** (or `tokenizer.model`) beside the `.gguf`, or set `RBITNET_TOKENIZER`.

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

## Inspect a GGUF

```bash
cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
```

## Test with `1bitLLM/bitnet_b1_58-large`

The HF repo ships Safetensors; convert to GGUF with **Microsoft BitNet** tooling, then add the tokenizer and point `RBITNET_MODEL` at the `.gguf`. Walkthrough: **[docs/MODEL_TESTING.md](docs/MODEL_TESTING.md)**.

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

## Environment

| Variable | Meaning |
|----------|---------|
| `RBITNET_BIND` | Host:port (default `127.0.0.1:8080`) |
| `RBITNET_MODEL` | Path to `.gguf` for real inference |
| `RBITNET_TOKENIZER` | Path to `tokenizer.json` or `tokenizer.model` if not beside the GGUF |
| `RBITNET_STUB` | `1` = stub text (no inference) |
| `RBITNET_TOY` | `1` = tiny in-process F32 toy LM (no GGUF) |
| `RBITNET_TOY_SEED` | Seed for toy weights (default `42`) |
| `RBITNET_TEST_GGUF` | Optional path for `optional_gguf_from_env_smoke` test only |

## Benchmarks

```bash
cargo bench -p bitnet-core
```

## License

MIT — see [LICENSE](LICENSE).
