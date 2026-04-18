# Rbitnet

Pure Rust BitNet-style inference (work in progress) and an **OpenAI-compatible HTTP server** for [Akasha](https://github.com/loicpeaudecerf/Akasha) (`BitNetProvider`).

## Status

- **bitnet-core**: GGUF parse (metadata, tensor table, mmap’d weights), [`LlamaHyperParams`](crates/bitnet-core/src/gguf/bitnet_meta.rs), reference ternary `matvec`, optional **toy** LM, [`Engine`](crates/bitnet-core/src/inference.rs).
- **bitnet-server** (`rbitnet-server`): `GET /`, `GET /v1/models`, `POST /v1/chat/completions` (JSON + SSE). Integration tests for stub mode (`crates/bitnet-server/tests/openai_compat.rs`).
- **Docs (English)**:
  - [docs/BITNET_SPEC.md](docs/BITNET_SPEC.md) — format / metadata expectations
  - [docs/GOLDEN_TESTS.md](docs/GOLDEN_TESTS.md) — golden / regression testing
  - [docs/MODEL_TESTING.md](docs/MODEL_TESTING.md) — **testing with Hugging Face `bitnet_b1_58-large` and GGUF conversion**
- Full **BitNet-quantized** forward pass is still **WIP**; use **stub** or **toy** for HTTP checks without weights.

## Run the server (stub)

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

## Inspect a GGUF (e.g. after BitNet conversion)

```bash
cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
```

## Test with `1bitLLM/bitnet_b1_58-large`

The HF repo ships Safetensors; convert to GGUF with **Microsoft BitNet** (`setup_env.py` / helpers), then point `RBITNET_MODEL` at the `.gguf` file. Full walkthrough: **[docs/MODEL_TESTING.md](docs/MODEL_TESTING.md)**.

Optional automated check (local only):

```bash
export RBITNET_TEST_GGUF=/path/to/model.gguf
cargo test -p bitnet-core optional_gguf_from_env_smoke -- --nocapture
```

## Akasha

1. Run `rbitnet-server` (stub/toy is enough for connectivity).
2. In your data directory, edit `llm_router.yaml`:

```yaml
providers:
  bitnet:
    base_url: "http://127.0.0.1:8080"

task_types:
  conversation:
    primary:
      provider: bitnet
      model: rbitnet-stub
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
| `RBITNET_STUB` | `1` = stub text (no inference) |
| `RBITNET_TOY` | `1` = tiny in-process F32 toy LM (no GGUF) |
| `RBITNET_TOY_SEED` | Seed for toy weights (default `42`) |
| `RBITNET_MODEL` | Path to `.gguf` (parsed; generation returns **501** until inference is implemented) |
| `RBITNET_TEST_GGUF` | Optional path for `optional_gguf_from_env_smoke` test only |

## Benchmarks

```bash
cargo bench -p bitnet-core
```

## License

MIT — see [LICENSE](LICENSE).
