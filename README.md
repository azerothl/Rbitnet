# Rbitnet

Pure Rust implementation of BitNet-style inference (in progress) and an **OpenAI-compatible HTTP server** for use with [Akasha](https://github.com/loicpeaudecerf/Akasha) (`BitNetProvider`).

## Status

- **bitnet-core**: Full **GGUF** parse (metadata KV, tensor infos, mmap’d weight blob), [`LlamaHyperParams`](crates/bitnet-core/src/gguf/bitnet_meta.rs) from `llama.*` keys, reference ternary `matvec`, optional **toy** LM for end-to-end smoke tests, [`Engine`](crates/bitnet-core/src/inference.rs).
- **bitnet-server** (`rbitnet-server`): `GET /`, `GET /v1/models`, `POST /v1/chat/completions` (JSON + SSE). Integration tests cover stub mode (`tests/openai_compat.rs`).
- **Docs**: [docs/BITNET_SPEC.md](docs/BITNET_SPEC.md), [docs/GOLDEN_TESTS.md](docs/GOLDEN_TESTS.md).
- Full **BitNet-quantized** forward pass is still **WIP**; use **stub** or **toy** mode to validate HTTP + Akasha without weights.

## Run the server (stub)

```bash
set RBITNET_STUB=1
set RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server
```

Toy LM (no GGUF file):

```bash
set RBITNET_TOY=1
cargo run -p bitnet-server --bin rbitnet-server
```

## Akasha

1. Start `rbitnet-server` (stub or toy is enough for connectivity tests).
2. In your data directory, edit `llm_router.yaml` — add a `bitnet` provider and point tasks to it, for example:

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

(`model` must match an `id` from `GET /v1/models`, e.g. `rbitnet-stub`, `rbitnet-toy`, or `rbitnet-<architecture>` when a GGUF is loaded.)

See also the Akasha repository file `spec/llm_router.example.yaml` for the full template (including the `bitnet` provider block).

## Environment

| Variable | Meaning |
|----------|---------|
| `RBITNET_BIND` | Host:port (default `127.0.0.1:8080`) |
| `RBITNET_STUB` | `1` = stub text (no inference) |
| `RBITNET_TOY` | `1` = tiny in-process F32 toy LM (no GGUF) |
| `RBITNET_TOY_SEED` | Seed for toy weights (default `42`) |
| `RBITNET_MODEL` | Path to `.gguf` (parsed; full BitNet still returns `501` until implemented) |

## Benchmarks

```bash
cargo bench -p bitnet-core
```

## License

MIT — see [LICENSE](LICENSE).
