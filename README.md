# Rbitnet

Pure Rust implementation of BitNet-style inference (in progress) and an **OpenAI-compatible HTTP server** for use with [Akasha](https://github.com/loicpeaudecerf/Akasha) (`BitNetProvider`).

## Status

- **bitnet-core**: Full **GGUF** parse (metadata KV, tensor infos, mmap’d weight blob), reference ternary `matvec` kernels, [`Engine`](crates/bitnet-core/src/inference.rs) façade. Tensor `ggml_type` is kept as a raw `u32` so exotic quantizations still load.
- **bitnet-server** (`rbitnet-server`): `GET /`, `GET /v1/models`, `POST /v1/chat/completions` (JSON + SSE streaming). With `RBITNET_MODEL`, `/v1/models` returns `rbitnet-<architecture>` when `general.architecture` is present.
- Full BitNet forward pass and packed-weight kernels are **not** implemented yet; use **stub mode** to test the HTTP stack and Akasha integration without weights.

## Run the server (stub)

```bash
set RBITNET_STUB=1
set RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server
```

On Unix:

```bash
RBITNET_STUB=1 RBITNET_BIND=127.0.0.1:8080 cargo run -p bitnet-server --bin rbitnet-server
```

## Akasha

Point `llm_router.yaml` at `http://127.0.0.1:8080` for the `bitnet` provider (default base URL matches).

## Environment

| Variable | Meaning |
|----------|---------|
| `RBITNET_BIND` | Host:port (default `127.0.0.1:8080`) |
| `RBITNET_STUB` | Set to `1` for stub completions (no GGUF required) |
| `RBITNET_MODEL` | Path to a `.gguf` file (full structure parsed; generation returns `501` until the engine is wired) |

## License

MIT — see [LICENSE](LICENSE).
