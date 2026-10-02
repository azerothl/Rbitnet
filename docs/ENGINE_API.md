# Engine embed API (`bitnet-core`) — semver surface

**Public:** Project akasha-infer / Rbitnet  
**Status:** Frozen **minimal** embed surface for in-process use.  
**Default Akasha contract remains HTTP** (`/v1` + `/metrics`) — see [AKASHA_INFER.md](AKASHA_INFER.md). Embed is an **optional** alternative for single-process latency; it does not replace `BitNetProvider` HTTP.

## Semver policy

| Change | Version |
|--------|---------|
| Rename/remove methods listed below, change `complete_*` return shapes, or alter env semantics relied on by the table | **MAJOR** |
| New optional methods / env flags with defaults | MINOR |
| Docs, examples, bugfixes without API break | PATCH |

Workspace version: root `Cargo.toml` `[workspace.package].version`. Treat `bitnet-core` as the publishable crate (path / private registry OK; crates.io when ready).

## Minimal frozen surface

| Item | Role |
|------|------|
| `Engine::from_env()` | Load via `RBITNET_*` (same as server) |
| `Engine::load_path` / `load_path_with_overrides` | Explicit GGUF (+ tokenizer / arch) |
| `stub_engine()` / `RBITNET_STUB=1` | HTTP-less smoke without weights |
| `Engine::is_ready` / `has_gguf` / `openai_model_id` | Readiness / id |
| `Engine::complete` / `complete_detailed` / `complete_detailed_with_options` | Sync generate |
| `Engine::complete_streaming` | Token/stream callbacks (`StreamEvent`) |
| `Engine::complete_batch_detailed` | Continuous-batching batch path |
| `BitNetError` / `Result` | Errors |
| `SamplingOptions` | Sampling knobs |
| `perf::prometheus_text` (optional) | Metrics hooks for embedders |

Anything else in `bitnet-core` (kernels, llama internals, loaders) is **unstable** unless promoted here.

## Smoke embed (CI-friendly)

```bash
# No GGUF required:
RBITNET_STUB=1 cargo run -p bitnet-core --example engine_smoke --locked
```

Example source: [`crates/bitnet-core/examples/engine_smoke.rs`](../crates/bitnet-core/examples/engine_smoke.rs).

With a real model:

```bash
export RBITNET_MODEL=/path/model.gguf
export RBITNET_TOKENIZER=/path/tokenizer.json
export RBITNET_SMOKE_MAX_TOKENS=1
cargo run -p bitnet-core --example engine_smoke --release --locked
```

## Features

| Cargo feature | Stability |
|---------------|-----------|
| (default) | Stable embed path |
| `experimental-flashinfer` / `experimental-ggml-kernels` / `blas-llama` | **Experimental** — not part of the frozen embed contract |

## Clarifications

- **HTTP is still the Akasha default** for doctor / SSE / metrics scrape.
- Native-first: embed uses the same Rust GGUF path; no FFI llama.cpp.
- Breaking HTTP and breaking Engine are independent MAJOR decisions (document both in CHANGELOG).
