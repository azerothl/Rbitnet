# Profiling snapshot — 2026-06-04 (serving hooks baseline)

**Purpose:** Document TTFT/TPOT measurement context after streaming, prefix-KV, and scheduler hook landings (Sprint 0).

## Environment (fill on your machine)

| Field | Value |
|-------|-------|
| Git SHA | *(run `git rev-parse --short HEAD`)* |
| `rustc -V` | *(local)* |
| OS / CPU | Windows 10 / *(your CPU)* |
| Backend | `cpu` stub: `RBITNET_STUB=1` |
| Model | `rbitnet-stub` or toy: `RBITNET_TOY=1` |

## Procedure

1. Start server: `RBITNET_STUB=1 cargo run -p bitnet-server --bin rbitnet-server --release`
2. Warmup: 3× `POST /v1/chat/completions` (fixed JSON, `max_tokens=32`)
3. Record p50/p95 wall time and `rbitnet_core_prefix_cache_hits_total` from `GET /metrics`
4. Repeat with `RBITNET_PREFIX_KV=1` and identical system prompt on requests 2–N

## Stub baseline (placeholder — refresh per release)

| Metric | Stub `RBITNET_STUB=1` | Notes |
|--------|----------------------|-------|
| TTFT p50 | ~5–15 ms | In-process; no GGUF |
| TPOT p50 | ~0 ms | Single-shot stub text |
| `prefix_cache_hits` | 0 → N | With `RBITNET_PREFIX_KV` + shared prefix on Llama path only |

## Next measurements

- One **frozen** GGUF CPU row in [BENCHMARKS.md](../BENCHMARKS.md) (see Sprint 0 table).
- After CUDA graph device capture: compare `rbitnet_core_cuda_graph_replays_total` vs eager steps.
