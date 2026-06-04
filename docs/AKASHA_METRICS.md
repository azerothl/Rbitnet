# Akasha ↔ Rbitnet metrics correlation

Rbitnet exposes Prometheus text at **`GET /metrics`** (default bind `127.0.0.1:8080`). The Akasha daemon uses port **3876** with its own task and memory metrics.

## Stable labels

| Rbitnet series | Meaning | Akasha analogue |
|----------------|---------|-----------------|
| `rbitnet_core_model_family` | `bitnet`, `llama`, `qwen35`, … | `llm_router` model id / provider |
| `rbitnet_core_backend` | `cpu`, `cuda`, `hybrid` | Provider backend selection |
| `rbitnet_core_prefix_cache_hits_total` | Prefix KV / radix hits | Deep-research prompt reuse (conceptual) |
| `rbitnet_core_scheduler_batch_size` | Batch packing | Daemon concurrent tasks |
| `rbitnet_core_cuda_graph_replays_total` | Graphed decode steps | N/A (Rbitnet-specific) |

## Ports and routing

- **Akasha daemon:** `http://127.0.0.1:3876` — `POST /api/message`, `GET /api/tasks/:id`
- **Rbitnet OpenAI API:** `http://127.0.0.1:8080` — `POST /v1/chat/completions`

Configure Akasha `llm_router.yaml` with an OpenAI-compatible endpoint pointing at Rbitnet when using BitNet or local GGUF backends.

## Recipes (SSOT)

- Catalog install writes `rbitnet.manifest.json` (optional `recipe` field).
- Serve presets: `rbitnet recipe recipes/bitnet-b158.recipe.json` or `rbitnet up MODEL --recipe recipes/example-qwen3.recipe.json`
- Tune presets: `rbitnet tune latency` | `battery` | `throughput`

See [BITNET_NATIVE.md](BITNET_NATIVE.md) and the Akasha [reference-products-parity-matrix](https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/reference-products-parity-matrix.md) (Perf / Rbitnet row).
