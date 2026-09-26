# Akasha ↔ Rbitnet metrics correlation

Rbitnet exposes Prometheus text at **`GET /metrics`** (default bind `127.0.0.1:8080`). The Akasha daemon uses port **3876** with its own task and memory metrics.

**Frozen contract:** series in the tables below are the stable Akasha scrape surface for the `bitnet` provider. Renames require a coordinated Akasha change. See [AKASHA_INFER.md](AKASHA_INFER.md).

## Stable HTTP / inference counters (`bitnet-server`)

| Rbitnet series | Meaning | Akasha use |
|----------------|---------|------------|
| `rbitnet_chat_requests_total` | Chat completion requests | Request volume |
| `rbitnet_chat_errors_total` | Handler 4xx/5xx | Error rate |
| `rbitnet_inference_timeouts_total` | Wall-clock 504 path | SLO timeouts |
| `rbitnet_inference_ms_sum` | Sum of end-to-end inference ms | Latency rollup |
| `rbitnet_inference_calls_total` | Completed inference calls | Denominator for averages |
| `rbitnet_inference_ttft_ms_sum` | Sum of time-to-first-token (encode+prefill) ms | **TTFT** |
| `rbitnet_inference_encode_ms_sum` | Tokenizer encode ms | Phase split |
| `rbitnet_inference_prefill_ms_sum` | Prefill ms | Phase split |
| `rbitnet_inference_decode_ms_sum` | Decode ms | Phase split |
| `rbitnet_inference_itl_us_sum` | Sum of per-request avg inter-token µs | Decode smoothness |
| `rbitnet_inference_tpot_us_sum` | Sum of per-request TPOT µs | Decode tok/s helper |
| `rbitnet_completion_tokens_total` | Generated tokens | Throughput |
| `rbitnet_inference_decode_tokens_per_sec` | Derived: completion_tokens / (decode_ms/1000) when decode_ms>0 | Live decode tok/s |
| `rbitnet_inference_ttft_ms_avg` | Derived: ttft_sum / calls when calls>0 | Live TTFT average |
| `rbitnet_speculative_requests_total` | Speculative path requests | Spec usage |
| `rbitnet_unauthorized_total` | Bad/missing API key | Auth failures |
| `rbitnet_model_reloads_total` / `_failures_total` / `_ms_sum` | Admin reload | Hot reload health |
| `rbitnet_model_unloads_total` | Idle/admin unload | Memory reclaim |
| `rbitnet_inference_calls_by_backend_family_total{backend,family}` | Calls by backend×family | Routing labels |

## Stable core labels / serving hooks (`bitnet-core`)

| Rbitnet series | Meaning | Akasha analogue |
|----------------|---------|-----------------|
| `rbitnet_core_prefix_cache_hits_total` | Prefix KV / radix hits | Prompt reuse |
| `rbitnet_core_prefix_hit` | Same counter as hits (Akasha **`prefix_hit`** alias) | **`prefix_hit`** |
| `rbitnet_core_prefix_cache_misses_total` | Prefix KV misses | Cache miss rate |
| `rbitnet_core_prefix_cache_bytes_saved_total` | Estimated KV bytes skipped | Cache efficiency |
| `rbitnet_core_speculative_draft_tokens_total` | Draft tokens proposed | Spec draft |
| `rbitnet_core_speculative_verified_tokens_total` | Tokens verified after draft | Spec verify |
| `rbitnet_core_speculative_accepted_tokens_total` | Draft tokens accepted | Spec accept |
| `rbitnet_core_draft_accept` | Same as accepted tokens (Akasha **`draft_accept`**) | **`draft_accept`** |
| `rbitnet_core_scheduler_batches_total` | Scheduler batch waves | Concurrent packing |
| `rbitnet_core_scheduler_batch_items_total` | Items in batch waves | Batch depth |
| `rbitnet_core_scheduler_decode_waves_total` | Continuous-batching decode waves | Multi-seq decode |
| `rbitnet_core_kv_physical_pages` | Current physical KV pages | Memory pressure |
| `rbitnet_core_kv_pool_active_seqs` | Active sequences in `RBITNET_KV_POOL` | Pool occupancy |
| `rbitnet_core_kv_pool_allocated_pages` | Shared-pool physical pages (incl. free-listed) | KV RSS proxy |
| `rbitnet_core_kv_pool_free_pages` | Pages on the shared free list | Reclaim headroom |
| `rbitnet_core_kv_pool_fragmentation_permille` | Free/allocated ×1000 | Fragmentation |
| `rbitnet_core_kv_quant_format_code` | 0=f32, 1=q8, 2=q4 | KV quant mode |
| `rbitnet_core_model_load_ms_total` / `_loads_total` | Load timing | Startup / reload |
| `rbitnet_core_cuda_graph_replays_total` | Graphed decode steps | N/A (Rbitnet-specific) |

Derived gauges (`*_avg`, `*_tokens_per_sec`) are computed at scrape time from the sums above; they are part of the frozen contract so Akasha UI can show live TTFT / tok/s without extra series math.

## Ports and routing

- **Akasha daemon:** `http://127.0.0.1:3876` — `POST /api/message`, `GET /api/tasks/:id`
- **Rbitnet OpenAI API:** `http://127.0.0.1:8080` — `POST /v1/chat/completions`

Configure Akasha `llm_router.yaml` with an OpenAI-compatible endpoint pointing at Rbitnet when using BitNet or local GGUF backends.

## Recipes (SSOT)

- Catalog install writes `rbitnet.manifest.json` (optional `recipe` field).
- Serve presets: `rbitnet recipe recipes/bitnet-b158.recipe.json` or `rbitnet up MODEL --recipe recipes/example-qwen3.recipe.json`
- Tune presets: `rbitnet tune interactive` | `latency` | `battery` | `throughput` | `bitnet-cpu`

See [BITNET_NATIVE.md](BITNET_NATIVE.md) and the Akasha [reference-products-parity-matrix](https://github.com/azerothl/Akasha/blob/main/spec/dev/roadmap/reference-products-parity-matrix.md) (Perf / Rbitnet row).
