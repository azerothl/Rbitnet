# akasha-infer = Rbitnet

**Public :** Loïc Peaudecerf / Project akasha-infer  
**Repo code name :** [Rbitnet](https://github.com/azerothl/Rbitnet) (`rbitnet`, `rbitnet-server`, crates `bitnet-*`)

Ce dépôt **est** le moteur d’inférence locale du projet **akasha-infer**.  
Le monorepo assistant [Akasha](https://github.com/azerothl/Akasha) consomme Rbitnet via le provider `bitnet` (`BitNetProvider`) en HTTP OpenAI-compatible — pas via FFI llama.cpp.

## Contrat gelé (surface Akasha)

Akasha peut dépendre de cette surface **sans Python** au runtime :

| Méthode | Chemin | Notes |
|---------|--------|-------|
| `GET` | `/health` | Liveness ; pas d’API key |
| `GET` | `/ready` | Readiness modèle ; pas d’API key |
| `GET` | `/metrics` | Prometheus text ; séries listées dans [AKASHA_METRICS.md](AKASHA_METRICS.md) |
| `GET` | `/v1/models` | Catalogue OpenAI-style |
| `POST` | `/v1/chat/completions` | JSON + SSE (`stream: true`) |
| `POST` | `/v1/admin/reload` \| `/unload` | Optionnel ; nécessite `RBITNET_ADMIN_TOKEN` |

**Bind par défaut :** `http://127.0.0.1:8080`  
**Recipe Akasha BitNet :** `recipes/bitnet-b158.recipe.json` (après `rbitnet models install microsoft-bitnet-b1.58-2b-4t`).

Checklist CI / smoke : test `akasha_contract_metrics_series_present` dans `crates/bitnet-server/tests/openai_compat.rs` + script `scripts/smoke_openai.sh`.

## Ce que ce contrat n’inclut pas

- IPC CBOR / caps / daemon Akasha (port **3876**) — côté monorepo Akasha / akasha-os.
- Second engine (vLLM, Ollama) sur le chemin **défaut** — native-first ; voir [NATIVE_FIRST.md](NATIVE_FIRST.md).
- Fused GPU attention / FlashInfer — hors chemin défaut ; voir [GPU_NATIVE_ROADMAP.md](GPU_NATIVE_ROADMAP.md).

## Docs liées

- État & inspirations (survey) : Project store `docs/etat-et-inspiration-inference.md`
- Métriques corrélées : [AKASHA_METRICS.md](AKASHA_METRICS.md)
- Multi-modèle process-per-model : [RUNNER_PROXY_SPEC.md](RUNNER_PROXY_SPEC.md)
- Bench vs llama.cpp : [BENCHMARKS.md](BENCHMARKS.md) + [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md)
