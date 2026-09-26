# Différenciation Future

Priorité indicative, de la plus proche du produit actuel vers les paris plus longs.

Alignement technique (serving, KV, speculative, BitNet, items **deferred** FA2/FA3 / DistServe / MoE / Medusa / AWQ) : [STATUS_AND_ROADMAP.md — Research-backed priorities](STATUS_AND_ROADMAP.md#research-backed-priorities-2026-09) et [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md).

1. **Akasha-first**: faire de Rbitnet le backend local de référence pour Akasha, avec profils de configuration prêts à l’emploi, métriques corrélables (TTFT, `prefix_hit`, `draft_accept`) et diagnostics communs.
2. **Fiabilité curatée**: maintenir un catalogue court mais vérifié, avec tokenizer, template, RAM minimale, **SHA256 / mode trusted**, commandes de reproduction et résultats bench publiés.
3. **Ops Rust sobres**: livrer un binaire autonome, peu de dépendances runtime, logs structurés, métriques Prometheus et packaging Windows/macOS/Linux reproductible.
4. **Mode local multi-tenant**: isoler plusieurs utilisateurs locaux par clé API, limites de concurrence et registres de modèles séparés, sans imposer Kubernetes — s’appuyer sur continuous batching + paged KV quand prêts.
5. **Runner proxy isolé**: passer à un processus enfant par modèle pour libérer VRAM/RAM proprement, survivre aux crashs natifs et se rapprocher de l’ergonomie Ollama (chemin multi-modèle documenté).
6. **Signature entreprise**: archives signées, SBOM, checksums automatisés et politique de provenance pour usage poste développeur en entreprise.
7. **Chemin NPU**: expérimenter DirectML / Windows NPU / ONNX-adjacent pour les machines grand public où le GPU discret n’est pas disponible.
8. **Évaluation fédérée**: permettre à plusieurs machines locales de publier des résultats anonymisés de compatibilité et performance vers une matrice communautaire.
9. **Templates de prompts sûrs**: fournir des templates audités par famille de modèles, avec tests dorés pour éviter les régressions silencieuses de format.
10. **Profils énergie/latence**: exposer des modes “batterie”, “latence”, “débit” (`interactive` / `batch` / `bitnet-cpu`) qui règlent max tokens, concurrence, mmap et cache préfixe de façon lisible.
11. **Interop OpenAI stricte**: compléter progressivement les endpoints clients attendus (`/v1/completions`, erreurs, SSE, modèles par défaut) avec tests de compatibilité.
12. **Distribution offline**: packs modèle + tokenizer + config + checksum pour déploiement sans accès Hub, adaptés aux environnements contraints.
13. **BitNet natif crédible**: kernels ternaires CPU (patterns I2_S/TL2 réimplémentés, pas FFI) + recipe `bitnet-b158` — différenciateur produit avant GPU packed.
