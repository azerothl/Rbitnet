# Différenciation Future

Priorité indicative, de la plus proche du produit actuel vers les paris plus longs.

1. **Akasha-first**: faire de Rbitnet le backend local de référence pour Akasha, avec profils de configuration prêts à l’emploi, métriques corrélables et diagnostics communs.
2. **Fiabilité curatée**: maintenir un catalogue court mais vérifié, avec tokenizer, template, RAM minimale, commandes de reproduction et résultats bench publiés.
3. **Ops Rust sobres**: livrer un binaire autonome, peu de dépendances runtime, logs structurés, métriques Prometheus et packaging Windows/macOS/Linux reproductible.
4. **Mode local multi-tenant**: isoler plusieurs utilisateurs locaux par clé API, limites de concurrence et registres de modèles séparés, sans imposer Kubernetes.
5. **Runner proxy isolé**: passer à un processus enfant par modèle pour libérer VRAM/RAM proprement, survivre aux crashs natifs et se rapprocher de l’ergonomie Ollama.
6. **Signature entreprise**: archives signées, SBOM, checksums automatisés et politique de provenance pour usage poste développeur en entreprise.
7. **Chemin NPU**: expérimenter DirectML / Windows NPU / ONNX-adjacent pour les machines grand public où le GPU discret n’est pas disponible.
8. **Évaluation fédérée**: permettre à plusieurs machines locales de publier des résultats anonymisés de compatibilité et performance vers une matrice communautaire.
9. **Templates de prompts sûrs**: fournir des templates audités par famille de modèles, avec tests dorés pour éviter les régressions silencieuses de format.
10. **Profils énergie/latence**: exposer des modes “batterie”, “latence”, “débit” qui règlent max tokens, concurrence, mmap et cache préfixe de façon lisible.
11. **Interop OpenAI stricte**: compléter progressivement les endpoints clients attendus (`/v1/completions`, erreurs, SSE, modèles par défaut) avec tests de compatibilité.
12. **Distribution offline**: packs modèle + tokenizer + config + checksum pour déploiement sans accès Hub, adaptés aux environnements contraints.
