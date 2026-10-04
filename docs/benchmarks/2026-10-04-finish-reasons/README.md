# Raisons d’arrêt réelles — EOS, budget et stop explicite

Les API indiquaient systématiquement `stop` / `end_turn`, même quand la génération atteignait son budget. Le runtime transmet désormais la cause observée jusqu’aux réponses OpenAI et Anthropic, complètes et SSE. Une observation EOS donne `stop` / `end_turn` ; un budget épuisé donne `length` / `max_tokens` ; les executors historiques qui ne fournissent pas cette observation donnent `null`.

Par exemple, sur Qwen3.5, une réponse visible d’un token renvoie `length` avec un budget d’un token et `stop` avec un budget de deux tokens, lorsque le second échantillonnage observe EOS. La sortie visible reste identique. Le budget zéro renvoie `length` avec zéro token, sans inventer un EOS.

Un stop OpenAI fourni par le client demeure distinct du budget. Les queues terminent aussi une entrée dont EOS ne produit aucun token visible ; un pas sans progression ni cause d’arrêt explicite devient une erreur plutôt qu’une nouvelle itération indéfinie. Ces corrections ne constituent pas une optimisation du débit GPU.

## Validation

- 296 tests workspace passent, un test réseau Hugging Face est ignoré. Clippy et la compilation release passent avec les avertissements existants sur tokenizers et les méthodes de pages encore inutilisées.
- Deux tests unitaires ciblés distinguent EOS spéculatif, budget et statistiques historiques ; neuf tests de scheduling couvrent aussi EOS sans sortie et terminaison de burst.
- Une fixture GGUF Llama réellement exécutée sur CPU couvre 42 cas HTTP/SSE OpenAI et Anthropic, y compris les arrêts explicites OpenAI et le budget zéro. Cette fixture synthétique vérifie le protocole, pas la qualité d’un grand modèle.
- Quatre GGUF réels sont ensuite exécutés sur cette RTX 4080 SUPER, avec leurs templates propres : 18 comparaisons JSON/SSE aux frontières EOS/budget, quatre comparaisons d’arrêt client et quatre recherches initiales d’EOS. Les requêtes/réponses exactes et les options effectives sont jointes.

| Modèle réel | Tokens visibles avant EOS | Frontières JSON/SSE |
|---|---:|---:|
| llama32-1b | 2 | 5 |
| qwen35-2b | 1 | 4 |
| gpt-oss-20b | 16 | 5 |
| glm47-flash | 1 | 4 |

CLI de preuve : `3fe8f07fb4c4ac3287221717e9ab434844d77278436f734542956d2b0661b87f`. DLL CUDA : `b86c8fa6d5cf34f500c01e860bae1f87fcb2411862d1ed2e0f68dc08f9411e38`. Les 179 sources Rust de la copie compilée sont contrôlées ; les 15 fichiers d’intégration livrés reprennent exactement les octets validés.

Le premier passage workspace avait omis le dossier `tests/data/golden` de la copie isolée. Son échec est conservé dans `diagnostics/first-workspace-missing-fixture.log`. La copie a été corrigée, puis le workspace entier a été relancé avec succès. Les journaux décrivent les chemins de mesure originaux.

## Reproduction

Compiler cette branche, adapter les chemins de modèles/tokenizers dans `docs/benchmarks/2026-10-03-parity-round2/manifest.json`, puis exécuter depuis la racine :

```powershell
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets
cargo build --release -p rbitnet-cli --bin rbitnet
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir target/paged-kv/cuda
python -B docs/benchmarks/2026-10-04-finish-reasons/source/finish_reason_live.py --binary target/release/rbitnet.exe --library target/paged-kv/cuda/rbitnet_cuda_quant64.dll --output target/finish-reason-live
```

Le harnais impose le chemin Qwen dense complet, GPT à banques fixes et MLA résident pour ces exports. Il enregistre les options transmises et ne lance pas les moteurs de référence. Cette validation ne remplace pas le comparatif final ni les travaux de batching continu, de décodage spéculatif ou de grammar JSON de [#24](https://github.com/azerothl/Rbitnet/issues/24) / [#98](https://github.com/azerothl/Rbitnet/issues/98).
