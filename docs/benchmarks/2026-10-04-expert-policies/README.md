# Politiques de cache d’experts — mesures NVIDIA

**Least-Stale reste expérimental : une réponse GLM a divergé de LRU dans un premier passage synchrone à 8 Gio.** Le passage complet suivant et les diagnostics supplémentaires passent, mais la cause de cette divergence demeure inconnue. Elle est conservée dans `negative/` et dans `manifest.json`. Ce lot ne clôture pas [#85](https://github.com/azerothl/Rbitnet/issues/85).

La politique par défaut reste LRU. Cette étude compare le placement des mêmes experts et des mêmes octets GGUF dans Rbitnet. Elle ne constitue pas une nouvelle comparaison avec Ollama ou llama.cpp.

CLI figé : `5d937ad9f035c20548e12bebf7a4121ff389a2d09f3d1034f706f8194cf91470`. DLL CUDA figée : `b86c8fa6d5cf34f500c01e860bae1f87fcb2411862d1ed2e0f68dc08f9411e38`. Les captures décrivent leurs chemins de mesure originaux ; `evidence-index.json` relie les fichiers publiés aux empreintes réellement observées.

## Protocole et portée

- Ryzen 7 9800X3D, 64 Gio de RAM, RTX 4080 SUPER 16 Gio, Windows ; aucune validation AMD, Intel ou Metal.
- GPT-OSS-20B Q4_K_M et GLM-4.7-Flash Q4_K_M ; budgets d’experts 512/8192 Mio, plafond CUDA géré 12288 Mio, marge 256 Mio, capacité de contexte 2048 tokens.
- Trois politiques, trois prompts : récit, code Python, capitale. Trois cycles, dont un de chauffe ; deux répétitions mesurées par prompt. Les tokens effectifs sont enregistrés dans chaque échantillon.
- Collecte des traces séparée des mesures sans traces. Les compteurs natifs correspondent exactement aux rejeux sur les douze captures ; les entrées canoniques des routeurs restent identiques entre politiques sur ces captures.
- Les huit captures complètes contiennent 216 réponses JSON, 72 comparaisons JSON/SSE et 24 arrêts explicites, en comptant séparément les passages avec et sans traces. Six suites réseau synchrones vérifient ensuite annulation/reprise, arrêt et quatre requêtes concurrentes exécutées en série par le runtime.
- Le harnais réseau initial réactivait les copies asynchrones avec une option héritée. Ces anciennes captures sont exclues de cette validation synchrone. Les nouvelles enregistrent les options effectives et exigent des jauges natives d’asynchronisme, de pool et de mémoire épinglée nulles.
- L’échec initial n’est pas inclus dans les médianes des passages complets : il reste une observation négative séparée, sans devenir une preuve de correction après une répétition réussie.

## Décodage mesuré

Médiane et plage des deux répétitions chaudes, par prompt. Le débit provient du temps de décodage. Les transferts et allocations couvrent toute la requête, prefill compris. Les résultats à deux répétitions ne permettent pas une conclusion statistique.

### gpt-oss-20b, cache 512 Mio

| Politique | Récit, tokens/s [min–max] | Code, tokens/s [min–max] | Transferts récit, Gio/requête | Allocations récit/requête |
|---|---:|---:|---:|---:|
| lru | 8.85 [8.83–8.86] | 8.76 [8.73–8.79] | 362.84 | 0 |
| lfu | 8.74 [8.52–8.96] | 8.96 [8.91–9.00] | 362.84 | 0 |
| least-stale | 8.99 [8.95–9.02] | 8.84 [8.72–8.96] | 362.84 | 0 |

### gpt-oss-20b, cache 8192 Mio

| Politique | Récit, tokens/s [min–max] | Code, tokens/s [min–max] | Transferts récit, Gio/requête | Allocations récit/requête |
|---|---:|---:|---:|---:|
| lru | 76.52 [74.51–78.53] | 73.11 [72.19–74.03] | 1.28 | 0 |
| lfu | 79.42 [78.19–80.66] | 37.82 [37.48–38.16] | 0.76 | 0 |
| least-stale | 49.73 [48.63–50.83] | 43.42 [42.85–43.99] | 32.49 | 0 |

### glm47-flash, cache 512 Mio

| Politique | Récit, tokens/s [min–max] | Code, tokens/s [min–max] | Transferts récit, Gio/requête | Allocations récit/requête |
|---|---:|---:|---:|---:|
| lru | 5.27 [5.26–5.27] | 5.29 [5.26–5.32] | 247.94 | 66294 |
| lfu | 5.24 [5.19–5.29] | 5.28 [5.26–5.31] | 247.94 | 66294 |
| least-stale | 5.30 [5.29–5.31] | 5.30 [5.29–5.31] | 247.94 | 66294 |

### glm47-flash, cache 8192 Mio

| Politique | Récit, tokens/s [min–max] | Code, tokens/s [min–max] | Transferts récit, Gio/requête | Allocations récit/requête |
|---|---:|---:|---:|---:|
| lru | 25.23 [24.68–25.78] | 22.71 [22.36–23.05] | 22.01 | 6111 |
| lfu | 12.39 [12.27–12.50] | 12.40 [12.39–12.41] | 47.68 | 14376 |
| least-stale | 19.02 [18.84–19.20] | 18.40 [18.26–18.55] | 80.95 | 0 |

## Diagnostic de la divergence GLM

Le premier écart de contenu est « Le robot s’assit » contre « Le robot s’approcha ». Les requêtes et le début de réponse concordent ; les métriques ne montrent pas de repli CPU dans la requête défaillante.

Une répétition ciblée de neuf requêtes puis un passage complet de 27 réponses/9 JSON-SSE/3 arrêts retrouvent la référence. Un diagnostic supplémentaire reconstruit exactement le prompt et la réponse HTTP, puis compare tous les logits bit à bit sur 1536 positions de génération, réparties entre trois politiques et quatre cycles. Les argmax GPU/CPU concordent. Ce diagnostic ajoute une lecture de la tête de sortie et change donc l’ordonnancement : il ne reproduit pas exactement le passage qui a échoué.

Compute Sanitizer ne signale aucune erreur mémoire ni conflit de mémoire partagée sur les fixtures d’attention MLA, qui vérifient aussi un oracle F64. Ce contrôle ciblé ne prouve pas l’absence de tout défaut du moteur.

LRU est conservée ; aucune promotion de Least-Stale ni revendication de correction de l’échec initial. À 8 Gio, LRU donne le meilleur débit GLM et le meilleur débit GPT sur le prompt code. Sur le récit GPT, LFU mesure 79,42 contre 76,52 tokens/s pour LRU ; deux répétitions ne permettent pas d’établir un gain robuste.

## Reproduire

Partir du code de #114/#116, compiler le CLI et la DLL CUDA avec les scripts du dépôt, puis adapter les chemins de modèles/tokenizers dans le manifeste en vérifiant leurs SHA-256. Les binaires ne sont pas distribués dans ce lot. Depuis la racine du dépôt :

```powershell
$policyCommon = @('--config', 'docs/benchmarks/2026-10-03-parity-round2/manifest.json', '--backend', 'gpu', '--binary', 'target/release/rbitnet.exe', '--library', 'target/paged-kv/cuda/rbitnet_cuda_quant64.dll', '--policies', '--cycles', '3', '--notes', '4', '--max-tokens', '128', '--device-mib', '12288')
python -B docs/benchmarks/2026-10-04-expert-policies/source/benchmark.py @policyCommon --model glm47-flash --mla-full --split-kv --moe-cache 8192 --trace-dir target/expert-policy-replay/traces --no-trace --output-dir target/expert-policy-replay/quiet
python -B docs/benchmarks/2026-10-04-expert-policies/source/live.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --binary target/release/rbitnet.exe --library target/paged-kv/cuda/rbitnet_cuda_quant64.dll --policy lru --mla-full --split-kv --moe-cache 512 --output-dir target/expert-policy-replay/live
python -B -m unittest discover -s docs/benchmarks/2026-10-04-expert-policies/source -p test_policy_network_options.py -v
python -B scripts/simulate_expert_cache.py docs/benchmarks/2026-10-04-expert-policies/traces/glm47-flash-cache8192/lru.jsonl.gz --budgets-mib 8192 --output target/expert-policy-replay/replay.json
```

Omettre `--no-trace` dans un passage séparé pour collecter les traces. Pour GPT-OSS, utiliser `--model gpt-oss-20b --gpt-full --gpt-segmented` à la place des options MLA. Les durées d’un passage instrumenté ne remplacent pas les mesures sans traces.

Les captures reprises antérieures ne disposaient pas d’empreinte individuelle de harnais. Leurs binaires, bibliothèques, protocoles complets et sorties sont contrôlés ; les sources actuelles et la correction réseau sont jointes. Les métriques de VRAM globale incluent les autres applications, tandis que le plafond CUDA géré décrit les allocations comptabilisées par Rbitnet.
