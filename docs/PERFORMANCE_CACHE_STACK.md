# Caches et préremplissage CUDA

Les optimisations de l'[épique #98](https://github.com/azerothl/Rbitnet/issues/98) sont activées séparément. Les poids quantifiés et les experts sélectionnés par le routeur restent ceux du GGUF. Le placement fixe et les chemins CPU restent les références de comparaison.

## Options expérimentales

| Option | Effet et limites |
|---|---|
| `RBITNET_PREFIX_KV=1` | Llama conserve son graphe CUDA résident et réutilise les K/V du préfixe commun. Qwen3.5 conserve des checkpoints complets : K/V des blocs d'attention, état GDN et historique de convolution. |
| `RBITNET_CUDA_PREFIX_MB=256` | Budget des snapshots par runtime, en MiB. Il inclut les états CPU et GPU Qwen. Les poids et le KV actif ont leurs propres allocations. |
| `RBITNET_CUDA_PREFIX_ENTRIES=16` | Nombre maximal de snapshots ; éviction LRU avant allocation. |
| `RBITNET_PREFIX_KV_MIN_TOKENS=8` | Longueur minimale réutilisable. Llama recalcule toujours le dernier token pour obtenir ses logits. |
| `RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS=256` | Intervalle des checkpoints Qwen pendant le préremplissage. L'état récurrent ne peut pas être tronqué à un préfixe arbitraire. |
| `RBITNET_MOE_CACHE_MB=4096` | Cache GPU par expert pour GPT-OSS et GLM/deepseek2. `0` garde le placement fixe. Gate/up/down sont chargés ensemble, sans conversion. |
| `RBITNET_MOE_CACHE_POLICY=lru` | Politique `lru`, `lfu` ou `least-stale`. LRU reste la référence ; les autres politiques peuvent être plus lentes. |
| `RBITNET_MOE_TRACE_DIR=...` | Trace JSONL du routage réel. Désactiver pour mesurer les performances sans coût de journalisation. |
| `RBITNET_CUDA_PREFILL=1` | Préremplissage Llama par blocs : GEMM quantifié SIMT, attention causale, résidus et FFN sur GPU. Désactivé par défaut. |
| `RBITNET_CUDA_PREFILL_TOKENS=128` | Taille du bloc, bornée entre 1 et 128. Les projections partagent les tuiles de poids décodées entre tokens ; les poids complets ne sont pas déquantifiés en F32. |

Le budget des experts est plafonné par `RBITNET_HYBRID_MAX_VRAM_MB` moins les poids non experts déjà résidents. Ce plafond ne compte pas le KV actif, les snapshots, les normes/biais et le scratch natif : il ne représente pas la consommation totale de VRAM. Les matrices d'experts gardent aussi un miroir RAM de leurs octets quantifiés.

Les experts utilisés sont protégés par des leases jusqu'à la fin du FFN. Une table d'adresses device stable permet aux graphes de lire les allocations actuelles après éviction ou réutilisation d'un slot. Si une sélection ne tient pas dans le budget, le FFN entier passe sur CPU, dans l'ordre du routeur. Un cache trop petit peut multiplier les transferts et ralentir la génération.

Les snapshots appartiennent au runtime : poids, tokenizer, configuration RoPE et layout ne changent pas pendant sa durée de vie. Un autre chargement de modèle crée un autre cache. Ils ne sont pas persistés sur disque. Une ancienne DLL sans API de snapshot garde le repli hôte Llama ou désactive la réutilisation Qwen ; une DLL sans GEMM garde le préremplissage résident token par token.

## Mesurer et reproduire

Les [mesures et preuves du 3 octobre 2026](benchmarks/2026-10-03-cache-foundation/README.md) publient les réponses brutes, les tests GPU/CPU, les limites du pilote MoE et les statistiques des ablations finales.

`scripts/benchmark_cache_stack.py` compare les modes sur le même binaire et la même DLL. Il conserve requêtes/réponses, chauffe, répétitions, temps de préremplissage/décodage, latence HTTP, compteurs, empreintes et mémoire. Il vérifie les réponses SSE en glouton, sampling avec seed et pénalités, ainsi qu'un arrêt explicite. Les séquences de référence et les tests de snapshots vérifient séparément l'annulation puis la reprise.

```powershell
python scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --model llama32-1b --backend gpu --binary target/release/rbitnet.exe --library target/performance-cache/cuda-final/rbitnet_cuda_quant64.dll --output-dir target/performance-cache/llama-http
python scripts/simulate_expert_cache.py chemin/trace.jsonl --budgets-mib 512 2048 4096 8192 --output target/replay.json
python scripts/test_simulate_expert_cache.py
```

Le simulateur utilise les mêmes experts sélectionnés pour toutes les politiques et protège les groupes déjà acquis dans la couche. Il simule le placement et les octets transférés ; il ne prédit pas les tok/s ni le recouvrement CPU/GPU. `--events N` permet de vérifier un segment contre les compteurs d'une exécution réelle.

## Correctness des transferts

Les graphes natifs utilisent des flux CUDA privés non bloquants. Une copie H2D depuis la RAM pageable peut rendre la main après staging, avant la fin du DMA, selon la [sémantique CUDA](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html). Le runtime attend donc la fin des chargements et des refills avant de publier leurs pointeurs. Les normes et biais natifs sont chargés et synchronisés sur le flux de leur contexte. Cela fixe la dépendance de transfert ; le préchargement asynchrone devra ensuite remplacer cette attente par des événements explicites.

## Étapes suivantes

Le cache à la demande ne termine pas [#84](https://github.com/azerothl/Rbitnet/issues/84) et [#86](https://github.com/azerothl/Rbitnet/issues/86) : il manque le préchargement épinglé avec recouvrement mesuré et le choix CPU/GPU par coût. Les pipelines complets Qwen/GPT-OSS/GLM, le KV device paginé/compressé, la persistance des sessions et les forwards multi-séquences restent dans leurs tickets [#87–#96](https://github.com/azerothl/Rbitnet/issues/98).

Le GEMM Llama retourne actuellement les logits du dernier token d'un bloc. La vérification matricielle de tous les tokens proposés, la correction et le rollback de [#97](https://github.com/azerothl/Rbitnet/issues/97) restent à implémenter avant de revendiquer un décodage spéculatif accéléré.
