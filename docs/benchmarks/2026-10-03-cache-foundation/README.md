# Cache et préremplissage : mesures du 3 octobre 2026

Ce lot valide les snapshots Llama/Qwen, un premier préremplissage matriciel Llama et un cache d'experts expérimental. Il compare les options de Rbitnet sur le même binaire et la même DLL ; il ne constitue pas une nouvelle comparaison avec Ollama/llama.cpp. Le [benchmark précédent](../2026-10-03-parity-round2/README.md) reste distinct.

Machine : Ryzen 7 9800X3D, 8 cœurs/16 threads, RAM 64 Go, RTX 4080 SUPER 16 Go, Windows, pilote 610.88. Les [empreintes de sources et d'artefacts](manifest.json), les empreintes du binaire/DLL et les paramètres sont publiés avec les réponses brutes. Révision de départ : `75c1e38`, plus les changements de cette PR. Aucune compilation, simulation de traces ou autre charge de benchmark n'a tourné pendant les deux mesures finales.

## Longs prompts, 128 tokens de sortie

Une chauffe (cycle 0) puis deux cycles mesurés ; deux prompts partageant les mêmes notes et un troisième prompt court de contrôle. Les statistiques ci-dessous portent sur les quatre réponses longues mesurées par mode. Llama : 1 729 tokens d'entrée ; Qwen : 1 480. Contexte maximal Rbitnet : 8 192. Cache de préfixes : 256 MiB, huit entrées. Les requêtes, les réponses, les compteurs et la dispersion complète sont dans [llama.json](llama.json), [qwen.json](qwen.json) et [summary.json](summary.json).

| Modèle / mode | Préremplissage médian (min–max), ms | Décodage médian, ms | Décodage, tok/s | Réponse HTTP médiane, ms |
|---|---:|---:|---:|---:|
| Llama 3.2 1B Q4_K_M, référence résidente | 10 753,5 (10 751–10 770) | 1 485 | 86,2 | 12 256,7 |
| Llama, préfixe | 11 (11–12) | 1 483 | 86,3 | 1 510,9 |
| Llama, blocs de 128 tokens | 2 004 (1 999–2 013) | 1 490,5 | 85,9 | 3 504,5 |
| Llama, préfixe + blocs | 12 (12–12) | 1 486 | 86,1 | 1 513,8 |
| Qwen3.5 2B Q8_0, référence | 13 152 (13 139–13 561) | 1 767 | 72,4 | 14 923,6 |
| Qwen, checkpoints complets | 21 (21–24) | 1 663,5 | 76,9 | 1 701,6 |

Le préremplissage froid Llama est 5,37 fois plus rapide avec les blocs. Le préfixe chaud réduit ici la durée HTTP d'environ 8,1 fois pour Llama et 8,8 fois pour Qwen. Le bénéfice dépend de la longueur effectivement réutilisée. Le débit de décodage Llama ne progresse pas sur ces prompts ; les chiffres du précédent benchmark court ne sont pas comparables à ce contexte long. Les 54 réponses gloutonnes des six modes sont identiques à leur référence respective, et les 18 paires HTTP/SSE sont identiques en glouton, sampling avec seed et pénalités.

RSS maximal : environ 1,99 Go pour Llama, 4,63 Go pour Qwen sans cache et 4,79 Go avec checkpoints. Les pics GPU globaux observés sont 3 538–3 769 MiB pour Llama et 4 482–4 609 MiB pour Qwen. Ces valeurs incluent le bureau et les autres allocations du GPU : leur différence ne mesure pas exactement les octets physiques du cache. Les budgets des snapshots sont comptés séparément dans le runtime ; le budget des poids n'est pas un plafond de toute la VRAM.

## Correction, compatibilité et arrêts

- [Workspace CPU](workspace-after-sse-fix.log) : 226 tests passent, zéro échec, un ignoré ; 25 suites, dont les tests sans modèle et les tests optionnels qui sortent tôt sans leur variable d'activation.
- [Llama GPU](llama-final-validation.log) et [Qwen GPU](qwen-final-validation.log) : séquence gloutonne de référence, 18 comparaisons préfixe chaud/froid par architecture (greedy/seed/pénalités), puis annulation et reprise dans ces trois modes. Llama exécute aussi le préremplissage matriciel.
- [Oracles CUDA](core-cuda-validation.log) : huit formats quantifiés, GEMM de tailles 1/2/17 tokens, FFN d'experts avec biais/SiLU/OAI, changements de pointeurs, éviction/rechargement et copies pageable de 64 MiB. Ces tests nécessitent le GPU et leurs variables explicites.
- [CPU Llama](llama-cpu-validation.log), [CPU Qwen](qwen-cpu-validation.log) et anciennes DLL [Llama](llama-old-dll-validation.log)/[Qwen](qwen-old-dll-validation.log) : référence numérique et replis sans les nouvelles API. Le test de préfixes GPU Llama sort tôt en CPU/ancienne DLL ; Qwen CPU valide ses checkpoints hôtes.
- [Validation HTTP/SSE réelle](streaming.json) : neuf déconnexions suivies d'une reprise, trois arrêts explicites HTTP/SSE et trois groupes de quatre requêtes concurrentes, tous conformes aux réponses sérialisées. Le runtime reste sérialisé ; cette validation ne prouve pas un forward GPU multi-séquences.

La validation SSE a révélé un blocage lorsque le stop supprimait un delta vide et que le stream attendait sans réveil, ainsi qu'un stop qui pouvait fuir lorsqu'il traversait plusieurs deltas. Le matcher conserve maintenant le suffixe nécessaire entre deltas, et le poll se réveille après avoir consommé un événement sans contenu. Le [rapport partiel avant correction](streaming-before-fix.json) conserve trois cas de reprise réussis ; l'exécution suivante, stop `Paris`, a été interrompue après blocage et n'a pas de résultat complet. Le [rapport après correction](streaming.json) contient les quinze cas terminés. Les tests couvrent aussi les partitions UTF-8 et la parité unary/SSE du backend stub.

## MoE : placement encore expérimental

[moe-pilot.json](moe-pilot.json) conserve le pilote GPT-OSS/GLM : 64 réponses, cache désactivé ou 4 096 MiB, LRU/LFU/Least-Stale, budgets de poids identiques. Une compilation et des vérifications CPU ont chevauché ce pilote ; ses temps sont exploratoires. Le compteur global d'upload était alors doublé sur le chemin cache, contrairement au compteur spécifique du cache : ne pas utiliser ce compteur global pour une conclusion de bande passante.

Une des 64 réponses, le premier récit GLM en LRU, divergeait de la référence. La synchronisation des chargements pageable a ensuite été rendue explicite avant la consommation sur les streams privés. Une nouvelle validation avec cinq serveurs cache GLM démarrés à froid, deux prompts et deux cycles donne [24 réponses identiques](glm-cold.json), dont quatre de référence fixe. Cette validation apporte une preuve de correction après le changement de dépendance ; elle ne transforme pas le pilote en mesure finale. Sur ces récits, le cache reste plus lent que le placement fixe et demeure désactivé par défaut.

La [trace GPT LRU compressée](gpt-lru-trace.jsonl.gz) contient 46 032 événements de routage. Le [rejeu des premiers 30 048 événements](gpt-counter-replay.json) reproduit exactement [les compteurs matériels des huit requêtes correspondantes](gpt-counter-validation.json) : 103 926 hits, 16 266 misses, 15 942 évictions et 215 023 507 200 octets transférés. Le [rejeu des quatre budgets et trois politiques](gpt-lru-replay.json) est une simulation de placement, pas une prédiction de tok/s. Une politique avec davantage de hits n'est pas nécessairement plus rapide.

```powershell
python scripts/test_simulate_expert_cache.py
python scripts/simulate_expert_cache.py docs/benchmarks/2026-10-03-cache-foundation/gpt-lru-trace.jsonl.gz --budgets-mib 512 2048 4096 8192 --output target/replay.json
python scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --model llama32-1b --backend gpu --binary target/release/rbitnet.exe --library target/performance-cache/cuda-final/rbitnet_cuda_quant64.dll --output-dir target/cache-reproduction
```

Les [options et limites](../../PERFORMANCE_CACHE_STACK.md) détaillent les budgets, l'ABI optionnelle, les architectures et les étapes encore ouvertes. Ce lot ne termine pas le préchargement asynchrone, les pipelines entièrement GPU Qwen/MoE, le KV paginé/compressé, la persistance SSD, le batching continu ou le décodage spéculatif.
