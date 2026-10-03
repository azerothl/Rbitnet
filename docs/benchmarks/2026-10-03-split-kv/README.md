# Attention CUDA par partitions K/V — 3 octobre 2026

Le nouveau chemin `RBITNET_CUDA_SPLIT_KV=1` réduit le coût de l'attention longue dans Llama résident et Qwen3.5 dense entièrement résident. Il est expérimental et désactivé par défaut. Les K/V restent en F32, tous les tokens causaux sont lus. Chaque partition de 256 positions produit un maximum, une somme et un numérateur ; la fusion remet ces états à la même échelle avant division. La composition suit le principe exact décrit dans [FlashInfer, §2.2 et §3.3](https://arxiv.org/html/2501.01005v1). Les erreurs d'arrondi FP32 subsistent. Ce code ne reprend pas le maximum heuristique unifié de FlashDecoding++.

Le scratch et la grille restent stables pendant les replays CUDA Graph. Llama réserve environ 264 Kio pour le décodage du modèle 1B et jusqu'à 33 Mio pour un bloc de 128 tokens ; Qwen 2B réserve environ 0,5 Mio par couche d'attention complète. Ces allocations s'ajoutent au KV actif et au budget des snapshots. Une ancienne DLL garde son ancien chemin ; le compteur `rbitnet_core_gpu_split_attention_queries_total` est alimenté par la capacité réellement interrogée sur le contexte natif.

## Ablations sur le même binaire et la même DLL

Ryzen 7 9800X3D, RTX 4080 SUPER 16 Gio, Windows, CUDA 13.3, pilote 610.88. Un cycle de chauffe puis deux cycles mesurés avec deux prompts longs et un contrôle court. Les tableaux excluent la chauffe et le contrôle. Chaque réponse longue comporte 128 tokens : 1 729 tokens d'entrée pour Llama3.2-1B Q4_K_M, 1 480 pour Qwen3.5-2B Q8_0. Aucun autre test GPU ni compilation pendant les mesures.

| Llama | Préremplissage médian (min–max), ms | Décodage médian, ms | tok/s décodage | HTTP médian, ms |
|---|---:|---:|---:|---:|
| Résident antérieur | 10 843 (10 769–10 920) | 1 489,5 | 85,9 | 12 337,0 |
| Attention partitionnée | 3 857 (3 841–3 893) | 339,5 | 377,0 | 4 204,7 |
| Attention + préfixe réutilisé | 3 (3–3) | 340 | 376,5 | 360,7 |
| Préremplissage matriciel antérieur | 2 023 (2 014–2 032) | 1 506,5 | 85,0 | 3 546,5 |
| Attention + préremplissage matriciel | 1 546,5 (1 535–1 558) | 342,5 | 373,7 | 1 901,6 |

| Qwen dense complet | Préremplissage médian (min–max), ms | Décodage médian, ms | tok/s décodage | HTTP médian, ms |
|---|---:|---:|---:|---:|
| Attention antérieure | 8 842 (8 829–8 877) | 1 269,5 | 100,8 | 10 122,1 |
| Attention partitionnée | 5 102,5 (5 099–5 164) | 549 | 233,2 | 5 667,6 |
| Attention + préfixe réutilisé | 4,5 (4–5) | 557 | 229,8 | 570,8 |

Les 45 réponses Llama et 27 réponses Qwen correspondent à leur référence respective, ainsi que 15 et 9 paires unary/SSE : glouton, sampling avec seed, pénalités et arrêts explicites. Les préfixes chauds mesurent la réutilisation d'une entrée déjà calculée, pas un nouveau prompt froid. Les mesures brutes et plages sont dans [llama-summary.json](llama-summary.json), [qwen-summary.json](qwen-summary.json) et les deux fichiers `*-ablation.json`.

Sur le contrôle court indépendant, trois mesures après chauffe, les 16 réponses sont identiques. Décodage médian Llama : 69 → 68 ms ; Qwen : 125 → 124 ms pour 32 tokens. Ces différences sont trop faibles pour revendiquer un gain court significatif. Aucun seuil de contexte supplémentaire n'a été introduit.

## Comparatif fraîchement remesuré contre les deux moteurs

Ollama 0.35.0 et llama.cpp b11351 / 631109b34, mêmes GGUF et prompts bruts tokenisés. Un échauffement et trois mesures, concurrence 1, température 0. Préfixes Rbitnet désactivés ; `cache_prompt=false` pour llama.cpp ; déchargement puis préchargement Ollama avant chaque requête chronométrée pour commencer sans KV réutilisé, avec modèle résident. Les fixtures `short/long-*-prompts.json` conservent les textes et IDs de tokens.

| Prompt / modèle / moteur | tok/s décodage | HTTP, ms | Préremplissage, ms | Premier événement SSE, ms |
|---|---:|---:|---:|---:|
| Court Llama / llama.cpp | 502,8 | 71,7 | 5,8 | 8,1 |
| Court Llama / Ollama | 484,6 | 119,9 | 44,9 | 69,5 |
| Court Llama / Rbitnet | 484,8 | 124,0 | 41,0 | 58,0 |
| Court Qwen / llama.cpp | 230,7 | 185,8 | 20,6 | 30,6 |
| Court Qwen / Ollama | 220,7 | 216,7 | 58,6 | 68,3 |
| Court Qwen / Rbitnet | 258,1 | 252,7 | 126,0 | 127,2 |
| Long Llama / llama.cpp | 417,7 | 366,2 | 43,9 | 65,0 |
| Long Llama / Ollama | 488,8 | 363,2 | 82,5 | 87,2 |
| Long Llama / Rbitnet | 381,0 | 1 873,7 | 1 516,0 | 1 549,3 |
| Long Qwen / llama.cpp | 221,6 | 681,2 | 86,0 | 100,7 |
| Long Qwen / Ollama | 227,7 | 688,4 | 118,0 | 149,6 |
| Long Qwen / Rbitnet | 232,3 | 5 709,2 | 5 139,0 | 5 170,3 |

Courts : 32 tokens de sortie ; longs : 128. Les sondes SSE utilisent 16 tokens et sont des requêtes distinctes des mesures HTTP. Qualité sur les trois sondes conservées : Llama 3/3, Qwen 2/3 dans les trois moteurs, avec la même erreur sur la mémoire ORION. Ce petit contrôle n'est pas une évaluation générale du modèle.

Le décodage Qwen dépasse les références dans ces cas, mais le préremplissage et la latence HTTP restent beaucoup plus lents. Llama court est proche d'Ollama et derrière llama.cpp ; Llama long reste plus lent en décodage aussi. **Pas de parité globale ni de supériorité générale.** Le prochain travail doit accélérer les projections matricielles et le préremplissage par blocs Qwen.

Les capacités de contexte diffèrent : 8 192 pour Rbitnet, 2 048 demandés aux références. Les longueurs réellement consommées sont identiques entre moteurs et restent sous ces limites. K/V F32 pour Rbitnet et llama.cpp, F16 par défaut pour Ollama. Le RSS est celui de l'arbre de processus. Les pics GPU incluent le bureau Windows et ne sont pas attribués à un processus. Les données de mémoire et chaque répétition sont publiées dans [comparison-short.json](comparison-short.json) et [comparison-long.json](comparison-long.json).

## Preuves de correction

- 237 tests workspace réussis, 0 échec, 1 ignoré, 25 suites ; Clippy tous targets passe avec les avertissements préexistants.
- Oracle indépendant FP64 : GQA, dimensions 32/64/80/256, blocs 1/7/16/128, positions autour des limites de tuiles, fenêtres glissantes, grands logits finis, replay d'un même graphe à des positions décroissantes et scratch ancien. Tolérance `3e-5 * (1 + abs(expected))`. Log : [oracle.log](oracle.log).
- Llama réel : logits de vérification par position, argmax, rollback, 43 IDs de référence, seed/pénalités/annulation et préfixes divergents/éviction. Qwen réel : 53 IDs, checkpoints complets, replay/eager ; oracle d'attention complète sur huit formats GGUF. Logs `llama-*` et `qwen-*`.
- HTTP/SSE réel : trois modes Llama/partiel Qwen et un mode full Qwen, chacun avec trois interruptions client puis reprises, stop unary/SSE et quatre requêtes identiques concurrentes. Le runtime sérialise encore ces requêtes : ceci ne prouve pas le batching GPU. Le chemin Qwen partiel conserve son ancien noyau et un compteur split égal à zéro. Rapports [streaming-llama.json](streaming-llama.json) et [streaming-qwen.json](streaming-qwen.json).
- Ancienne DLL sans nouveau symbole : séquence Llama réelle préservée, compteur de capacité absent traité comme zéro ; [legacy-dll.log](legacy-dll.log).

## Reproduction et empreintes

[manifest.json](manifest.json) relie les empreintes CLI, DLL, modèles/tokenizers et sources aux preuves. Tous les rapports du lot emploient les mêmes CLI et DLL. Les empreintes de sources couvrent les octets physiques du checkout Windows ; la base est `f173f12` avec ces changements locaux lors des mesures. [comparison-manifest.json](comparison-manifest.json) donne matériel, versions, chemins et options. Adapter les chemins aux GGUF locaux.

```powershell
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir "$PWD/target/split-kv/cuda"
cargo build --release -p rbitnet-cli
python scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-split-kv/comparison-manifest.json --model llama32-1b --backend gpu --binary target/release/rbitnet.exe --library target/split-kv/cuda/rbitnet_cuda_quant64.dll --cycles 3 --split-kv --output-dir target/repro/llama
python scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-split-kv/comparison-manifest.json --model qwen35-2b --backend gpu --binary target/release/rbitnet.exe --library target/split-kv/cuda/rbitnet_cuda_quant64.dll --cycles 3 --qwen-full --split-kv --output-dir target/repro/qwen
python scripts/benchmark_split_kv_short.py --config docs/benchmarks/2026-10-03-split-kv/comparison-manifest.json --binary target/release/rbitnet.exe --library target/split-kv/cuda/rbitnet_cuda_quant64.dll --output-dir target/repro/short
```

Le [driver du comparatif](comparison-driver.ps1) démarre un daemon Ollama isolé avec son répertoire de modèles dédié et l'arrête dans `finally`. Ses chemins/ports et ceux du manifeste doivent être adaptés. Ne pas chronométrer en parallèle des compilations ou tests GPU. Les flags sont lus à la création des contextes ; redémarrer le serveur entre configurations. Ce lot avance #95/#90/#91/#97 et conserve ouverts les travaux de Tensor Cores, KV paginé/compressé et batching.
