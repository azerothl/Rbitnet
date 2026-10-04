# Pipeline Qwen entièrement résident — 3 octobre 2026

Le pipeline CUDA Qwen3.5 dense supprime les transferts d'activations entre les 18 blocs GDN et les six blocs d'attention complète du modèle testé. Les K/V, la convolution, le GDN, les FFN et la projection de sortie restent sur un flux GPU. Le mode glouton admissible télécharge seulement un ID ; l'échantillonnage et les pénalités gardent le sampler Rust et téléchargent les logits. Le mode reste expérimental, activé par `RBITNET_CUDA_QWEN_FULL=1`.

Machine : Ryzen 7 9800X3D, 64 Go de RAM, RTX 4080 SUPER 16 Go, Windows, pilote 610.88, CUDA 13.3, Rust 1.94.1. Modèle Qwen3.5-2B Q8_0 et tokenizer identiques au [tour précédent](../2026-10-03-parity-round2/README.md). Les empreintes et limites des artefacts testés sont publiées dans le manifeste.

## Ablation des prompts longs

Même binaire et même DLL pour les quatre modes, contexte maximal 8 192, 1 480 tokens d'entrée et 128 de sortie. Une chauffe puis deux cycles mesurés avec deux prompts longs et un contrôle court : quatre réponses longues mesurées par mode. Aucun test GPU, compilation ou simulation n'a tourné pendant les mesures. Cache : 256 MiB, huit entrées, checkpoints tous les 128 tokens et juste avant le dernier token du prompt.

| Mode | Préremplissage médian (min–max), ms | Décodage médian (min–max), ms | Décodage, tok/s | HTTP médian, ms |
|---|---:|---:|---:|---:|
| Ancien pipeline | 12 628,5 (12 527–12 661) | 1 617 (1 605–1 646) | 79,2 | 14 255,1 |
| Ancien pipeline + préfixe | 21,5 (21–22) | 1 659 (1 633–1 706) | 77,2 | 1 707,9 |
| Pipeline complet | 8 620,5 (8 595–8 829) | 1 236,5 (1 234–1 275) | 103,5 | 9 869,1 |
| Pipeline complet + préfixe chaud | 10 (10–10) | 1 266,5 (1 263–1 275) | 101,1 | 1 289,7 |

Le pipeline complet gagne environ 31 % en débit de décodage et réduit le préremplissage froid d'environ 32 %. Le préfixe chaud réduit encore le temps HTTP ; il ne remplace pas l'optimisation du décodage. Les 36 réponses sont identiques à l'ancien chemin et les 12 paires HTTP/SSE correspondent, en glouton, sampling avec seed et pénalités. Les quatre arrêts explicites retirent bien la chaîne d'arrêt.

Les compteurs de la première réponse longue mesurée passent de 1 067 376 640 octets H2D et 1 272 455 168 D2H à 13 170 972 H2D et 512 D2H. Le pipeline exécute 1 607 forwards : 1 480 entrées et 127 tokens générés réinjectés. Chaque forward transfère un embedding de 2 048 F32 et une position, puis synchronise explicitement une fois son flux. Il n'évalue pas à nouveau le dernier token de sortie. Avec le préfixe chaud, 128 forwards suffisent : 1 049 088 octets H2D et 512 D2H. Les copies D2D des snapshots restent distinctes de ces compteurs H2D/D2H.

RSS maximal : 4,63 Go sans pipeline complet, 4,79 avec préfixes hôte, 4,44 avec pipeline complet. Les pics GPU globaux sont 4 698–4 901 MiB selon le mode et incluent le bureau ; ils ne constituent pas une mesure exacte des allocations du runtime. Le budget des poids n'inclut pas tous les buffers, KV et snapshots.

## Comparatif court remesuré

Une chauffe, puis trois prompts de débit avec 32 tokens de sortie, les mêmes prompts/token IDs et le même GGUF que le tour précédent. Cache de préfixes désactivé. Ollama 0.35.0 utilise un daemon isolé et son modèle est rechargé avant chaque mesure ; llama.cpp b11351 (`631109b34`) utilise le KV F32. Ollama conserve son KV F16 par défaut. Contexte demandé aux références : 1 024 ; capacité Rbitnet : 8 192, comme le tour précédent. Les temps de phases proviennent des métriques propres à chaque moteur ; la mesure HTTP provient du client commun. Aucun gain de préremplissage matriciel n'est fourni dans ce lot.

| Moteur | Décodage médian (min–max), tok/s | Préremplissage médian, ms | HTTP médian, ms | Premier contenu SSE, ms | Contrôles de qualité |
|---|---:|---:|---:|---:|---:|
| Rbitnet, pipeline complet | 252,0 (252,0–254,0) | 125,0 | 269,1 | 141,2 | 2/3 |
| Ollama | 219,7 (216,2–223,7) | 51,9 | 219,8 | 68,3 | 2/3 |
| llama.cpp | 231,7 (230,7–232,3) | 20,3 | 178,0 | 29,1 | 2/3 |

Le débit de décodage Rbitnet dépasse ici Ollama d'environ 14,7 % et llama.cpp de 8,7 %. Le [tour précédent](../2026-10-03-parity-round2/README.md) mesurait 144,8 tok/s pour Rbitnet, mais cette comparaison temporelle n'est pas une ablation isolée ; l'ablation longue ci-dessus fournit le contrôle sur le même binaire. La latence totale et le premier contenu restent supérieurs aux références, et trois prompts courts n'établissent pas une supériorité générale. Chaque moteur échoue au même contrôle de mémoire ORION ; cette réponse du modèle n'est pas corrigée artificiellement. Le probe SSE est une mesure unique de 16 tokens, pas une distribution de TTFT.

RSS maximal : Rbitnet 4,43 Go, Ollama 1,13 Go, llama.cpp 2,68 Go. Pics GPU globaux : 4 557 / 4 238 / 4 313 MiB respectivement, incluant le bureau. Le benchmark final ne modifie pas le serveur Ollama de l'utilisateur.

## Validation et limites

- Oracle F64 des blocs d'attention complets : huit formats de poids (F32, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4), GQA, porte activée/désactivée, dimensions de têtes différentes de l'embedding, RoPE partiel/absent, séquences successives, restauration tronquée et graphes/eager. Le test utilise des couches synthétiques conditionnées et une tolérance `5e-5*(1+abs(reference))`, pas une promesse universelle de précision FP32.
- Les 53 IDs du modèle réel correspondent à la référence llama.cpp en graphe et eager, avec reset entre séquences. Dans chaque mode, 18 paires de réponses après divergence de conversation/cache correspondent à un runtime froid, y compris sampling avec seed et pénalités, puis annulation/reprise. Les normes Q/K et le layout intercalé query/gate sont ceux du GGUF.
- Une ancienne DLL reproduit les séquences par l'ancien chemin. Elle n'exécute pas les nouveaux tests de snapshots. Les erreurs pendant un forward complet sont remontées ; aucun repli vers un état CPU obsolète n'est tenté au milieu d'une séquence.
- L'utilisation réelle HTTP/SSE avec le pipeline complet et les checkpoints passe trois déconnexions/reprises (glouton, seed, pénalités), un arrêt HTTP/SSE et quatre requêtes concurrentes correctes. Le mutex du runtime sérialise encore les forwards : ce contrôle ne constitue pas un batch GPU fusionné.
- Le pipeline est limité aux modèles Qwen3.5 **denses**, backend `cuda`, poids entièrement résidents et contexte maximal 8 192. `hybrid`, MoE, debug partiel, tracing par couche ou symboles manquants conservent l'ancien chemin. `RBITNET_REQUIRE_QWEN_FULL=1` transforme ce repli en erreur de chargement.
- Les snapshots d'attention device sont associés aux états GDN/convolution et ne changent pas l'exigence de restauration à un checkpoint récurrent exact. Ils restent propres au runtime/modèle et ne persistent pas sur SSD.
- Le préremplissage reste token par token. L'attention utilise encore un tableau de scores par tête ; ce lot ne fournit ni préremplissage matriciel Qwen, ni KV device paginé/compressé, ni batch GPU multi-séquences.
- Le workspace avec backend CPU explicite passe 231 tests, zéro échec et un test ignoré, sur 25 suites. Certains tests GPU optionnels y rendent la main quand leur flag est absent ; les validations GPU ci-dessus ont été exécutées séparément avec leurs flags requis.
- `cargo clippy --workspace` passe avec les avertissements du dépôt conservés dans [le log](clippy.log). La CI GitHub actuelle cible les PR vers `main/master` ; cette PR empilée publie les validations locales, sans prétendre avoir une CI distante déclenchée sur sa branche de base.

Données : [ablation et réponses brutes](ablation.json), [résumé des mesures longues](summary.json), [comparatif court](short.json), [paramètres du comparatif](short-manifest.json), [HTTP/SSE et déconnexions](streaming.json), [empreintes](manifest.json), [oracle GPU](gpu-attention-oracle.log), [séquences avec graphe](gpu-sequence-graph.log), [séquences eager](gpu-sequence-eager.log), [ancienne DLL](legacy-dll.log), [workspace CPU](workspace-cpu.log).

## Reproduction

```powershell
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir target/qwen-full/cuda
cargo build --release -p rbitnet-cli
$env:RBITNET_CUDA_QUANT_LIB = (Resolve-Path target/qwen-full/cuda/rbitnet_cuda_quant64.dll).Path
$env:RBITNET_CUDA_QUANT_SMOKE = '1'
cargo test --release -p bitnet-core --lib native::qwen_full::tests -- --test-threads=1
pwsh -NoProfile -File scripts/validate_qwen_full.ps1
pwsh -NoProfile -File scripts/validate_qwen_full.ps1 -Eager
python scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --model qwen35-2b --binary target/release/rbitnet.exe --library target/qwen-full/cuda/rbitnet_cuda_quant64.dll --output-dir target/qwen-full/ablation --cycles 3 --qwen-full
python scripts/summarize_cache_benchmark.py target/qwen-full/ablation/results.json target/qwen-full/summary.json
python scripts/validate_cache_streaming.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --binary target/release/rbitnet.exe --library target/qwen-full/cuda/rbitnet_cuda_quant64.dll --output-dir target/qwen-full/streaming --qwen-full
```
