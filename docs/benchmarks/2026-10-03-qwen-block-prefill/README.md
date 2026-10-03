# Préremplissage Qwen3.5 dense par blocs — 2026-10-03

Sur RTX 4080 SUPER, le préremplissage long passe de 5 262 à 808,5 ms, soit environ 6,5 fois plus rapide, avec le même GGUF Q8_0. Le décodage reste similaire. Le comparatif confirme un retard de latence sur les prompts longs ; le mode reste opt-in.

## Implémentation

`RBITNET_CUDA_QWEN_FULL=1` et `RBITNET_CUDA_QWEN_PREFILL=1` activent les blocs causaux jusqu'à 128 tokens, bornés par `RBITNET_PREFILL_CHUNK_TOKENS`. Une ABI explicite configure chaque contexte et alloue un workspace partagé par toutes les couches, avec des dimensions de têtes indépendantes de l'embedding. Les adresses des graphes restent stables pour chaque taille/mode de sortie. Une ancienne DLL ou une allocation refusée garde le pipeline GPU sériel.

Les projections GGUF partagent leurs tuiles de poids entre tokens. La convolution traite l'historique causal dans le kernel ; chaque warp GDN conserve une ligne de l'état pendant le bloc et n'écrit son état final qu'une fois. Les sorties intermédiaires restent disponibles. L'attention exclut les clés futures du bloc. La dernière position seule produit logits/argmax, avec une synchronisation par bloc. Les kernels convolution/GDN à un token sont conservés séparément, inchangés.

`RBITNET_CUDA_PREFILL_TF32X3=1` sélectionne les grandes projections compensées du lot Llama : trois produits TF32, sous-totaux FP32, petites matrices SIMT. Les poids GGUF restent inchangés, sans miroir complet F32. Les checkpoints coupent les blocs à leur longueur exacte ; l'état GDN ne se tronque pas comme le KV dense.

## Ablation isolée

Même CLI/DLL, Qwen3.5-2B Q8_0, 1 480 tokens d'entrée, 128 tokens de sortie. Un cycle de chauffe puis deux cycles mesurés pour deux prompts longs ; le contrôle court est exclu des médianes. Réponses naturelles communes, distinctes des fixtures brutes du comparatif des moteurs.

| Mode | Prefill ms, médiane (min–max) | Décodage ms | tok/s | HTTP ms |
|---|---:|---:|---:|---:|
| baseline | 5262.0 (5256–5273) | 568.0 | 225.4 | 5840.9 |
| block | 1616.5 (1610–1618) | 568.0 | 225.4 | 2189.9 |
| tensor | 808.5 (803–813) | 566.0 | 226.1 | 1381.0 |
| tensor-prefix | 5.0 (4–5) | 565.0 | 226.5 | 590.1 |

36 réponses et 12 paires HTTP/SSE identiques au chemin sériel, en glouton, sampling avec seed et pénalités ; quatre arrêts explicites. Les blocs exécutés et les lancements Tensor Cores sont vérifiés par les compteurs natifs. Les requêtes chaudes réutilisent un checkpoint : aucun GEMM de préremplissage n'est attendu sur ces hits. [Mesures brutes](ablation.json), [résumé et mémoire](summary.json), [streaming](streaming.json).

## Comparatif fraîchement remesuré

Ollama 0.35.0, llama.cpp b11351 / 631109b34, même GGUF/tokenizer et mêmes prompts bruts. Un passage de chauffe puis trois mesures, 32 tokens courts / 128 longs ; KV froid pour les mesures. La sonde SSE est distincte des mesures HTTP unary. Les trois moteurs gardent les mêmes contrôles de qualité : calcul et français passent, rappel ORION échoue pour tous. Pas de comparaison de qualité à grande échelle.

| Prompt | Moteur GPU | tok/s | HTTP ms | Prefill ms | Premier contenu SSE ms | Qualité |
|---|---|---:|---:|---:|---:|---:|
| short | llama.cpp | 230.5 | 189.8 | 19.6 | 35.2 | 2/3 |
| short | ollama | 215.7 | 206.7 | 49.8 | 68.8 | 2/3 |
| short | rbitnet | 258.1 | 195.8 | 55.0 | 70.0 | 2/3 |
| long | llama.cpp | 223.3 | 674.2 | 85.8 | 95.3 | 2/3 |
| long | ollama | 223.5 | 714.6 | 129.5 | 157.6 | 2/3 |
| long | rbitnet | 229.8 | 1343.5 | 783.0 | 799.6 | 2/3 |

Rbitnet dépasse ici le débit de décodage des références et se rapproche de leur latence courte. Il reste environ deux fois plus lent en HTTP long. Son contexte alloué est 8 192 contre 2 048 demandés aux références ; KV F32 pour Rbitnet/llama.cpp, défaut F16 Ollama. Ces différences empêchent une revendication de supériorité générale. RSS = arbre de processus ; VRAM = relevés globaux incluant le bureau. Aucun autre travail GPU ni compilation pendant ces mesures. [Court](comparison-short.json), [long](comparison-long.json), [configuration](comparison-manifest.json).

## Validation et limites numériques

- 242 tests workspace, zéro échec et un ignoré dans 25 suites ; Clippy termine avec les warnings publiés, notamment des warnings incrémentaux Windows. CUDA et CLI release compilent.
- Oracles de couches FP64 utilisant la déquantification CPU indépendante : huit formats, GDN head 32/128/256, convolution 1/4/16, ratios de têtes, blocs 7/16/33/64/128, reset et continuation. Tolérance 5e-5 × (1+abs(référence)) sur les blocs ; comparaison au GPU sériel incluse pour GDN. Les formats K utilisent les alignements valides, avec sortie SSM F32 lorsqu'elle n'est pas alignée.
- Modèle réel : 53 IDs de référence, cache/divergence/seed/pénalités/annulation ; 384 positions avec tokens imposés en eager/graphe, SIMT/TF32. Argmax identiques, logits dans 0.003 × (1+abs(référence)), pire KL 8.334e-12 et pire delta de log-vraisemblance 3.917e-6. Ce contrôle limité n'est pas une perplexité sur corpus.
- Annulation pendant le préremplissage : après un bloc sur 1 112 tokens, reprise depuis checkpoint identique à une génération froide. Trois déconnexions/reprises, arrêt HTTP/SSE et quatre requêtes concurrentes vérifiés séparément. Le runtime reste sérialisé : cela ne constitue pas du batching.
- Ancienne DLL du lot split-KV : vraie réponse Paris, capacité de bloc 1, compteurs de blocs/Tensor Cores nuls, pipeline Qwen GPU réel conservé.

Les premières fixtures synthétiques fortement couplées ont dépassé la tolérance FP64 : attention Q4_K avec sorties de l'ordre de 2 480, et GDN Q5_K (3.8544497 bloc / 3.8541913 GPU sériel / 3.8541301 FP64). Ces échecs sont conservés dans [initial-oracle-gap.log](initial-oracle-gap.log) et [scalar-oracle-diagnosis.log](scalar-oracle-diagnosis.log). Les coefficients synthétiques ont été bornés sans élargir la tolérance ; cela ne démontre ni une précision universelle, ni les résultats d'un vrai GGUF Q4_K Qwen. Les huit formats demeurent un contrôle numérique borné ; le benchmark réel utilise Q8_0.

La suite workspace a aussi reproduit une course entre deux tests modifiant `RBITNET_BACKEND`. Un mutex commun et la restauration de la valeur précédente corrigent cette interférence ; [échec conservé](initial-workspace-race.log), [suite finale](workspace.log).

## Reproduction et traçabilité

```powershell
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir E:/devs/Rbitnet/target/qwen-block/cuda
cargo build -p rbitnet-cli --release
pwsh -NoProfile -File docs/benchmarks/2026-10-03-qwen-block-prefill/gpu-validation.ps1
pwsh -NoProfile -File docs/benchmarks/2026-10-03-qwen-block-prefill/http-driver.ps1
```

Le driver de comparaison utilise un daemon Ollama isolé, le répertoire de modèles indiqué et uniquement son propre PID ; le daemon utilisateur n'est pas arrêté. Les chemins locaux sont dans les drivers/manifests. [Empreintes des sources et preuves](manifest.json), [validation GPU](gpu-validation.log), [annulation en prefill](prefill-cancel.log), [ancienne DLL](legacy.json), [Clippy](clippy.log). Base : 667a700, changements de ce lot dans la branche `codex/qwen-block-prefill`. #95 reste ouvert pour les autres architectures et optimisations matricielles ; la comparaison des quatre modèles CPU/GPU reste à terminer dans #98.
