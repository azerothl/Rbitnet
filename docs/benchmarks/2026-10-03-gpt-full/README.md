# GPT-OSS : pipeline CUDA résident — 2026-10-03

Le pipeline complet pour les banques d'experts fixes supprime les échanges d'activations et de routage par couche. Sur RTX 4080 SUPER et GPT-OSS-20B Q4_K_M, l'ablation longue passe de 39,1 à 90,6 tok/s avec l'attention fractionnée. Le préremplissage reste sériel et le comparatif conserve un retard sur les références. Mode expérimental opt-in.

## Implémentation

`RBITNET_CUDA_GPT_FULL=1` active le chemin si toutes les projections et banques d'experts sont résidentes. `RBITNET_REQUIRE_GPT_FULL=1` refuse le chargement lorsqu'il ne peut être créé ; sinon l'ancienne exécution partielle reste disponible. `RBITNET_CUDA_GPT_FULL_GRAPH=0` sélectionne le mode eager ; les trois graphes par défaut correspondent au préremplissage sans tête, aux logits F32 et à l'ID glouton.

Normes, biais Q/K/V, RoPE NeoX partiel avec YaRN, GQA, sinks, fenêtres alternées, router et FFN OpenAI restent sur un stream privé. Les banques existantes sont empruntées avec durée de vie garantie. Les sinks participent uniquement au dénominateur de l'attention, aussi dans la réduction split-KV. Le routeur conserve les experts du top-k, la probabilité softmax des experts retenus et leur ordre ; le FFN utilise les biais/clamps OpenAI. Une synchronisation par token, embedding/position H2D et sortie logits/ID D2H.

Le modèle réel utilise 24 banques fixes résidentes et 10 659 MiB de poids quantifiés. Le cache MoE dynamique n'est pas compatible avec ce nouveau chemin ; l'activation de ce cache conserve le chemin partiel. Pas encore de préremplissage GPT par blocs ni de snapshots de préfixe. Le budget actuel est un budget de poids, qui exclut états/KV/scratch : son unification demeure dans #83. SIMD AVX2/AVX512 actif requis pour respecter le routeur CPU testé ; les autres cas gardent le chemin existant.

## Ablation isolée

Même CLI/DLL et GGUF/tokenizer. Un cycle de chauffe, puis deux cycles mesurés de deux prompts longs, 128 tokens de sortie. Le contrôle court est exclu des médianes. Contexte natif alloué 8 192 ; prompts de chat naturels distincts des fixtures brutes du comparatif.

| Mode | Prefill ms, médiane (min–max) | Décodage ms | tok/s | HTTP ms |
|---|---:|---:|---:|---:|
| baseline | 31405.5 (31010–31603) | 3274.0 | 39.1 | 34744.4 |
| full | 18870.0 (18837–18911) | 2174.5 | 58.9 | 21065.1 |
| full-split | 14854.0 (14851–14856) | 1412.5 | 90.6 | 16279.2 |

27 réponses et neuf paires HTTP/SSE identiques au chemin partiel, dont seed et pénalités, trois arrêts explicites. Les compteurs attestent l'exécution complète et le split-KV effectifs. Les transferts par requête longue passent de 3 195 585 792 à 18 426 876 octets H2D et de 2 401 832 960 à 512 octets D2H en glouton. [Mesures brutes](ablation.json), [résumé et mémoire](summary.json).

Trois déconnexions/reprises, arrêt HTTP/SSE et quatre requêtes concurrentes donnent les résultats attendus. Le runtime sérialise ces requêtes et repart de zéro : aucun gain de batching ou de préfixe GPT n'est revendiqué. [Contrôles séparés](streaming.json).

## Comparatif des trois moteurs

Ollama 0.35.0, llama.cpp b11351 / 631109b34, même GGUF et IDs de prompts, 16 threads CPU configurés. Un passage de chauffe et trois mesures, KV froid pour la performance, 32 tokens courts / 128 longs. Contexte demandé 2 048 pour les trois moteurs, capacité native Rbitnet 2 048 ; KV F32 Rbitnet/llama.cpp et défaut F16 Ollama. Sonde SSE séparée du HTTP unary.

| Prompt | Moteur GPU | tok/s | HTTP ms | Prefill ms | Premier contenu SSE ms | Qualité |
|---|---|---:|---:|---:|---:|---:|
| short | llama.cpp | 205.5 | 220.4 | 47.4 | 50.6 | 3/3 |
| short | ollama | 203.4 | 345.0 | 163.5 | 170.5 | 3/3 |
| short | rbitnet | 96.1 | 1232.2 | 883.0 | 1045.5 | 3/3 |
| long | llama.cpp | 185.0 | 893.3 | 185.7 | 188.3 | 3/3 |
| long | ollama | 202.5 | 964.2 | 300.8 | 331.2 | 3/3 |
| long | rbitnet | 91.4 | 16142.3 | 14744.0 | 14926.1 | 3/3 |

Ces mesures ne montrent pas la parité. Les trois contrôles de qualité limités passent, sans constituer une évaluation de corpus. RSS = arbre des processus ; VRAM = consommation globale incluant le bureau. Aucun autre test GPU ni compilation pendant les mesures. Le premier comparatif court avait conservé 1 024 pour les références ; ses [résultats initiaux](comparison-initial-1024-short.json) sont conservés mais exclus du tableau, remesuré à 2 048. [Court](comparison-short.json), [long](comparison-long.json), [configuration](comparison-manifest.json).

## Correction numérique et preuves

Les premières réductions parallèles passaient l'oracle synthétique mais échouaient sur le vrai modèle : à la position 147, couche 7, deux experts presque ex aequo (écart environ 1e-6) changeaient de rang, puis l'erreur se propageait au KV. La quatrième sélection était 9 contre 10 dans la référence. Le [premier échec de logits](initial-real-logit-gap.log) et la [trace par couche](layer-diagnosis.log) sont publiés. Des normes à somme ordonnée, un router utilisant les huit/seize accumulateurs FMA du SIMD CPU actif, des phases RoPE calculées sur hôte au chargement et une exponentielle précise sur les experts retenus restaurent les séquences testées. Les phases restent en VRAM ; aucun calcul hôte par couche ne revient dans le hot path.

- Deux oracles FP64 synthétiques : huit formats GGUF, biais/clamps, sinks/fenêtres, GQA, dimensions de têtes différentes de l'embedding, RoPE partiel, reset, modes logits/glouton, graphes/eager et split/ordinaire. F32 franchit la tuile 256 ; fixtures bornées et tolérance 5e-5 × (1+abs(référence)). Pas de précision universelle revendiquée.
- Routeur : ties, quiet NaN positifs/négatifs, zéros signés et infinis vérifient l'ordre `total_cmp`. Un échec initial de signe NaN canonicalisé par CUDA est [conservé](initial-router-nan.log) et corrigé ; cela ne prouve pas l'arithmétique de tous les NaN signaling.
- Modèle réel : 72 positions avec tokens imposés sur deux corpus, 18 générations gloutonnes/seed/pénalités et reset, trois modes GPU. Argmax et réponses identiques ; logits dans 0.003 × (1+abs(référence)), pire KL 3.465e-12 et pire delta NLL 6.442e-6. [Validation](gpu-validation.log), [ancienne configuration huit accumulateurs](ordered-eight-lane-real.log).
- 248 tests workspace passent, zéro échec, un ignoré ; Clippy et compilations CUDA/CLI réussissent avec les warnings publiés. Dix-huit tests natifs GPU de régression passent, dont Llama, Qwen, MoE et split-KV. [Workspace](workspace.log), [Clippy](clippy.log), [régression native](native-regression.log).
- Ancienne DLL sans nouvelle ABI : vraie réponse Paris, 24 couches d'experts GPU résidentes, compteur de tokens complets nul. [Résultat](legacy.json), [driver](legacy-driver.py).

Ces vérifications sont bornées : pas de perplexité générale, ni garantie d'égalité binaire sur tous les prompts/architectures. Le CPU reste la référence actuelle du choix de routage.

## Reproduction

```powershell
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir E:/devs/Rbitnet/target/gpt-full/cuda
cargo build -p rbitnet-cli --release
pwsh -NoProfile -File scripts/validate_gpt_full.ps1
C:/Users/azero/anaconda3/python.exe docs/benchmarks/2026-10-03-gpt-full/configure.py
pwsh -NoProfile -File docs/benchmarks/2026-10-03-gpt-full/http-driver.ps1
pwsh -NoProfile -File docs/benchmarks/2026-10-03-gpt-full/comparison-driver.ps1
```

Les drivers donnent les chemins de cette machine ; le daemon Ollama isolé est arrêté uniquement par son PID dans `finally`. [Empreintes des sources/binaires/modèle/proofs](manifest.json). Base 72b3b9c, branche `codex/gpt-oss-resident`. #88 reste ouvert pour l'intégration du cache dynamique et #95 pour le préremplissage/matrices ; le comparatif final CPU/GPU des quatre modèles appartient à #98.
