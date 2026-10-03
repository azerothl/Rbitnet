# Vérification spéculative Llama : 3 octobre 2026

Le runtime propose des continuations à partir des IDs déjà présents dans le prompt ou la génération. Il vérifie le token courant et jusqu'à quinze propositions dans un forward CUDA matriciel, avec un résultat par position. Seules les décisions du modèle cible sont publiées. Au premier rejet, le KV est tronqué au préfixe accepté, puis le token de correction poursuit le décodage. Des graphes séparés par taille/mode conservent les adresses et lisent la position depuis un buffer device.

Ce lot dépend des primitives de la [PR #99](https://github.com/azerothl/Rbitnet/pull/99) ; il avance [#95](https://github.com/azerothl/Rbitnet/issues/95) et [#97](https://github.com/azerothl/Rbitnet/issues/97). Le petit modèle draft GGUF et la restauration spéculative GDN/convolution Qwen restent à traiter. L'ancienne comparaison de mots entre générations indépendantes a été retirée du scheduler : les architectures/DLL sans vérification native effectuent une génération ordinaire, sans faux compteurs d'acceptation.

## Décision de sampling

Le [papier de Leviathan et al., §2.3](https://arxiv.org/html/2211.17192v2) accepte une proposition avec probabilité `min(1, p/q)` et corrige un rejet avec la distribution normalisée `max(0, p-q)`. Notre draft déterministe a `q = delta(d)`. Nous utilisons le couplage équivalent suivant : tirer `T` avec le sampler cible, accepter si `T=d`, sinon publier `T` comme correction. Ainsi `P(acceptation)=p(d)` et, après rejet, `P(T=t)=p(t)/(1-p(d))` pour `t!=d`. Ce couplage découle de la formule du papier ; il ne nécessite pas un second tirage et conserve l'ordre des tirages seeded. Pénalités et masques utilisent uniquement l'historique accepté.

Cette propriété porte sur la distribution calculée par le modèle cible. Les kernels matriciels et sériels accumulent en F32 dans des ordres différents : les logits sont vérifiés dans une tolérance numérique, sans promesse d'identité bit à bit sur tout prompt. Les séquences et réponses réelles testées sont identiques, y compris avec seed et pénalités.

## Mesures finales et décision de défaut

Même machine et GGUF Llama 3.2 1B Q4_K_M que le [lot cache](../2026-10-03-cache-foundation/README.md). Même binaire/DLL par ablation, préremplissage par blocs actif, préfixe désactivé, F32 KV, contexte maximal 8 192. Une chauffe puis deux cycles mesurés ; les temps de décodage ci-dessous sont médians avec leurs min–max. Le [rapport brut](results.json) publie requêtes, réponses, latence HTTP, prefill, mémoire, uploads/downloads, temps draft/verify et compteurs. Les [statistiques](summary.json) et [empreintes](manifest.json) complètent la reproduction.

| Prompt (sortie réelle) | Référence, ms | PLD 4, ms | PLD 8, ms | PLD 15, ms |
|---|---:|---:|---:|---:|
| Séquence grecque (14 tokens) | 31 (31–31) | 68 (68–68) | 48 (48–48) | 26 (26–26) |
| Code répété (75 tokens) | 180 (180–180) | 191 (191–191) | 192,5 (183–202) | 198,5 (197–200) |
| Récit libre (256 tokens) | 731 (730–732) | 752 (751–753) | 753,5 (751–756) | 753,5 (753–754) |
| Capitale (1 token) | 2 (2–2) | 2 (2–2) | 2 (2–2) | 2 (2–2) |

PLD 15 gagne environ 19 % en tok/s sur la réponse grecque courte (538 contre 452 tok/s). Le modèle ne suit pas entièrement la demande de huit répétitions : le benchmark conserve sa réponse de référence de 14 tokens, sans la présenter comme une répétition complète. Les autres prompts ne gagnent pas. Le débit du récit reste proche de 340 contre 350 tok/s ; ce contexte est beaucoup plus court que les prompts de 1 729 tokens du lot cache.

**No-go pour l'activation générale.** Le coût des GEMM de vérification/head reste trop élevé sur ce petit modèle pour compenser la plupart des propositions. Une garde par requête mesure le coût du forward sériel, exclut son premier appel froid et supprime les nouvelles propositions lorsqu'un bloc dépasse de 10 % le coût observé par token avancé. Elle limite les pertes, sans garantir une accélération. Une moyenne de timings peut varier entre répétitions et changer le nombre de tentatives ; les décisions du modèle cible restent indépendantes de ce choix de placement du calcul.

Le [premier essai sans graphes de vérification ni garde](initial-ungraphed.json) reste visible : PLD 15 donnait 1 217,5 ms sur le récit contre 729 ms en référence, avec cinq propositions acceptées sur 288. Il emploie un autre binaire/DLL ; ce résultat est un essai intermédiaire, pas une ablation isolant séparément graphes et garde. Le rapport final utilise un seul binaire/DLL pour ses quatre modes. Les 48 réponses sont identiques à la référence, de même que les douze paires HTTP/SSE en glouton, seed et pénalités. Quatre stops et quatre déconnexions/reprises supplémentaires terminent correctement.

## Validation matérielle

- [Oracles avec graphes](graph-replay-oracles.log) : logits/argmax de chaque position, tailles 1/2/7/15/16, rejeu du même graphe à une autre position, troncature puis nouvelle queue comparée au forward sériel. [Même oracle sans graphes](eager-oracles.log).
- [Modèle réel](graph-real-sequences.log) : golden de 43 IDs, comparaisons sériel/PLD en greedy/seed/pénalités, 18 paires de préfixes chaud/froid, annulation après une vérification réelle puis reprise identique.
- [Workspace CPU](graph-workspace.log) : 230 tests passent, zéro échec, un ignoré, dont les tests optionnels qui sortent tôt sans leurs variables ; la suite de sampling contrôlé valide 100 000 tirages et la distribution résiduelle après rejet.
- [Ancienne DLL avec flag PLD actif](old-dll.log) : le golden réel passe en décodage résident ordinaire ; les deux tests de nouvelles fonctions sont désactivés sans leurs variables d'activation.
- [Préfixe + spéculation via HTTP](streaming-prefix.json) : trois déconnexions sur récit long suivies de reprises, stop HTTP/SSE et quatre requêtes concurrentes, dont les répétitions qui exécutent réellement des blocs spéculatifs. Le runtime partagé reste sérialisé ; cette preuve ne couvre pas le batching GPU du ticket #96.

RSS maximal autour de 1,99–2,00 Go ; pics GPU globaux 3 545–3 557 MiB. La vérification alloue un scratch borné à seize positions : logits `16*vocab*4`, deux tables d'argmax `16*ceil(vocab/256)*4`, seize IDs et les activations du bloc de prefill. Les graphes et le bureau sont également présents dans les mesures ; les différences globales de VRAM ne constituent pas une mesure exacte de chaque allocation.

## Reproduire

`RBITNET_SPECULATIVE_PLD=1` ou `RBITNET_SPECULATIVE=1` active le chemin natif lorsqu'il est disponible. `RBITNET_SPECULATIVE_TOKENS=1..15` borne les propositions (8 par défaut). `RBITNET_SPECULATIVE_ADAPTIVE=0` désactive la garde pour des ablations ; le défaut est actif dans l'option spéculative. Le mode reste désactivé par défaut. Une DLL antérieure sans verify/truncate garde le décodage ordinaire ; aucun fallback de format/état n'est effectué au milieu d'une séquence en cours d'exécution CUDA.

```powershell
python scripts/benchmark_speculative.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --binary target/release/rbitnet.exe --library target/speculative-blocks/cuda-graph/rbitnet_cuda_quant64.dll --output-dir target/speculative-reproduction
python scripts/validate_cache_streaming.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --binary target/release/rbitnet.exe --library target/speculative-blocks/cuda-graph/rbitnet_cuda_quant64.dll --output-dir target/speculative-streaming --speculative
```
