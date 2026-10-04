# Fusion parallèle des experts : ablation et décision

Cette expérience parallélise les projections descendantes des experts sélectionnés, puis combine leurs contributions dans l’ordre du kernel initial. Elle est conservée dans `experimental/` pour reproduction et revue ; aucun kernel ni fichier Rust du moteur livré n’est modifié dans cette branche.

La variante ne justifie pas une activation par défaut dans ces mesures. Le faible gain GPT-OSS en placement fixe ne se retrouve pas avec le cache d’experts ; GLM en placement fixe ralentit. Le réglage de production est conservé. Deux répétitions chaudes par prompt ne suffisent pas à établir un gain statistique.

## Débit mesuré

RTX 4080 SUPER 16 Gio, Ryzen 7 9800X3D, 64 Gio, Windows. Même CLI expérimental et DLL pour les trois modes ; contexte 1024, huit notes communes, 128 tokens de sortie maximum, trois cycles dont un de chauffe exclu. Même modèle/tokenizer et politique LRU par placement. Médiane [min–max] de deux répétitions, chaque prompt séparément.

| Placement | Mode | Récit, tokens/s [min–max] | Code, tokens/s [min–max] |
|---|---|---:|---:|
| gpt-fixed | unfused | 89.26 [89.07–89.45] | 89.26 [89.01–89.51] |
| gpt-fixed | serial-fused | 89.42 [89.01–89.82] | 90.20 [90.01–90.40] |
| gpt-fixed | parallel-fused | 90.65 [90.46–90.84] | 90.62 [90.40–90.84] |
| gpt-segmented | unfused | 74.81 [74.29–75.34] | 76.88 [76.19–77.58] |
| gpt-segmented | serial-fused | 74.91 [73.95–75.87] | 77.37 [76.69–78.05] |
| gpt-segmented | parallel-fused | 74.22 [73.10–75.34] | 77.17 [76.24–78.10] |
| glm-fixed | unfused | 20.55 [20.17–20.94] | 20.80 [20.43–21.17] |
| glm-fixed | serial-fused | 20.05 [19.99–20.10] | 19.57 [18.94–20.20] |
| glm-fixed | parallel-fused | 19.92 [19.84–20.00] | 20.15 [20.11–20.19] |
| glm-cache | unfused | 24.53 [24.36–24.71] | 24.51 [24.49–24.53] |
| glm-cache | serial-fused | 24.73 [24.60–24.86] | 24.53 [24.52–24.54] |
| glm-cache | parallel-fused | 24.62 [24.48–24.75] | 24.25 [24.10–24.39] |

Les captures publient aussi préremplissage, latence HTTP, transfert des experts, allocations et mémoire. Les placements `*-fixed` utilisent les banques admises au chargement ; `gpt-segmented` et `glm-cache` utilisent un cache de 8192 Mio. Le budget device est de 12288 Mio. Les résultats ne constituent pas une nouvelle comparaison avec Ollama ou llama.cpp.

## Qualité et intégration

- Le test GPU synthétique contrôle huit formats de poids, banques fixes/dynamiques, biais mixtes, graphes et remplissage du cache : résultat bit à bit identique au kernel initial, avec une référence F64 indépendante.
- Les fixtures GPT-OSS et GLM réelles contrôlent distributions/logits, générations, graines, pénalités, préfixes, réinitialisation et annulation. Les comparaisons de distributions utilisent leurs seuils KL/NLL : elles ne promettent pas des vecteurs réels bit à bit identiques entre toutes les stratégies de prefill.
- Les quatre ablations passent 108 réponses JSON, 36 comparaisons JSON/SSE et 12 arrêts explicites. Les deux suites réseau contrôlent la combinaison volontaire fusion parallèle + transferts asynchrones, avec un cache de 8192 Mio. Ces suites réseau ne sont pas les mesures de débit synchrones.
- Compilation, Clippy et fixtures ciblées passent ; la suite workspace complète et l’intégration de ce mode au produit ne sont pas revendiquées.

CLI : `7d3c8022e10c93c6736a4b393b45997b435db2289c9fb1f831799a21f2f22be7`. DLL : `c47dfc70baf629233df1f5755eda2881f61d93e64f023cdc756deb62eb612369`. Le manifeste relie les sources compilées aux empreintes ; `evidence-index.json` relie chaque fichier conservé à la capture originale.

## Reproduction de l’expérience

Partir de la base #114 (`6f67224184e767f1c9dfcfc5633d106ae4d0463a`) dans un checkout isolé, puis copier les fichiers de `experimental/` en conservant leur chemin relatif à cette racine. Compiler le CLI et la DLL ; utiliser `source/benchmark.py` depuis la racine avec les modèles du manifeste, `--parallel-fusion`, trois cycles, huit notes et 128 tokens. La suite `source/live.py` utilise `--async --fusion 2`. Le checker original est fourni comme diagnostic et contient des chemins de préparation locaux à adapter.

Refs [#88](https://github.com/azerothl/Rbitnet/issues/88), [#98](https://github.com/azerothl/Rbitnet/issues/98).
