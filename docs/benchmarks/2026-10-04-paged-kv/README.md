# KV device paginé F32 Llama — 4 octobre 2026

Le chemin optionnel `RBITNET_CUDA_KV_PAGE_LIMIT=128` remplace les deux allocations KV denses du pipeline natif Llama par un pool de pages physiques de 32 tokens. L'attention normale et split-KV lisent ces pages ; les forwards, blocs, vérifications spéculatives, graphes CUDA et snapshots utilisent la même table persistante. Le format reste F32 et le défaut reste dense.

Les snapshots partagent les pages immuables. Écrire dans une page encore référencée par un snapshot ou un autre contexte déclenche une copie avant écriture. Le pool réutilise les pages libérées et refuse les admissions dépassant sa limite ou le plafond CUDA géré. Les pages physiques et tables sont comptées dans `kv_state`, une seule fois par allocation ; les snapshots partagés ne créent pas une seconde charge `prefix`. Le budget logique du cache de snapshots reste conservateur.

## Mesures du binaire intégré

Même Llama-3.2-1B Q4_K_M, RTX 4080 SUPER, 24 notes communes, contexte 2048, sortie 128, plafond 12 Gio/marge 256 Mio, split-KV, blocs de 128. Trois cycles par mode : le premier est la chauffe ; la table utilise quatre observations longues des deux cycles suivants. Les réponses et prompts exacts sont conservés.

| Mode | Préremplissage ms | Décodage tok/s | HTTP ms | KV + snapshots à la fin du protocole, Mio |
|---|---:|---:|---:|---:|
| dense | 518.0 | 390.24 | 855.4 | 128.00 |
| dense-prefix | 3.0 | 374.82 | 349.8 | 248.00 |
| paged | 545.0 | 358.05 | 906.8 | 48.00 |
| paged-prefix | 3.0 | 356.07 | 379.7 | 56.00 |

La dernière colonne vient des catégories absolues après tout le corpus, y compris les vérifications SSE/sampling/arrêt ; ce n'est pas la moyenne des quatre requêtes longues. `summary.json` distingue mémoire live médiane de ces requêtes, pic géré et catégories finales. La mémoire du pilote et la RAM hôte sont conservées séparément dans les captures brutes. Les allocations hôtes F32 existantes ne sont pas supprimées par ce lot.

Le décodage paginé varie de -8.25 % face au dense dans cette ablation. Cette mesure borne une réduction de mémoire ; elle ne justifie pas de remplacer le dense par défaut ni une supériorité générale de tok/s. Aucun nouveau comparatif Ollama/llama.cpp n'est effectué ici.

## Correction et cycle de vie observés

- 284 tests workspace réussis, 1 ignoré(s), Clippy, build CUDA multi-SM et CLI release réussis ; les avertissements existants sont visibles.
- Deux fixtures GGUF exécutées explicitement dans deux processus, split-KV 0 puis 1. Graphes 0/1, 1/4/8 contextes partageant un préfixe, page partielle à 33 tokens, branchements/COW, blocs et vérification multi-positions : logits F32 bit à bit identiques au dense de la même variante.
- Les refus d'un snapshot dense d'un autre contexte, d'un pool avec autre propriétaire de poids ou variante d'attention/TF32 sont vérifiés. Un peer et un snapshot peuvent survivre au contexte racine ; les allocations restent chargées une seule fois puis sont libérées.
- Limite de deux pages : refus d'une nouvelle écriture, libération puis reprise, sans dépasser la limite physique. La génération du propriétaire empêche la réutilisation accidentelle d'une adresse de contexte.
- 36 observations, 12 paires unary/SSE et quatre arrêts exacts sur le même binaire/DLL. Deux suites serveur, blocs 0/1 : déconnexion/reprise, glouton, seed/pénalités, arrêt et quatre requêtes concurrentes sérialisées.

Les 1/4/8 contextes de la fixture mesurent le partage/cycle de vie, pas un forward multi-séquences. Le serveur utilise encore un seul runtime de génération. Les formats F16/Q8, pages Qwen/GPT/MLA, le KV CPU/SSD et le continuous batching restent dans leurs lots distincts (#93, #94, #96). #92 reste ouvert pour ces architectures et l'intégration serveur multi-séquences.

Le harness réseau initial réassignait sa liste de modes après la restriction Llama : il a aussi exécuté cinq contrôles Qwen avec le chemin Qwen existant. Ces captures supplémentaires sont conservées, mais ne prouvent aucun KV paginé Qwen. Les dix contrôles Llama des deux modes constituent la validation serveur paginée rapportée ici. La restriction de modes du harness de travail est corrigée pour les expériences suivantes ; la copie publiée conserve le script effectivement exécuté.

## Provenance et reproduction

Les captures `production/`, `ablation/` et `live/` appartiennent à la version intégrée. Les captures `prototype-*` appartiennent au premier prototype et ne sont pas regroupées dans les résultats intégrés. `manifest.json` lie les sources, le binaire et la DLL. Les scripts `reproduction/` documentent les commandes et variables ; les captures publiées normalisent uniquement BOM/CRLF et conservent leurs empreintes originales.

Ancienne DLL, backend CPU ou KV paginé hôte avec l'option native demandée : refus explicite plutôt qu'un repli silencieux en dense. L'ABI dense existante reste disponible ; l'ABI paginée reçoit explicitement les variantes pour éviter une divergence de configuration Rust/CRT sous Windows.
