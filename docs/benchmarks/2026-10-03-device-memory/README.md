# Budget CUDA commun et cache d'experts — 2026-10-03

Le modèle dispose désormais d'un plafond commun pour les allocations CUDA gérées par Rbitnet. Le déchargement du serveur libère les poids, caches et états du modèle ; un espace de travail générique peut rester vivant dans son thread. Les mesures ne justifient pas l'activation automatique du cache d'experts : sous 512 Mio, il provoque ici des transferts répétés sans réutilisation, et avec 8 Gio le débit de décodage reste inférieur au placement fixe.

## Comportement et limites

`RBITNET_CUDA_DEVICE_BUDGET_MB` fixe le plafond en Mio ; `RBITNET_CUDA_DEVICE_MARGIN_MB` réserve une marge sur la mémoire libre observée, 256 Mio dans les mesures. Zéro désactive le plafond explicite, sans désactiver la comptabilité. Le registre distingue poids, KV/états, activations, snapshots, experts, scratch et autres allocations. Les vues et leases partagées ne comptent pas deux fois la même allocation. Les allocations internes du pilote, des graphes CUDA et de cuBLAS ne sont pas couvertes par ce registre ; la VRAM globale observée est publiée séparément.

La réservation anticipée de l'état est implémentée pour GPT-OSS et MLA. Elle précède le placement des poids et la capacité du cache MoE. Les autres chemins restent soumis au plafond physique, avec une réservation anticipée encore à compléter. Une ancienne DLL sans ABI de comptabilité est refusée lorsqu'un plafond explicite est demandé. Un état trop grand est refusé avant le placement des poids ; les six routes HTTP/SSE exposent une erreur de chargement structurée et `/ready` renvoie 503.

Le serveur détenait une référence supplémentaire au moteur initial pendant toute son exécution. Le premier essai de déchargement renvoyait 200 et bloquait les nouvelles inférences, mais conservait environ 11,4 Go d'allocations gérées. Le [résultat initial](initial-unload-retention.log) est conservé. Cette référence a été supprimée : après la fin des requêtes actives, les catégories du modèle retombent maintenant à zéro, et le rechargement dans le même processus fonctionne.

## Ablation à plafond de 12 Gio

RTX 4080 SUPER 16 Gio, Ryzen 7 9800X3D, Windows, 16 threads CPU. Même exécutable, DLL, GGUF et tokenizer pour tous les modes. Contexte alloué 2 048, LRU, cache désactivé ou 16/512/8 192 Mio. Pipeline GPT complet, split-KV, préfixes et préremplissage par blocs désactivés pour isoler le placement et le cache sur le chemin partiel. Ce protocole ne compare pas le cache au chemin GPT fixe le plus rapide de la PR #106.

Deux prompts étendus de chat, environ 326 tokens GPT et 287 tokens GLM, 64 tokens de sortie, plus un contrôle court « Paris ». Un cycle de chauffe, puis deux cycles mesurés ; quatre échantillons étendus par mode. Ces prompts ne constituent pas un stress à la capacité maximale du contexte. Le temps de préremplissage est une métrique interne ; le premier contenu SSE n'a pas été chronométré dans ce protocole.

| Modèle / cache Mio | Prefill ms, médiane (min–max) | Décodage tok/s, médiane (min–max) | HTTP ms | Réutilisation des acquisitions | Upload Gio / requête | Upload ms | Géré vivant / pic Mio |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT / 0 | 6552,5 (6476–7002) | 46,32 (40,38–48,93) | 8026,6 | sans cache | 0 | 0 | 10881,6 / 10881,6 |
| GPT / 16 | 32698,5 (29607–32925) | 12,90 (12,52–13,41) | 37564,9 | aucune acquisition | 0 | 0 | 1199,6 / 1199,6 |
| GPT / 512 | 101366,5 (100612–101967) | 3,27 (3,19–3,31) | 121001,1 | 0 % | 459,76 | 109322,8 | 1703,9 / 1703,9 |
| GPT / 8192 | 5248,5 (5077–5332) | 43,17 (38,10–50,28) | 6728,4 | 99,44 % | 2,57 | 591,3 | 9381,4 / 9381,4 |
| GLM / 0 | 22711 (20187–23688) | 12,72 (11,98–13,45) | 27802,7 | sans cache | 0 | 0 | 12254,2 / 12254,2 |
| GLM / 16 | 31874,5 (31529–33052) | 8,31 (7,71–8,76) | 39811,2 | aucune acquisition | 0 | 0 | 1676,2 / 1676,2 |
| GLM / 512 | 114979 (113627–119348) | 2,44 (2,43–2,45) | 141217,3 | 0 % | 340,18 | 99809,1 | 2184,1 / 2188,1 |
| GLM / 8192 | 28430,5 (27627–29408) | 12,11 (10,94–12,96) | 34049,7 | 86,79 % | 44,94 | 12861,2 | 9866,0 / 9868,2 |

Les taux concernent les acquisitions réellement faites : à 16 Mio, une sélection entière ne tient pas et le FFN utilise le repli existant, sans acquisitions de cache. Ce repli peut encore employer des projections GPU individuelles ; il ne constitue pas un mode CPU forcé. À 512 Mio, les groupes tiennent mais le parcours des couches évince les groupes avant leur prochaine réutilisation. Le compteur d'upload mesure les octets transférés vers le GPU ; sa durée inclut les copies depuis la mémoire hôte paginable et les synchronisations du chemin actuel.

Avec 8 Gio, GPT réduit la latence HTTP d'environ 16 % et le prefill de 6,55 à 5,25 s, mais son débit de décodage baisse. GLM conserve aussi une forte dépense de transfert. Le cache reste désactivé par défaut. Les compteurs mesurés rendent prioritaire le choix CPU/GPU fondé sur le coût réel (#86) et le recouvrement des copies (#84). Aucune parité avec Ollama/llama.cpp n'est revendiquée par cette ablation. [Données brutes](ablation.json), [résumé](summary.json).

## Plafond réduit de 6 Gio

Même protocole et contexte, plafond total réduit à 6 144 Mio, cache désactivé ou 4 096 Mio. Un cycle de chauffe puis un cycle mesuré de deux prompts étendus ; seulement deux échantillons par mode, donc estimation limitée. Les quatre modes respectent le plafond et donnent les mêmes textes sur les 24 réponses. Le cache de 4 Gio ne justifie pas une activation par défaut ici.

| Modèle / cache Mio | Prefill ms, médiane (min–max) | Décodage tok/s, médiane (min–max) | HTTP ms | Hits | Upload Gio / requête | Géré vivant / pic Mio |
|---|---:|---:|---:|---:|---:|---:|
| GPT / 0 | 17349.0 (16582–18116) | 19.29 (19.05–19.53) | 20669.7 | 0.00% | 0.00 | 6026.1 / 6026.1 |
| GPT / 4096 | 20805.5 (19968–21643) | 15.07 (10.67–19.47) | 25456.5 | 84.83% | 69.74 | 5284.2 / 5284.2 |
| GLM / 0 | 27247.5 (26666–27829) | 10.14 (10.07–10.21) | 33574.8 | 0.00% | 0.00 | 6123.5 / 6123.5 |
| GLM / 4096 | 58069.0 (57820–58318) | 6.37 (6.08–6.66) | 68144.9 | 61.83% | 130.05 | 5772.1 / 5772.2 |

[Données brutes](cap6144.json), [résumé](summary-cap6144.json).

## Correction et contrôles

- 248 tests workspace réussis, un ignoré ; Clippy et compilations release/CUDA réussis avec les warnings publiés.
- Test matériel du registre : partage de vues, catégories, refus au plafond, libération invalide, rollback d'une création native échouée, changement de plafond et admission concurrente. Sous 64 Kio, huit threads demandent chacun 16 Kio ; exactement quatre allocations sont admises.
- 19 régressions natives et un contrôle réel des leases de groupes d'experts ; la sélection reste protégée jusqu'à la fin des lectures GPU.
- GPT réel : 72 positions de logits et 18 générations, graphes/eager/split, maximum KL 3,465e-12 et delta NLL 6,442e-6. Cette vérification numérique est bornée, sans garantie universelle sur tous les prompts.
- 72 réponses principales et 24 réponses sous plafond réduit identiques au placement fixe correspondant ; 36 paires HTTP/SSE concordantes, 12 arrêts explicites et 12 cycles déchargement/rechargement. Les catégories du modèle sont nulles après chaque déchargement, à l'exception du scratch conservé par le thread.
- Pour GPT et GLM avec cache de 8 Gio : trois déconnexions/reprises (glouton, seed, pénalités), arrêt HTTP/SSE et quatre requêtes concurrentes. Le runtime sérialise ces requêtes ; ces contrôles ne prouvent pas un batching natif.
- Refus d'un modèle avec plafond de 64 Mio et d'une ancienne DLL sans ABI : aucune allocation de modèle retenue, erreurs structurées sur les six routes HTTP/SSE, readiness 503.

[Allocateur](allocator.log), [régression native](native-regression.log), [leases](cache-leases.log), [workspace](workspace.log), [Clippy](clippy.log), [GPT réel](gpt-teacher.log), [streaming GPT](streaming-gpt.json), [streaming GLM](streaming-glm.json), [refus](refusals.json).

## Reproduction et provenance

Le [driver](reproduce.ps1) donne les chemins de cette machine et exécute les lots séquentiellement. Les résultats contiennent les SHA-256 de l'exécutable et de la DLL. Le [manifest](manifest.json) indique le commit de code testé, les sources, les GGUF/tokenizers et les preuves publiées. Les champs imbriqués `config.environment` hérités de l'ancien comparatif décrivent son contexte historique ; les empreintes au premier niveau et ce manifest identifient le build mesuré ici.

Le plafond concerne un device explicitement choisi ; les compteurs sont globaux au processus. Les métriques par modèle/couche, le premier contenu SSE, les copies asynchrones, la politique fondée sur le coût et le cache de sessions RAM/SSD restent à traiter. #83 reste ouvert. PR empilée sur #106 ; refs #83, #88, #89, #24 et #98.
