# Fusion RoPE et stockage KV F16/Q8 : décision de ne pas adopter

La fusion remplace deux lancements GPU par un lancement par couche pour les formats KV F16 et Q8. Le format F32 conserve ses kernels. Malgré des sorties identiques et des contrôles GPU réussis, le débit ne progresse pas de façon régulière. Aucun kernel ni fichier Rust du moteur livré ne change dans cette branche.

## Mesures

RTX 4080 SUPER 16 Gio, Ryzen 7 9800X3D, 64 Gio, Windows. CLI identique ; DLL canonique contre DLL fusionnée. Budget device 12288 Mio. Capacités KV 2048/8192, prompts avec 24/96 notes, sortie maximale 128 tokens, trois cycles dont une chauffe exclue. La capacité 8192 ne signifie pas un prompt actif de 8192 tokens. Deux répétitions chaudes par prompt ; ce volume ne permet pas une conclusion statistique.

Extraits des médianes, récit et code séparés. Les 96 groupes avec médiane/min/max sont publiés dans `warm-rates.json` ; toutes les réponses et métriques sont conservées.

| Capacité | Allocation | KV sans préfixe | Initial récit/code (tokens/s) | Fusion récit/code (tokens/s) |
|---:|---|---|---:|---:|
| 2048 | dense | f16 | 330.75/332.47 | 330.78/332.95 |
| 2048 | dense | q8 | 243.35/245.92 | 245.68/244.75 |
| 2048 | paged | f16 | 304.40/308.44 | 305.93/309.18 |
| 2048 | paged | q8 | 240.83/243.35 | 242.89/243.35 |
| 8192 | dense | f16 | 319.60/318.43 | 322.01/319.60 |
| 8192 | dense | q8 | 235.75/235.32 | 233.79/235.08 |
| 8192 | paged | f16 | 298.37/292.24 | 298.72/296.99 |
| 8192 | paged | q8 | 233.37/231.70 | 228.37/228.80 |

À 8192, le KV paginé Q8 passe de 233,37/231,70 à 228,37/228,80 tokens/s. Le F16 paginé progresse légèrement, mais ce résultat ne suffit pas à justifier la complexité supplémentaire pour les deux formats. Décision : conserver la DLL canonique. Les variations F32 servent de contrôle du bruit de mesure. Aucun nouveau comparatif Ollama/llama.cpp n’est revendiqué.

## Vérification

- 64 cas GPU avec décodeur/attention F64 indépendant ; exactitude et durée de vie du KV, préremplissage, préfixes et pages contrôlés avec split-KV désactivé et activé.
- Les 12 combinaisons F32/F16/Q8, dense/paginé, graphes désactivés/activés restituent exactement les vecteurs de logits de la référence canonique.
- Huit captures : 432 réponses JSON, 144 comparaisons JSON/SSE, 48 arrêts explicites. Les réponses de la DLL fusionnée correspondent aux réponses de la DLL canonique pour les mêmes requêtes.
- Quatre suites réseau F16/Q8 × dense/paginé, compilation et Clippy réussis. Ces mesures ne prouvent pas la qualité linguistique générale ni une amélioration du débit sur contexte long actif.
- Le prototype refuse aussi une activation mutable TF32 pour le KV encodé. Ce garde reste archivé ici avec l’expérience ; il n’est pas adopté par cette PR.

## Sources et reproduction

La base canonique est conservée par la PR #115. Les trois fichiers Native modifiés sont dans `experimental/native/`. Le checker et le script de préparation originaux sont conservés dans `diagnostics/`, avec leurs chemins locaux à adapter ; les captures utilisent les harnesses de `source/`. Dans un checkout isolé de la base quantifiée canonique, reproduire les trois substitutions Native puis compiler la DLL. Exécuter les captures avec le même CLI et les deux DLL, les formats F32/F16/Q8 et les paramètres du manifeste. Ces scripts de diagnostic ne constituent pas une commande de build du produit livré.

Pour utiliser les harnesses déplacés dans la documentation depuis la racine du checkout : `$env:PYTHONPATH = Join-Path (Get-Location) 'scripts'`. Leurs imports d’origine sont conservés ; les deux commandes `--help` ont été vérifiées avec cette variable.

CLI : `f8b4de53ebf5b9a4d6d621831eb02fe9895c361866c311e230b061bc29bf52a3`. DLL initiale : `c3a661fd2d9f0158f922dfa6b94f918363834024db5f10377e6a17bb814ddf4a`. DLL fusionnée : `3e63739cd27ed8089bb4b2e2970f3b443ecf5e4e75ebb907c640006f090013c4`. Les empreintes des sources compilées sont dans `manifest.json`, celles des captures conservées dans `evidence-index.json`.

Refs [#93](https://github.com/azerothl/Rbitnet/issues/93), [#98](https://github.com/azerothl/Rbitnet/issues/98).
