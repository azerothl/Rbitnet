# Tests golden (non-régression numérique)

## Objectif

Garantir que les noyaux Rust et, plus tard, le graphe complet, restent alignés sur la **référence bitnet.cpp** (ou le script Python d’inférence BitNet).

## Pipeline recommandé

1. Choisir un modèle petit (ex. dummy généré par `utils/generate-dummy-bitnet-model.py` dans BitNet) ou un GGUF officiel léger.
2. Fixer une **graine**, un **prompt** et une **longueur** de génération.
3. Exporter depuis la référence :
   - soit les **logits** du premier pas (fichier binaire ou JSON limité),
   - soit le texte généré complet pour un test bout-en-bout.
4. Placer les artefacts attendus sous `tests/data/golden/` (ou générés en CI à partir d’un cache).
5. Les tests Rust dans `crates/bitnet-core/tests/` chargent ces fichiers et comparent avec une tolérance (`eps` sur floats).

## État dans Rbitnet

- Des tests **sans fichier externe** vérifient déjà les noyaux de référence (matvec ternaire) avec des valeurs attendues embarquées.
- L’extension vers des vecteurs exportés depuis bitnet.cpp est documentée ici pour les itérations suivantes.

## Tolérance

- Opérations FP32 : `1e-5` relatif ou absolu selon l’accumulation.
- Poids quantifiés : comparer plutôt les entiers déquantifiés ou les logits avec tolérance plus large au début.
