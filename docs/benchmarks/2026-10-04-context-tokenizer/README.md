# Capacité allouée et tokenizers immuables — 4 octobre 2026

La capacité annoncée est désormais celle du runtime réellement configuré, distincte du contexte d'entraînement GGUF. Elle est figée avant le chargement paresseux des poids. Les requêtes `prompt + max_tokens` dépassant cette capacité reçoivent une erreur HTTP 400 JSON avant l'ouverture du SSE, sur chat/completions OpenAI et messages Anthropic. Les valeurs nulles, négatives, non numériques et les débordements sont refusés.

Les executors chargent une seule instance immuable du tokenizer partagée par `Arc` : comptage et génération utilisent le même propriétaire, même si le fichier est déplacé après chargement. Un nouveau chargement relit et valide son propre fichier. Cela retire les reparsings de comptage et les divergences avant l'initialisation des poids ; aucun gain de tok/s n'est revendiqué pour ce lot.

Le codec `tokenizer.model` conserve les octets protobuf, le normalizer original, les champs inconnus et la correspondance du vocabulaire GGUF. Il respecte les politiques BOS/EOS du modèle et préserve les contrôles visibles au milieu d'une conversation. La copie de décodage visible change seulement le type des pièces CONTROL en USER_DEFINED. Les modèles WORD qui ne peuvent reconnaître les contrôles intérieurs sont refusés explicitement. Les modèles, IDs, types et dimensions incompatibles sont refusés.

## Validation observée sur les sources intégrées

- 282 tests workspace réussis, 1 ignoré(s), Clippy et CLI release réussis ; logs conservés dans `production/`. Les avertissements préexistants restent visibles.
- Huit tests SentencePiece exécutés explicitement en release. Mistral-7B-Instruct-v0.1 Q4_K_M réel, 32 000 pièces vérifiées, CPU puis CUDA ; par backend, six générations HF de référence, douze générations/replays SP sur les mêmes IDs et six générations/replays multi-tours SP. Paris est présent et aucun caractère de remplacement n'est admis.
- Modèle SentencePiece entraîné upstream de 1 000 pièces : IDs connus, normalisation, quatre politiques BOS/EOS et contrôles intérieurs confrontés au processeur original. Les tests sans fixtures externes sont également conservés.
- Serveur réseau Mistral CPU et CUDA : capacité 64, six refus par backend sur les trois routes en stream/non-stream, frontière exacte, déplacement du fichier avant la première génération, réponse unary/SSE identique et quatre requêtes concurrentes sérialisées identiques. Métadonnées, réponses et métriques brutes : `live/results.json`.
- Petit GGUF Llama F32 généré par le test serveur : capacité 16 figée, changement d'environnement à 4 avant la première allocation, dépassement puis reprise, nouveau chargement à 4 et refus du tokenizer remplacé par un fichier invalide. Ce modèle sert à vérifier la capacité, pas à mesurer les performances.

## Limites et échecs conservés

Le JSON HF converti de Mistral applique une politique différente sur certains espaces initiaux (`Metaspace` prepend-first contre dummy-prefix SentencePiece). Deux différences du corpus sont explicitement rapportées ; on ne force pas une égalité artificielle avec ce JSON lorsque les IDs diffèrent. Les générations HF/SP sont comparées sur les entrées dont les IDs concordent, tandis que la normalisation SP est confrontée au processeur original. L'hypothèse initiale erronée est conservée dans les logs du prototype.

Le premier essai réseau demandait quatre appels concurrents avec admission serveur par défaut à un : les 503 étaient le comportement attendu de cette configuration. L'essai final fixe explicitement l'admission à quatre ; le moteur exécute encore les requêtes une par une. Ce lot ne constitue pas un forward GPU multi-séquences ni du continuous batching. Le batching et le draft GGUF restent dans #96 et #97 ; les autres capacités du ticket #24 ne sont pas déclarées terminées.

## Reproduction et provenance

`cargo test --workspace -- --test-threads=1`, `cargo clippy --workspace --all-targets` et `cargo build --release -p rbitnet-cli` sont les vérifications ordinaires. Les scripts de `reproduction/` documentent les variables des tests Mistral, du modèle entraîné, de la DLL CUDA et les routes réseau. Les fichiers GGUF/tokenizers restent locaux ; leurs révisions et empreintes figurent dans les manifestes. Les scripts reflètent l'arborescence de travail et utilisent `target/performance-cache` pour les preuves.

Le manifeste de production lie les sources Rust, le CLI et la DLL utilisée. La suite réseau choisit exclusivement une DLL dont le SHA-256 correspond à ce manifeste. Les captures publiées normalisent seulement BOM/CRLF ; `manifest.json` conserve aussi leurs empreintes originales. Les résultats du prototype sont séparés des résultats intégrés.
