# BitNet (b1.58) — spécification pour Rbitnet

Ce document fixe les attentes d’interopérabilité avec les modèles issus de [microsoft/BitNet](https://github.com/microsoft/BitNet) et les conversions Hugging Face → GGUF.

## Quantifications GGUF courantes

| Nom BitNet / outil | Rôle |
|--------------------|------|
| `i2_s`, `tl1` | Types utilisés par `setup_env.py` et les scripts de conversion officiels. |
| Poids 1.58-bit | Ternaires {-1, 0, +1} (parfois packés avec scales par bloc). |

Les fichiers `.gguf` décrivent les tenseurs avec un `ggml_type` (`u32`). Rbitnet conserve la valeur brute pour accepter les types étendus ou futurs sans casser le parseur.

## Métadonnées GGUF utiles

Clés usuelles (préfixe `llama.*` ou équivalent selon l’architecture déclarée dans `general.architecture`) :

- `general.architecture` — ex. `llama`
- `llama.context_length` — taille de contexte
- `llama.embedding_length` — dimension des embeddings
- `llama.block_count` — nombre de blocs transformer
- `general.alignment` — alignement des tenseurs (souvent 32)

BitNet peut ajouter des clés propres au layout des poids ; le chargeur les expose via `GgufArchive::metadata`.

## Tenseurs (noms indicatifs)

Les noms exacts suivent le fork llama.cpp / BitNet. Exemples fréquents :

- `token_embd.*`, `blk.N.attn_*`, `blk.N.ffn_*`, `output.*`

Pour la validation golden, on compare couche par couche ou logits finaux avec la sortie de **bitnet.cpp** (référence) sur les mêmes entrées.

## Références

- Rapports et noyaux : dépôt Microsoft BitNet, dossiers `preset_kernels/`, `src/`.
- Format GGUF : [ggml GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
