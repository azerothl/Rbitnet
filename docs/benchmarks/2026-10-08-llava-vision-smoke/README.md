# LLaVA vision smoke — 2026-10-08 (RTX 4080 SUPER)

Preuve locale pour [#143](https://github.com/azerothl/Rbitnet/issues/143) : réponse greedy différente selon l’image.

## Artefacts

| Fichier | Rôle |
|---|---|
| `mys/ggml_llava-v1.5-7b` `ggml-model-q4_k.gguf` | décodeur Llama |
| `mmproj-model-f16.gguf` | projecteur LLaVA MLP |
| `tokenizer.json` | vocabulaire exporté depuis `tokenizer.ggml.*` du GGUF |
| `red.png` / `blue.png` | 336×336 unicolores |

Machine : Ryzen 7 9800X3D, RTX 4080 SUPER 16 Go, Windows. Binaire : `cargo run -p bitnet-core --release --example vision_smoke`.

## Résultats

Prompt : `What primary color is dominant in this image? Answer with one word.` (`max_tokens=8`, greedy).

| Image | Sortie | Prefill tokens | Wall decode+prefill |
|---|---|---:|---:|
| `red.png` | **Red** | 663 | ~113 s |
| `blue.png` | **Blue** | 663 | ~108 s |

Encode mmproj ~9 s ; chargement modèle ~1 s. `supports_vision=true` une fois le mmproj résolu.

```powershell
$env:RBITNET_MODEL='D:\Rbitnet-benchmark-models\llava-v15-7b\ggml-model-q4_k.gguf'
$env:RBITNET_MMPROJ='D:\Rbitnet-benchmark-models\llava-v15-7b\mmproj-model-f16.gguf'
$env:RBITNET_TOKENIZER='D:\Rbitnet-benchmark-models\llava-v15-7b\tokenizer.json'
$env:RBITNET_MAX_SEQ='1024'
$env:RBITNET_VISION_MAX_TOKENS='8'
cargo run -p bitnet-core --release --example vision_smoke -- red.png
cargo run -p bitnet-core --release --example vision_smoke -- blue.png
```

Hors scope de cette preuve : `/v1` HTTP data-URL (même chemin `complete_with_vision_patches`), Qwen-VL, Gemma 3.
