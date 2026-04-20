# Testing Rbitnet with Hugging Face models (e.g. `bitnet_b1_58-large`)

## No Microsoft BitNet required to *run* Rbitnet

**Rbitnet has zero runtime dependency on the Microsoft BitNet repository.** It is pure Rust: point `RBITNET_MODEL` at a `**.gguf*`* file and provide `**tokenizer.json**` (see [USAGE.md](USAGE.md)). No checkout, no BitNet CMake, no `bitnet.cpp`.

What you need is **a file**: a Llama-shaped, GGUF-packaged model. *Where that GGUF comes from* is separate from Rbitnet.

---

## Path 1 — You already have a `.gguf` (recommended if you want zero BitNet)

Use any GGUF produced by **any** toolchain, as long as it is compatible with Rbitnet’s loader (llama metadata + tensor names — see [BITNET_SPEC.md](BITNET_SPEC.md)):

- A **community upload** on Hugging Face (search for `.gguf` in the model files).
- A file built on another machine with **llama.cpp** converters (`convert_hf_to_gguf.py`, etc.) for a standard Llama-family checkpoint.
- A file shared by a colleague or CI artifact.

Then skip straight to **[Validate with Rbitnet](#validate-with-rbitnet)** below. No Python, no BitNet.

---

## Path 1b — List / search / download from Hugging Face (Rust CLI only)

If you have the **`rbitnet`** binary (from a release archive or `cargo build -p rbitnet-cli`), you do **not** need `scripts/setup_env.py` to discover or fetch **`.gguf`** files:

- **`rbitnet models list`** — curated, versioned index (`data/compatible_models.json`); this is the only list the project treats as **recommended**.
- **`rbitnet models search`** — Hub API search filtered to repos that expose at least one **`.gguf`** file. Results are **not** guaranteed compatible with Rbitnet until tested; stderr prints an explicit warning.
- **`rbitnet models download`** — uses the Hugging Face cache and copies into `--dir`. Set **`HF_TOKEN`** (or pass a token flag) for gated repositories.

Repos that contain **only Safetensors** will not appear in search results that require a `.gguf` sibling; you still need an external conversion step to produce GGUF if you only have Safetensors.

See [USAGE.md](USAGE.md) for flags and environment variables.

### Mettre à jour `data/compatible_models.json` sans saisie manuelle

Pour remplir (ou régénérer) le catalogue à partir du Hub : **`rbitnet models generate-catalog`** — même principe que la recherche HF, mais écrit un JSON au schéma `compatible_models.json` (un GGUF « principal » par dépôt + tokenizer si le fichier est listé côté Hub). **Ce n’est pas un test Rbitnet** : relire les entrées, retirer les dépôts douteux, éventuellement ajuster les noms de fichiers, puis commit.

Exemple : `cargo run -p rbitnet-cli --release -- models generate-catalog --output data/compatible_models.json` (requête par défaut `gguf`). Pour cibler une famille : `--query llama --max-inspect 400`.

Pour une liste **100 % exploratoire** sans toucher au dépôt, continue d’utiliser **`rbitnet models search <requête>`**.

---

## Path 2 — Download weights only (no BitNet clone)

If you only need the Hugging Face **Safetensors** tree locally before converting elsewhere:

```bash
pip install -r scripts/requirements-setup.txt
python scripts/setup_env.py download --hf-repo 1bitLLM/bitnet_b1_58-large --model-dir models
```

This uses `**huggingface_hub` only** (listed in `scripts/requirements-setup.txt`). It does **not** clone Microsoft BitNet.

You must still convert Safetensors → GGUF with **some** converter (see next section). Rbitnet does not ship that conversion in Rust.

---

## Path 3 — Converting without Microsoft BitNet (conceptual)

- **Standard Llama / Mistral-style** checkpoints are often convertible with **[llama.cpp](https://github.com/ggml-org/llama.cpp)** `convert_hf_to_gguf.py` (or newer equivalents), then you run Rbitnet on the resulting GGUF.
- **BitNet-specific 1.58-bit** checkpoints (e.g. `1bitLLM/bitnet_b1_58-large`) use custom layouts; practical options are: obtain an **already converted GGUF** from the community, or use Microsoft’s tooling **once** if no alternative exists. Rbitnet itself does not mandate which converter you use.

---

## Path 4 — Microsoft BitNet tooling (optional, only if you choose it)

Use this **only** if you want the upstream `**convert-hf-to-gguf-bitnet.py`** / `llama-quantize` pipeline from [microsoft/BitNet](https://github.com/microsoft/BitNet).

Prerequisites: Python, CMake, Clang (see BitNet README). Clone BitNet with submodules and follow their docs.

Example inside a BitNet clone:

```bash
python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large --model-dir models -q i2_s
```

From the Rbitnet repo you can **delegate** to that script without retyping repo ids:

```bash
export BITNET_ROOT=/path/to/BitNet
python scripts/setup_env.py bitnet-setup --hf-repo 1bitLLM/bitnet_b1_58-large --model-dir models -q i2_s
```

This is **optional** convenience; it does not change Rbitnet’s runtime.

---

## Helper: `scripts/setup_env.py` (no BitNet needed for most commands)


| Command           | Needs BitNet clone?                    |
| ----------------- | -------------------------------------- |
| `list-models`     | No                                     |
| `download`        | No (only `huggingface_hub`)            |
| `env --gguf PATH` | No                                     |
| `doctor`          | No                                     |
| `bitnet-setup`    | **Yes** — runs BitNet’s `setup_env.py` |


```bash
python scripts/setup_env.py env --gguf /path/to/model.gguf
```

---

## Reference: `bitnet_b1_58-large` on Hugging Face

This guide often cites **[1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)**. That repo ships **Safetensors** (FP32). Rbitnet does **not** read Safetensors directly — you need a **GGUF** on disk. The conversion step is upstream of Rbitnet; BitNet’s repo is one possible converter, not a dependency of the Rbitnet binary.

## What works today in Rbitnet

- **Parse** a Llama-compatible GGUF: metadata, tensor table, mmap’d weights.
- **Inference** in pure Rust: dequantize + forward + tokenizer-driven generation (see **[USAGE.md](USAGE.md)**).
- **HTTP server**: OpenAI-shaped API when `RBITNET_MODEL` and tokenizer are set (or stub/toy for tests).

---

## Validate with Rbitnet

**1. Inspect the archive (no server):**

```bash
cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
```

You should see GGUF version, architecture, tensor count, first tensors, and `llama.*` hyperparameters when present.

**2. Point the engine at the file and run the HTTP server:**

```bash
# Linux / macOS
export RBITNET_MODEL=/absolute/path/to/model.gguf
export RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server
```

```powershell
# Windows PowerShell
$env:RBITNET_MODEL="C:\path\to\model.gguf"
$env:RBITNET_BIND="127.0.0.1:8080"
cargo run -p bitnet-server --bin rbitnet-server
```

**3. Check endpoints:**

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rbitnet-llama","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

Expect **200** on chat when the tokenizer is available and weights load; otherwise see error messages from the server. **200** on `/v1/models` if the server started correctly.

## Automated test against a local GGUF (optional)

Set:

```bash
export RBITNET_TEST_GGUF=/path/to/model.gguf
cargo test -p bitnet-core optional_gguf_from_env_smoke -- --nocapture
```

If `RBITNET_TEST_GGUF` is unset, the test **passes without doing I/O** (skipped logic).

## Model card summary (`bitnet_b1_58-large`)

- **HF**: [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
- **Format on HF**: Safetensors, ~729M parameters (FP32 in repo)
- **License**: MIT (see model card)

For BitNet-accurate inference speeds and numerics in *their* stack, compare against **bitnet.cpp** in the Microsoft repo; Rbitnet is a separate pure-Rust implementation (see [GOLDEN_TESTS.md](GOLDEN_TESTS.md)).