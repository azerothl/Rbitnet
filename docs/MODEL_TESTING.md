# Testing Rbitnet with Hugging Face models (e.g. `bitnet_b1_58-large`)

This guide uses **[1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)** as the reference checkpoint. That repo publishes **Safetensors** (FP32 weights). Rbitnet’s loader expects **GGUF**, which you produce with **Microsoft BitNet**’s conversion tooling—not by pointing Rbitnet at the `.safetensors` file directly.

## What works today

- **Parse** a BitNet- or llama-compatible GGUF: metadata, tensor table, mmap’d weight blob.
- **Inference** in pure Rust: dequantize + Llama-shaped forward + tokenizer-driven generation (see **[USAGE.md](USAGE.md)**).
- **HTTP server**: `GET /v1/models` shows `rbitnet-<architecture>` (e.g. `rbitnet-llama` when `general.architecture` is `llama`); `POST /v1/chat/completions` runs real generation when `RBITNET_MODEL` and a tokenizer are set (use `RBITNET_STUB=1` or `RBITNET_TOY=1` only for smoke tests without weights).

Use this flow to **convert** HF / Safetensors checkpoints to GGUF (Python upstream), then run Rbitnet **without Python** at runtime.

## Prerequisites

- Python 3.9+, CMake, Clang (see [microsoft/BitNet](https://github.com/microsoft/BitNet) README).
- Clone BitNet with submodules:

```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
pip install -r requirements.txt
```

Follow BitNet’s platform notes (Windows: VS + Clang toolchain as documented upstream).

## Option A — `setup_env.py` (recommended)

BitNet’s `setup_env.py` can download a Hugging Face repo and prepare the environment for inference. For the **0.7B** `bitnet_b1_58-large` model, use the repo id **`1bitLLM/bitnet_b1_58-large`** (supported in upstream help where listed).

Example (adjust `--model-dir` and `-q` per [BitNet README](https://github.com/microsoft/BitNet)):

```bash
python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large --model-dir models/bitnet_b1_58-large -q i2_s
```

Alternatively, download Safetensors first:

```bash
huggingface-cli download 1bitLLM/bitnet_b1_58-large --local-dir ./models/bitnet_b1_58-large-src
```

Then use BitNet’s conversion path documented for local checkpoints (e.g. `convert-helper-bitnet.py` / `setup_env.py -md` on the downloaded folder), producing a `*.gguf` under your `models/` tree.

Exact output filenames (e.g. `ggml-model-i2_s.gguf`) depend on BitNet version—check the directory after the run.

## Option B — Quick inspection only (no BitNet build)

If you already have a **BitNet-compatible `.gguf`** from another machine, skip straight to validation below.

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

For BitNet-accurate inference speeds and numerics, compare against **bitnet.cpp** in the Microsoft repo; Rbitnet aims for a pure-Rust path with golden checks (see [GOLDEN_TESTS.md](GOLDEN_TESTS.md)).
