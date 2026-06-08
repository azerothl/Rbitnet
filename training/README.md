# Optional fine-tuning recipes (Python)

Rbitnet inference stays **Rust-only**. This folder holds **optional** Hugging Face–style **LoRA / SFT** scripts that produce **Safetensors** checkpoints. You must convert those to **GGUF** with **llama.cpp** (or use a prebuilt GGUF) before pointing `RBITNET_MODEL` at a file.

See **[docs/TRAINING_AND_COMPATIBILITY.md](../docs/TRAINING_AND_COMPATIBILITY.md)** for compatibility rules, **[ml-intern](https://github.com/huggingface/ml-intern)** if you want an agent-driven research/train loop upstream, and **[docs/USAGE.md](../docs/USAGE.md)** for runtime env vars.

## Setup

Use a **virtual environment** (recommended).

```bash
cd training
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

For **GPU**, install a CUDA build of PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then `pip install -r requirements.txt`.

QLoRA (`--int4`) needs **`bitsandbytes`** (Linux/WSL often easiest; Windows support varies).

## Recipe: `recipes/sft_lora.py`

Minimal **instruction / response** JSONL (`instruction`, `output` fields). Example: [`data/example.jsonl`](data/example.jsonl).

From the **repository root** (`Rbitnet/`):

```bash
python training/recipes/sft_lora.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset-jsonl training/data/example.jsonl \
  --output-dir ./out/sft-tiny \
  --max-steps 20 \
  --bf16
```

Or via the Rust CLI (same machine, checkout visible):

```bash
cargo run -p rbitnet-cli -- train \
  --repo-root . \
  --recipe recipes/sft_lora.py \
  -- \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset-jsonl training/data/example.jsonl \
  --output-dir ./out/sft-tiny \
  --max-steps 20 \
  --bf16
```

`RBITNET_REPO_ROOT` defaults to `.` if you omit `--repo-root`.

Outputs:

- `output-dir/adapter/` — LoRA adapter + tokenizer snapshot  
- `output-dir/merged/` — only if `--merge-and-save` (no `--int4`)

## Export to GGUF (not automated here)

Conversion scripts and flags change with each **llama.cpp** release. After you have a merged HF folder (`config.json` + Safetensors) or an adapter you merged yourself:

1. Follow the current **llama.cpp** `convert_hf_to_gguf.py` (or successor) documentation for your architecture.
2. Quantize with the project’s `quantize` tool if you need a smaller `.gguf`.
3. Place `tokenizer.json` (or `tokenizer.model`) beside the GGUF or set `RBITNET_TOKENIZER`.
4. Run `rbitnet serve` / `rbitnet-server` with `RBITNET_MODEL` pointing at the `.gguf`.

Use:

```bash
cargo run -p rbitnet-cli -- export-gguf --checkpoint path/to/hf_checkpoint
```

for a short checklist printed to the terminal.

## CI note

These scripts are **not** executed by default Rust CI; keep them small and documented so users can adapt versions locally.
