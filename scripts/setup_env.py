#!/usr/bin/env python3
"""
Rbitnet — optional Hugging Face / env helper.

**Rbitnet does not require Microsoft BitNet at runtime** (pure Rust + GGUF). This script
only helps *before* inference: downloading HF checkpoints, printing shell exports, or
*optionally* invoking BitNet’s repo if you already cloned it for conversion.

It does **not** replace BitNet’s full setup_env.py (kernel codegen, CMake, quant).

Commands:

  * **download** — HF snapshot via huggingface_hub (no BitNet).
  * **env** — print RBITNET_MODEL / tokenizer hints (no BitNet).
  * **list-models**, **doctor** — discovery / checks (no BitNet).
  * **bitnet-setup** — *only if* you set BITNET_ROOT: runs BitNet’s setup_env.py.

Install for download::

    pip install -r scripts/requirements-setup.txt
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Aligned with microsoft/BitNet setup_env.py (subset can grow).
SUPPORTED_HF_MODELS: Dict[str, Dict[str, str]] = {
    "1bitLLM/bitnet_b1_58-large": {"model_name": "bitnet_b1_58-large"},
    "1bitLLM/bitnet_b1_58-3B": {"model_name": "bitnet_b1_58-3B"},
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {
        "model_name": "Llama3-8B-1.58-100B-tokens",
    },
    "tiiuae/Falcon3-7B-Instruct-1.58bit": {
        "model_name": "Falcon3-7B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-7B-1.58bit": {"model_name": "Falcon3-7B-1.58bit"},
    "tiiuae/Falcon3-10B-Instruct-1.58bit": {
        "model_name": "Falcon3-10B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-10B-1.58bit": {"model_name": "Falcon3-10B-1.58bit"},
    "tiiuae/Falcon3-3B-Instruct-1.58bit": {
        "model_name": "Falcon3-3B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-3B-1.58bit": {"model_name": "Falcon3-3B-1.58bit"},
    "tiiuae/Falcon3-1B-Instruct-1.58bit": {
        "model_name": "Falcon3-1B-Instruct-1.58bit",
    },
    "microsoft/BitNet-b1.58-2B-4T": {"model_name": "BitNet-b1.58-2B-4T"},
    "tiiuae/Falcon-E-3B-Instruct": {"model_name": "Falcon-E-3B-Instruct"},
    "tiiuae/Falcon-E-1B-Instruct": {"model_name": "Falcon-E-1B-Instruct"},
    "tiiuae/Falcon-E-3B-Base": {"model_name": "Falcon-E-3B-Base"},
    "tiiuae/Falcon-E-1B-Base": {"model_name": "Falcon-E-1B-Base"},
}


def _model_local_dir(model_dir: Path, hf_repo: str) -> Path:
    name = SUPPORTED_HF_MODELS[hf_repo]["model_name"]
    return model_dir / name


def cmd_list_models(_: argparse.Namespace) -> None:
    print("Supported --hf-repo values (same ids as BitNet setup_env.py):")
    for rid in sorted(SUPPORTED_HF_MODELS.keys()):
        print(f"  {rid}")


def cmd_download(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(
            "Missing dependency: pip install -r scripts/requirements-setup.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    hf_repo = args.hf_repo
    model_dir = Path(args.model_dir).resolve()
    local_dir = _model_local_dir(model_dir, hf_repo)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_repo} -> {local_dir} ...")
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print("Done.")
    print(
        "Next: convert to GGUF using Microsoft BitNet (see docs/MODEL_TESTING.md), "
        "or run: python scripts/setup_env.py bitnet-setup --help"
    )


def cmd_bitnet_setup(args: argparse.Namespace) -> None:
    root = Path(args.bitnet_root).resolve()
    setup_py = root / "setup_env.py"
    if not setup_py.is_file():
        print(f"BitNet setup_env.py not found at {setup_py}", file=sys.stderr)
        raise SystemExit(1)

    model_dir = str(Path(args.model_dir).resolve())
    cmd = [
        sys.executable,
        str(setup_py),
        "--hf-repo",
        args.hf_repo,
        "--model-dir",
        model_dir,
        "-q",
        args.quant_type,
    ]
    if args.quant_embd:
        cmd.append("--quant-embd")
    if args.use_pretuned:
        cmd.append("--use-pretuned")
    if args.log_dir:
        cmd.extend(["--log-dir", args.log_dir])

    print(f"Running BitNet: {' '.join(cmd)} (cwd={root})")
    subprocess.run(cmd, cwd=str(root), check=True)
    print(
        "\nWhen conversion finishes, set RBITNET_MODEL to your *.gguf path, "
        "and ensure tokenizer.json is beside it or use RBITNET_TOKENIZER."
    )
    print("Hint: python scripts/setup_env.py env --gguf /path/to/model.gguf")


def cmd_env(args: argparse.Namespace) -> None:
    gguf = Path(args.gguf).expanduser().resolve()
    if not gguf.is_file():
        print(f"Warning: file not found: {gguf}", file=sys.stderr)

    tok = gguf.parent / "tokenizer.json"

    print("# Bash / zsh")
    print(f'export RBITNET_MODEL="{gguf}"')
    if tok.is_file():
        print(f"# tokenizer.json found: {tok}")
    else:
        print("# export RBITNET_TOKENIZER=/path/to/tokenizer.json")

    print("\n# PowerShell")
    print(f'$env:RBITNET_MODEL="{gguf}"')
    if not tok.is_file():
        print('# $env:RBITNET_TOKENIZER="C:\\path\\to\\tokenizer.json"')


def cmd_doctor(_: argparse.Namespace) -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import huggingface_hub  # noqa: F401

        print("huggingface_hub: OK")
    except ImportError:
        print("huggingface_hub: not installed (needed for: download)")
    bitnet = os.environ.get("BITNET_ROOT")
    if bitnet:
        p = Path(bitnet) / "setup_env.py"
        print(f"BITNET_ROOT={bitnet} -> setup_env.py {'OK' if p.is_file() else 'MISSING'}")
    else:
        print("BITNET_ROOT: unset (optional; use bitnet-setup --bitnet-root PATH)")
    rbn = shutil.which("rbitnet-server")
    if rbn:
        print(f"rbitnet-server in PATH: {rbn}")
    else:
        print("rbitnet-server: not in PATH (cargo install or use target/release/)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rbitnet setup helper (HF download + BitNet wrapper + env hints)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-models", help="Print supported Hugging Face repo ids")
    p_list.set_defaults(func=cmd_list_models)

    p_doc = sub.add_parser("doctor", help="Check Python, huggingface_hub, BITNET_ROOT, rbitnet-server")
    p_doc.set_defaults(func=cmd_doctor)

    p_dl = sub.add_parser(
        "download", help="Download a supported HF repo (requires huggingface_hub)"
    )
    p_dl.add_argument(
        "--hf-repo",
        "-hr",
        required=True,
        choices=sorted(SUPPORTED_HF_MODELS.keys()),
    )
    p_dl.add_argument(
        "--model-dir",
        "-md",
        default="models",
        help="Parent directory; files go under <model-dir>/<model_name>/",
    )
    p_dl.set_defaults(func=cmd_download)

    p_bn = sub.add_parser(
        "bitnet-setup",
        help="Run Microsoft BitNet's setup_env.py (needs BitNet clone + build deps)",
    )
    p_bn.add_argument(
        "--bitnet-root",
        "-br",
        default=os.environ.get("BITNET_ROOT", ""),
        help="Path to microsoft/BitNet clone (or set BITNET_ROOT)",
    )
    p_bn.add_argument(
        "--hf-repo",
        "-hr",
        required=True,
        choices=sorted(SUPPORTED_HF_MODELS.keys()),
    )
    p_bn.add_argument("--model-dir", "-md", default="models")
    p_bn.add_argument(
        "--quant-type",
        "-q",
        default="i2_s",
        help="Passed to BitNet (e.g. i2_s, tl1, tl2 — see BitNet docs / your arch)",
    )
    p_bn.add_argument("--quant-embd", action="store_true")
    p_bn.add_argument("--use-pretuned", "-p", action="store_true")
    p_bn.add_argument("--log-dir", "-ld", default="logs")
    p_bn.set_defaults(func=cmd_bitnet_setup)

    p_env = sub.add_parser(
        "env", help="Print RBITNET_* exports for a .gguf (bash + PowerShell)"
    )
    p_env.add_argument("--gguf", "-g", required=True, help="Path to .gguf file")
    p_env.set_defaults(func=cmd_env)

    args = p.parse_args()
    if args.command == "bitnet-setup" and not str(args.bitnet_root).strip():
        p.error("bitnet-setup requires --bitnet-root or BITNET_ROOT in the environment")

    args.func(args)


if __name__ == "__main__":
    main()
