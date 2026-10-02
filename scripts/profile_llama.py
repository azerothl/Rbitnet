#!/usr/bin/env python3
"""Sequential real-model ablations with the opt-in profile_llama example."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cuda-lib", required=True)
    parser.add_argument("--binary", default="target/release/examples/profile_llama.exe")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--only", help="Run just one named variant")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"gguf_sha256": digest(args.gguf), "prompts_sha256": digest(args.prompts),
                "tokenizer_sha256": digest(args.tokenizer), "binary_sha256": digest(args.binary),
                "cuda_lib_sha256": digest(args.cuda_lib),
                "source_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                "note": "source_base plus opt-in profiling instrumentation; no production optimization"}
    variants = [("cpu", "cpu", "cpu", "0", "0"),
                ("cuda-default", "cuda", "cuda", "0", "0"),
                ("cuda-output", "cuda", "cuda", "1", "0"),
                ("cuda-cpu-attention", "cuda", "cpu", "0", "0"),
                ("cuda-output-cpu-attention", "cuda", "cpu", "1", "0"),
                ("cuda-dense-output-cpu-attention", "cuda", "cpu", "1", "1")]
    if args.only and args.only not in {v[0] for v in variants}:
        parser.error("Unknown variant")
    for name, weights, attention, output_gpu, dense in variants:
        if args.only and args.only != name:
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith("RBITNET_")}
        env.update({"RBITNET_LLAMA_WEIGHT_MODE": "auto", "RBITNET_QUANT_KERNEL": "auto",
                    "RBITNET_HYBRID_OUTPUT": output_gpu, "RBITNET_CUDA_QUANT_LIB": args.cuda_lib,
                    "RBITNET_PROFILE_CUDA_DENSE": dense,
                    "RBITNET_KV_QUANT": "f32", "RAYON_NUM_THREADS": "16"})
        command = [args.binary, args.gguf, args.prompts, args.tokenizer, weights, attention,
                   str(args.repeats), str(args.tokens)]
        print(f"Measuring {name}", flush=True)
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (output / f"{name}.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"{name} failed: {completed.stderr[-2000:]}")
        result = json.loads(completed.stdout)
        result["provenance"] = metadata
        result["variant"] = name
        (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": name, "decode_tps": [r["decode_tps"] for r in result["measurements"]]}), flush=True)


if __name__ == "__main__":
    main()
