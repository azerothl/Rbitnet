#!/usr/bin/env python3
"""Run Gate A benchmark for CPU then CUDA and output markdown row."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def wait_ready(base_url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_backend(
    repo_root: Path,
    python_exec: str,
    base_url: str,
    model: str,
    runs: int,
    backend: str,
) -> dict:
    env = os.environ.copy()
    env["RBITNET_BACKEND"] = backend
    server_cmd = [
        "cargo",
        "run",
        "-p",
        "bitnet-server",
        "--bin",
        "rbitnet-server",
        "--release",
    ]
    server = subprocess.Popen(
        server_cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_ready(base_url, timeout_s=45.0):
            raise RuntimeError(f"server not ready for backend={backend}")
        bench_cmd = [
            python_exec,
            str(repo_root / "scripts" / "bench_backend_compare.py"),
            "--base-url",
            base_url,
            "--model",
            model,
            "--runs",
            str(runs),
        ]
        out = subprocess.check_output(bench_cmd, cwd=str(repo_root), text=True)
        return json.loads(out)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate A CPU/CUDA benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="rbitnet-llama")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cpu = run_backend(repo_root, args.python, args.base_url, args.model, args.runs, "cpu")
    cuda = run_backend(repo_root, args.python, args.base_url, args.model, args.runs, "cuda")
    ratio = 0.0
    if cpu["mean_tok_s"] > 0.0:
        ratio = cuda["mean_tok_s"] / cpu["mean_tok_s"]

    print("CPU:", json.dumps(cpu, indent=2))
    print("CUDA:", json.dumps(cuda, indent=2))
    print()
    print(
        "| Auto Gate A run | cpu->cuda | "
        f"{cpu['p50_ms']:.1f}->{cuda['p50_ms']:.1f} | "
        f"{cpu['p95_ms']:.1f}->{cuda['p95_ms']:.1f} | "
        f"{cpu['mean_tok_s']:.2f}->{cuda['mean_tok_s']:.2f} | "
        f"x{ratio:.2f} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
