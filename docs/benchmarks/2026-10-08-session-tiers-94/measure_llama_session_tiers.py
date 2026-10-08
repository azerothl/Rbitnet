"""Measure Llama Native CUDA prefix-checkpoint TTFT tiers over real HTTP.

This deliberately excludes server startup/model reload from TTFT: every server is
healthy before the timed request.  A fresh process is used for every SSD sample
so its first matching lookup must deserialize the sealed checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results.json"
RAW = Path(__file__).resolve().parent / "raw"
PORT = 18194
URL = f"http://127.0.0.1:{PORT}"
MODEL = Path(r"E:\devs\Rbitnet\models\exported-llama\Llama-3.2-1B-Instruct-Q4_K_M.gguf")
TOKENIZER = MODEL.with_name("tokenizer.json")
SERVER = Path(r"D:\rbitnet-build-session-main\release\rbitnet-server.exe")
CUDA_LIBRARY = ROOT / "native" / "cuda_quant" / "build" / "rbitnet_cuda_quant64.dll"
REPEATS = 3

PROMPT = " ".join(["Explain deterministic restoration of model state clearly."] * 80)
PAYLOAD = {
    "model": "rbitnet",
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0,
    "max_tokens": 2,
    "stream": False,
}
WARMUP = {
    "model": "rbitnet",
    "messages": [{"role": "user", "content": "Reply with READY."}],
    "temperature": 0,
    "max_tokens": 2,
    "stream": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(text: str) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in re.findall(r"^([a-zA-Z0-9_]+) ([0-9]+)$", text, re.MULTILINE)
    }


def request(payload: dict) -> tuple[dict, float]:
    encoded = json.dumps(payload).encode()
    started = time.perf_counter()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{URL}/v1/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        ),
        timeout=600,
    ) as response:
        body = json.load(response)
    return body, (time.perf_counter() - started) * 1000


def metrics() -> dict[str, int]:
    with urllib.request.urlopen(f"{URL}/metrics", timeout=10) as response:
        return metric_values(response.read().decode())


def start(name: str, tiers: bool, cache: Path | None) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.update(
        {
            "RBITNET_BIND": f"127.0.0.1:{PORT}",
            "RBITNET_BACKEND": "cuda",
            "RBITNET_MODEL": str(MODEL),
            "RBITNET_TOKENIZER": str(TOKENIZER),
            "RBITNET_CUDA_QUANT_LIB": str(CUDA_LIBRARY),
            "RBITNET_CUDA_RESIDENT_GRAPH": "1",
            "RUST_LOG": "info",
        }
    )
    if tiers:
        assert cache is not None
        environment.update(
            {
                "RBITNET_CONTEXT_TIERS": "1",
                "RBITNET_CONTEXT_DIR": str(cache),
                "RBITNET_CONTEXT_RAM_MB": "512",
                "RBITNET_CONTEXT_DISK_MB": "2048",
            }
        )
    else:
        environment.pop("RBITNET_CONTEXT_TIERS", None)
        environment.pop("RBITNET_CONTEXT_DIR", None)
    stdout = (RAW / f"{name}.stdout.log").open("w")
    stderr = (RAW / f"{name}.stderr.log").open("w")
    process = subprocess.Popen(
        [str(SERVER)],
        cwd=ROOT,
        env=environment,
        stdout=stdout,
        stderr=stderr,
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name}: server exited {process.returncode}")
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=1) as response:
                if response.read().strip() == b"ok":
                    return process
        except OSError:
            time.sleep(0.1)
    process.kill()
    raise TimeoutError(f"{name}: health check timed out")


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def sample(name: str) -> dict:
    before = metrics()
    body, wall_ms = request(PAYLOAD)
    after = metrics()
    data = {
        "name": name,
        "wall_ms": round(wall_ms, 3),
        "text": body["choices"][0]["message"]["content"],
        "finish_reason": body["choices"][0]["finish_reason"],
        "usage": body["usage"],
        "metrics": {
            key: value - before.get(key, 0)
            for key, value in after.items()
        },
        "metrics_after": {
            key: after.get(key, 0)
            for key in (
                "rbitnet_process_rss_bytes",
                "rbitnet_core_cuda_managed_live_bytes",
                "rbitnet_core_cuda_managed_peak_bytes",
            )
        },
    }
    (RAW / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def summary(samples: list[dict]) -> dict:
    ttft = [item["metrics"]["rbitnet_inference_ttft_ms_sum"] for item in samples]
    wall = [item["wall_ms"] for item in samples]
    return {
        "n": len(samples),
        "ttft_ms": ttft,
        "ttft_median_ms": statistics.median(ttft),
        "ttft_min_ms": min(ttft),
        "ttft_max_ms": max(ttft),
        "ttft_sample_stdev_ms": round(statistics.stdev(ttft), 3),
        "wall_ms": wall,
        "wall_median_ms": round(statistics.median(wall), 3),
    }


def main() -> None:
    for required in (SERVER, MODEL, TOKENIZER, CUDA_LIBRARY):
        if not required.is_file():
            raise FileNotFoundError(required)
    RAW.mkdir(exist_ok=True)
    cache = RAW / "context-cache"
    shutil.rmtree(cache, ignore_errors=True)

    # Cold recompute: warmed model, tiers entirely disabled.
    cold_server = start("cold", tiers=False, cache=None)
    try:
        request(WARMUP)
        cold = [sample(f"cold-{index}") for index in range(REPEATS)]
    finally:
        stop(cold_server)

    # RAM restore: seed once, then use the same live process.
    ram_server = start("ram", tiers=True, cache=cache)
    try:
        seed, _ = request(PAYLOAD)
        ram = [sample(f"ram-{index}") for index in range(REPEATS)]
    finally:
        stop(ram_server)

    # SSD restore: preserve the sealed seed, then use a new process per sample.
    ssd = []
    for index in range(REPEATS):
        server = start(f"ssd-{index}", tiers=True, cache=cache)
        try:
            ssd.append(sample(f"ssd-{index}"))
        finally:
            stop(server)

    outputs = [item["text"] for item in cold + ram + ssd]
    if len(set(outputs)) != 1 or seed["choices"][0]["message"]["content"] != outputs[0]:
        raise AssertionError(f"tier output mismatch: {outputs!r}")
    for item in ram:
        if item["metrics"]["rbitnet_core_context_ram_hits_total"] < 1:
            raise AssertionError(f"missing RAM hit: {item['name']}")
    for item in ssd:
        if item["metrics"]["rbitnet_core_context_disk_hits_total"] != 1:
            raise AssertionError(f"missing SSD hit: {item['name']}")

    result = {
        "protocol": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "server_sha256": sha256(SERVER),
            "model": str(MODEL),
            "model_sha256": sha256(MODEL),
            "tokenizer": str(TOKENIZER),
            "tokenizer_sha256": sha256(TOKENIZER),
            "cuda_library": str(CUDA_LIBRARY),
            "cuda_library_sha256": sha256(CUDA_LIBRARY),
            "backend": "cuda",
            "prompt_tokens": cold[0]["usage"]["prompt_tokens"],
            "completion_tokens": cold[0]["usage"]["completion_tokens"],
            "repetitions": REPEATS,
            "ttft_scope": "server already healthy; excludes server startup/model load",
            "sealed_checkpoint_bytes": sum(
                file.stat().st_size for file in cache.rglob("*.state")
            ),
        },
        "cold_recompute": {"summary": summary(cold), "samples": cold},
        "ram_restore": {"summary": summary(ram), "samples": ram},
        "ssd_restore": {"summary": summary(ssd), "samples": ssd},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["summary"] for key, value in result.items() if key != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
