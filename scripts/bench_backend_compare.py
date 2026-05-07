#!/usr/bin/env python3
"""Compare /v1/chat/completions latency across backends.

This script expects rbitnet-server to be running and reachable.
It sends fixed requests, then reports p50/p95 and approximate tokens/s.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def post_json(url: str, payload: dict, timeout_s: float) -> tuple[float, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    parsed = json.loads(body.decode("utf-8"))
    return elapsed_ms, parsed


def extract_text(response: dict) -> str:
    return (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    )


def run_bench(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    warmup: int,
    runs: int,
    timeout_s: float,
) -> dict:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    for _ in range(warmup):
        post_json(endpoint, payload, timeout_s)
    latencies = []
    tok_s = []
    for _ in range(runs):
        ms, response = post_json(endpoint, payload, timeout_s)
        latencies.append(ms)
        completion = extract_text(response)
        token_est = max(1, len(completion.split()))
        tok_s.append(token_est / (ms / 1000.0))
    return {
        "runs": runs,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": statistics.fmean(latencies),
        "mean_tok_s": statistics.fmean(tok_s),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark rbitnet backend latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="rbitnet-llama")
    parser.add_argument("--prompt", default="Explain what Rust ownership means in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    try:
        result = run_bench(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            warmup=args.warmup,
            runs=args.runs,
            timeout_s=args.timeout_seconds,
        )
    except urllib.error.URLError as exc:
        print(f"Failed to reach server: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Benchmark failed: {exc}")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
