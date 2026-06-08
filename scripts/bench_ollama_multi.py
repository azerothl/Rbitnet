#!/usr/bin/env python3
"""Benchmark several Ollama models with identical prompt/options (JSON lines)."""

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


def chat_once(base: str, model: str, prompt: str, num_predict: int, temperature: float, timeout_s: float) -> tuple[float, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    return elapsed, body


def bench_model(
    base: str,
    model: str,
    prompt: str,
    num_predict: int,
    temperature: float,
    warmup: int,
    runs: int,
    timeout_s: float,
) -> dict:
    for _ in range(warmup):
        chat_once(base, model, prompt, num_predict, temperature, timeout_s)
    latencies: list[float] = []
    tok_per_s: list[float] = []
    eval_counts: list[int] = []
    for _ in range(runs):
        elapsed, body = chat_once(base, model, prompt, num_predict, temperature, timeout_s)
        latencies.append(elapsed)
        eval_count = int(body.get("eval_count") or 0)
        eval_counts.append(eval_count)
        if elapsed > 0 and eval_count > 0:
            tok_per_s.append(eval_count / elapsed)
    mean_eval = int(statistics.fmean(eval_counts)) if eval_counts else 0
    return {
        "model": model,
        "runs": runs,
        "mean_eval_tokens": mean_eval,
        "p50_wall_s": percentile(latencies, 0.50),
        "p95_wall_s": percentile(latencies, 0.95),
        "mean_wall_s": statistics.fmean(latencies),
        "mean_tokens_per_s_wall": statistics.fmean(tok_per_s) if tok_per_s else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-model Ollama benchmark.")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument(
        "--models",
        nargs="+",
        default=["qwen3:0.6b", "mistral:7b", "llama3.2:latest"],
    )
    p.add_argument(
        "--prompt",
        default=(
            "Explique la différence entre heap et stack en Rust en deux phrases courtes, "
            "sans exemple de code."
        ),
    )
    p.add_argument("--num-predict", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--timeout-seconds", type=float, default=900.0)
    args = p.parse_args()

    rows = []
    for m in args.models:
        try:
            row = bench_model(
                args.base_url,
                m,
                args.prompt,
                args.num_predict,
                args.temperature,
                args.warmup,
                args.runs,
                args.timeout_seconds,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        except urllib.error.HTTPError as exc:
            print(json.dumps({"model": m, "error": f"HTTP {exc.code}: {exc.reason}"}))
            return 1
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"model": m, "error": str(exc)}))
            return 1

    print(json.dumps({"summary": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
