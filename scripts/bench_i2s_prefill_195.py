#!/usr/bin/env python3
"""Issue #195 A/B bench: I2_S prefill batching vs per-token matvec.

Expects rbitnet-server to be running with the BitNet I2_S GGUF on CPU.
Calibrates two user prompts so the engine sees exactly --short-tokens and
--long-tokens prompt tokens, then runs warmup + N measured greedy requests
per prompt and records per-request prefill_ms (from /metrics counters),
wall ms, completion tokens and the SHA-256 of the greedy text.

Output: a JSON file with one row per request. Compare two runs (before /
after) with compare_i2s_prefill_195.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def post_chat(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url=f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def metric_value(metrics_text: str, name: str) -> int:
    for line in metrics_text.splitlines():
        if line.startswith(name + " "):
            return int(line.split(" ")[1])
    raise KeyError(name)


def calibrate(base_url: str, model: str, base_text: str, target: int, tokenizer) -> str:
    """Build a user message so the engine reports exactly `target` prompt tokens.

    The engine re-tokenizes the templated prompt, so a token prefix of
    base_text does not map 1:1 onto the server-side count and some counts
    are skipped. Probe the server for each candidate text directly; try a
    few leading paddings to shift merge alignment until one lands exactly.
    """
    ids = tokenizer.encode(base_text).ids

    def probe_text(text: str) -> int:
        out = post_chat(
            base_url,
            {
                "model": model,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        return out["usage"]["prompt_tokens"]

    paddings = ["", "Note. ", "Ok. ", "A ", "# ", "1. ", "Q: ", ">> "]
    for pad in paddings:
        def probe(k: int, pad: str = pad) -> int:
            return probe_text(pad + tokenizer.decode(ids[:k], skip_special_tokens=False))

        full = probe(len(ids))
        if full < target:
            continue
        lo, hi = 0, len(ids)
        best_k = None
        while lo <= hi:
            mid = (lo + hi) // 2
            seen = probe(mid)
            if seen == target:
                best_k = mid
                break
            if seen < target:
                lo = mid + 1
            else:
                hi = mid - 1
        if best_k is None:
            for k in range(max(0, hi - 5), min(len(ids), hi + 6)):
                if probe(k) == target:
                    best_k = k
                    break
        if best_k is not None:
            return pad + tokenizer.decode(ids[:best_k], skip_special_tokens=False)
    raise RuntimeError(f"no padding/prefix lands on {target} prompt tokens")


def measure_once(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    before = get(f"{base_url}/metrics")
    t0 = time.perf_counter()
    out = post_chat(
        base_url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    after = get(f"{base_url}/metrics")
    prefill_ms = metric_value(after, "rbitnet_inference_prefill_ms_sum") - metric_value(
        before, "rbitnet_inference_prefill_ms_sum"
    )
    ttft_ms = metric_value(after, "rbitnet_inference_ttft_ms_sum") - metric_value(
        before, "rbitnet_inference_ttft_ms_sum"
    )
    text = out["choices"][0]["message"]["content"] or ""
    return {
        "wall_ms": round(wall_ms, 1),
        "prefill_ms": prefill_ms,
        "ttft_ms": ttft_ms,
        "prompt_tokens": out["usage"]["prompt_tokens"],
        "completion_tokens": out["usage"]["completion_tokens"],
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="rbitnet-bitnet")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--short-tokens", type=int, default=61)
    parser.add_argument("--long-tokens", type=int, default=381)
    parser.add_argument("--base-text-file", required=True)
    parser.add_argument("--tokenizer", default="models/bitnet-b158-2b/tokenizer.json")
    parser.add_argument("--prompts-cache")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_text = open(args.base_text_file, encoding="utf-8-sig").read()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompts: dict[str, str]
    if args.prompts_cache:
        try:
            prompts = json.load(open(args.prompts_cache, encoding="utf-8"))
            if prompts.get("short") and prompts.get("long"):
                print(f"loaded cached prompts from {args.prompts_cache}")
            else:
                prompts = {}
        except (OSError, ValueError):
            prompts = {}
    else:
        prompts = {}

    if not prompts.get("short"):
        print(f"calibrating short prompt to {args.short_tokens} tokens…")
        prompts["short"] = calibrate(args.base_url, args.model, base_text, args.short_tokens, tokenizer)
    if not prompts.get("long"):
        print(f"calibrating long prompt to {args.long_tokens} tokens…")
        prompts["long"] = calibrate(args.base_url, args.model, base_text, args.long_tokens, tokenizer)
    if args.prompts_cache:
        with open(args.prompts_cache, "w", encoding="utf-8") as fh:
            json.dump(prompts, fh, ensure_ascii=False, indent=2)

    rows = []
    for name in ("short", "long"):
        for i in range(args.warmup):
            row = measure_once(args.base_url, args.model, prompts[name], args.max_tokens)
            print(f"[{args.label}] {name} warmup {i + 1}: prefill {row['prefill_ms']} ms")
        for run in range(1, args.runs + 1):
            row = measure_once(args.base_url, args.model, prompts[name], args.max_tokens)
            row.update(label=args.label, fixture=name, run=run)
            rows.append(row)
            decode_ms = row["wall_ms"] - row["prefill_ms"]
            print(
                f"[{args.label}] {name} run {run}: prefill {row['prefill_ms']} ms, "
                f"wall {row['wall_ms']} ms, decode {decode_ms:.0f} ms, "
                f"{row['completion_tokens']} tokens"
            )

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"label": args.label, "rows": rows}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
