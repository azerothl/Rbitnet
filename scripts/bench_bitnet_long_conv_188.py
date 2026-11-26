#!/usr/bin/env python3
"""Issue #188 closure protocol: BitNet 2.4B long conversation + agent loop.

Run 1: eight successive chat turns against one loaded model, history
accumulating (~200 extra tokens per turn, context ~256 -> ~2048). Every turn
must stay >= min-tok/s and the last turn within 20% of the first.

Run 2: agent-style loop, eight generations of at most 64 tokens each with the
transcript growing; the eighth must stay >= min-tok/s and its prefill must
stay usable.

Greedy (temperature 0), single client, model loaded once for the whole run.
Prefill/decode per turn come from rbitnet_inference_prefill_ms_sum deltas.
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


def metric(metrics_text: str, name: str) -> int:
    for line in metrics_text.splitlines():
        if line.startswith(name + " "):
            return int(line.split(" ")[1])
    raise KeyError(name)


def chat(base_url: str, model: str, messages: list[dict], max_tokens: int):
    before = get(f"{base_url}/metrics")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url=f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    after = get(f"{base_url}/metrics")
    prefill_ms = metric(after, "rbitnet_inference_prefill_ms_sum") - metric(
        before, "rbitnet_inference_prefill_ms_sum"
    )
    usage = out["usage"]
    return out, prefill_ms, wall_ms, usage


def turn_row(tag: str, index: int, out: dict, prefill_ms: int, wall_ms: float, usage: dict) -> dict:
    decode_ms = max(1, wall_ms - prefill_ms)
    text = out["choices"][0]["message"]["content"] or ""
    return {
        "tag": tag,
        "turn": index,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prefill_ms": prefill_ms,
        "wall_ms": round(wall_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "decode_tok_s": round(usage["completion_tokens"] / (decode_ms / 1000.0), 2),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="rbitnet-bitnet-b1-58")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--min-tok-s", type=float, default=20.0)
    parser.add_argument("--chat-max-tokens", type=int, default=32)
    parser.add_argument("--agent-max-tokens", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []

    # Warmup: one short greedy request so the first measured turn does not
    # pay mmap page faults and cold caches (issue protocol measures the short
    # prompt first for the same reason).
    chat(args.base_url, args.model, [{"role": "user", "content": "Warmup."}], 8)

    # --- Run 1: long conversation, history accumulates -----------------
    system = (
        "You are a careful inventory assistant. Continue the inventory lists. "
        "Keep each answer short."
    )
    history: list[dict] = [{"role": "system", "content": system}]
    for turn in range(1, args.turns + 1):
        filler = " ".join(
            f"Item {i}: cedar box, glass lamp, river stone, copper key."
            for i in range((turn - 1) * 11, turn * 11)
        )
        history.append(
            {
                "role": "user",
                "content": f"Series {1000 + turn}. Add these entries and continue: {filler}",
            }
        )
        out, prefill_ms, wall_ms, usage = chat(
            args.base_url, args.model, history, args.chat_max_tokens
        )
        row = turn_row("conversation", turn, out, prefill_ms, wall_ms, usage)
        rows.append(row)
        print(
            f"[conv] turn {turn}: ctx {row['prompt_tokens']} tok, prefill {row['prefill_ms']} ms, "
            f"{row['decode_tok_s']} tok/s"
        )
        history.append(
            {"role": "assistant", "content": out["choices"][0]["message"]["content"] or ""}
        )

    conv = [r for r in rows if r["tag"] == "conversation"]
    first, last = conv[0], conv[-1]
    conv_ok = (
        all(r["decode_tok_s"] >= args.min_tok_s for r in conv)
        and last["decode_tok_s"] >= 0.8 * first["decode_tok_s"]
    )

    # --- Run 2: agent loop, one short instruction per step --------------
    agent_history: list[dict] = [
        {"role": "system", "content": "You are an agent that answers in one short line."}
    ]
    for step in range(1, args.turns + 1):
        agent_history.append(
            {
                "role": "user",
                "content": f"Step {step}: name one object found in a workshop and stop.",
            }
        )
        out, prefill_ms, wall_ms, usage = chat(
            args.base_url, args.model, agent_history, args.agent_max_tokens
        )
        row = turn_row("agent", step, out, prefill_ms, wall_ms, usage)
        rows.append(row)
        print(
            f"[agent] step {row['turn']}: ctx {row['prompt_tokens']} tok, "
            f"prefill {row['prefill_ms']} ms, {row['decode_tok_s']} tok/s"
        )
        agent_history.append(
            {"role": "assistant", "content": out["choices"][0]["message"]["content"] or ""}
        )

    agent_rows = [r for r in rows if r["tag"] == "agent"]
    agent_ok = all(r["decode_tok_s"] >= args.min_tok_s for r in agent_rows)

    summary = {
        "conversation": {
            "turns": len(conv),
            "first_decode_tok_s": first["decode_tok_s"],
            "last_decode_tok_s": last["decode_tok_s"],
            "last_ctx_tokens": last["prompt_tokens"],
            "min_decode_tok_s": min(r["decode_tok_s"] for r in conv),
            "pass": conv_ok,
        },
        "agent": {
            "steps": len(agent_rows),
            "last_decode_tok_s": agent_rows[-1]["decode_tok_s"],
            "last_prefill_ms": agent_rows[-1]["prefill_ms"],
            "min_decode_tok_s": min(r["decode_tok_s"] for r in agent_rows),
            "pass": agent_ok,
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
