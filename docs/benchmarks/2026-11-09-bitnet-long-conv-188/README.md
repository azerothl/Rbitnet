# BitNet 2.4B long conversation + agent loop — 2026-11-09 (issue #188)

Same host/model/tokenizer as the #195 bench (Ryzen 7 9800X3D, GGUF I2_S
SHA-256 `4221b252…`, CPU backend, greedy, single client, model loaded once,
one short warmup request before the measured runs). Harness:
`scripts/bench_bitnet_long_conv_188.py`.

- `results.json` — binary from `main` (per-token I2_S matvec, post #194).
- `results-pr211.json` — binary with the #211 row-reuse batch kernel.

## Conversation (8 turns, history grows, ctx 256 -> ~1786, 32 tokens/turn)

| Binary | Turn 1 | Turn 8 | Min | >= 20 tok/s every turn | Last within 20% of first |
|---|---:|---:|---:|---|---|
| main | 13.27* | 19.43 | 13.27* | no | yes |
| #211 | 21.53 | 18.28 | 15.56 | **no** | yes (85%) |

*main run had no warmup; the #211 run includes the protocol warmup.

Decode degrades with context length (attention over the growing KV): dips at
turns 4 (15.56) and 7 (16.35) with neighbors near 19-20 — part trend, part
run-to-run noise on a long benching session.

## Agent loop (8 steps, <= 64 tokens each, transcript accumulates)

| Binary | Min tok/s | Step 8 | Step-8 prefill | Pass |
|---|---:|---:|---:|---|
| main | 17.80 | 19.77 | 11.3 s | no |
| #211 | 20.78 | 26.10 | 8.7 s | **yes** |

With the #211 prefill kernel the agent loop clears the issue bar: every step
>= 20.78 tok/s and the 8th prefill (8.7 s at 625 ctx) chains a tool without
waiting a minute.

## Verdict for #188

- Agent loop: **pass** on the #211 binary.
- Long conversation: **still failing** the >= 20 tok/s floor at high context
  (min 15.56 at turn 4; 18.28 at turn 8 / 1786 ctx). The gap is decode
  attention cost as KV grows, not prefill — prefill is what #211 fixed.
  Next lever: decode-time attention cost at long context (KV layout /
  SlimAttention-class work), tracked by #196's decode kernel and the
  attention backlog.

Issue #188 stays open; closing it needs the conversation floor met on every
turn or an explicit decision to revise the bar.
