# I2_S prefill batching — 2026-11-09 (issue #195)

A/B on the same machine and build flags, BitNet b1.58 2.4B I2_S GGUF
(SHA-256 `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`),
CPU backend, greedy (temperature 0), 32 generated tokens, one warmup, three
measured runs per prompt. Prompts calibrated so the engine sees exactly 61
and 381 prompt tokens (`prompts.json`, built from the 2026-10-08 inventory
fixtures with the shipped tokenizer).

- Host: Ryzen 7 9800X3D (8C/16T), Windows 11, `RBITNET_BACKEND=cpu`,
  default `RBITNET_PREFILL_CHUNK_TOKENS=128`, single client, sequential.
- Harness: `scripts/bench_i2s_prefill_195.py`; per-request prefill from
  `rbitnet_inference_prefill_ms_sum` deltas; raw rows in the `*.json` files.
- Before = branch tip without the batch kernel (per-token `fill_i2s_i8`).
- After = row-reuse batch (`dot_i2s_i8_avx2_x4`, `write_i2s_i8_reused_rows`),
  full pool for prefill tiles, single-token GEMV unchanged (4-worker cap).

## Headline (medians)

| Prompt | Prefill before | Prefill after | Speedup | Decode before | Decode after |
|--------|---------------:|--------------:|--------:|--------------:|-------------:|
| 61 tok  | 905 ms  | 638 ms  | **1.42x** | 1089 ms | 1078 ms |
| 381 tok | 5846 ms | 4390 ms | **1.33x** | 1146 ms | 1119 ms |

Greedy parity: identical `text_sha256` and token counts (61/381 prompt,
32 completion) between before and after on both prompts.

## Batch worker sweep (AVX2 x4, 381-token prompt)

| `RBITNET_I2S_BATCH_THREADS` | 4 | 8 | 16 (default) |
|---|---:|---:|---:|
| Prefill median | 5812 ms | 5185 ms | **4199 ms** |

4 workers shows no gain: the ternary dot is compute-bound at low thread
counts, so weight reuse alone does not help — the win needs parallelism.

## AVX-512 prototype (opt-in `RBITNET_I2S_AVX512=1`): no-go by default

`dot_i2s_i8_avx512` / `_x4` (pre-shuffled activation layout, exact integer
parity with the AVX2 reference, covered by unit tests):

- Single-threaded microbench (n=2560, 4096 rows): single 98 → 74 us,
  x4 over 128 tokens 7376 → 5386 us (**1.37x**).
- End-to-end back-to-back A/B (final binary, 16 batch threads):
  long-prompt prefill 4390 ms (AVX2) vs 4560 ms (zmm) — the 512-bit FPU is
  shared per SMT pair on Zen 5 and 16-thread contention eats the microbench
  lead. **Shipped off by default**; `RBITNET_I2S_AVX512=1` re-enables it.

## Verdict against #195 acceptance criteria

- 2x prefill on the 381-token prompt: **not reached** (1.33x conservative
  same-session median; 1.40x on the best sweep). Remaining time is outside
  the I2_S matvec (attention/RMSNorm path and per-layer serialization).
- Decode regression <= 5%: met (no regression).
- Same greedy outputs and token counts: met.
- Numeric coverage of the new kernel: met (x4 vs reference test, batch vs
  single matvec test, AVX-512 parity tests).

Raw data: `before.json`, `after.json`, `sweep-t{4,8,16}.json`,
`after-avx512*.json`, `zmm-*.json`, `avx512-{off,on}.json`.
