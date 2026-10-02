#!/usr/bin/env python3
"""Validate measured spans, archive real results and plot operator time per token."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["cpu", "cuda-default", "cuda-output", "cuda-cpu-attention",
            "cuda-output-cpu-attention", "cuda-dense-output-cpu-attention"]
STAGES = ["qkv", "attn_output", "ffn_gate_up", "ffn_down", "attention", "output"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, records, reference_ids = [], [], None
    for name in VARIANTS:
        record = json.loads((source / f"{name}.json").read_text(encoding="utf-8"))
        samples = record["measurements"]
        assert len(samples) >= 2 and record["format"] == "rbitnet-llama-profile-v1"
        for sample in samples:
            assert sample["completion_tokens"] == len(sample["token_ids"]) == 16
            reference_ids = reference_ids or sample["token_ids"]
            assert sample["token_ids"] == reference_ids, f"greedy drift: {name}"
            for phase, steps in [("prefill", sample["prompt_tokens"]), ("decode", sample["completion_tokens"])]:
                timings = sample[f"{phase}_stages"]
                assert set(timings) == set(STAGES)
                for stage in STAGES:
                    assert timings[stage]["calls"] == steps * (1 if stage == "output" else 16)
                assert sum(t["elapsed_ns"] for t in timings.values()) <= sample[f"{phase}_ms"] * 1e6
                counters = sample[f"{phase}_counters"]
                cpu_calls = 113 if name == "cpu" else (1 if record["output_offload"] == "0" else 0)
                gpu_calls = 512 if record["attention_backend"] == "cuda" else (113 if "dense" in name else 0)
                assert counters["cpu_quant_calls"] == cpu_calls * steps
                assert counters["gpu_cublas_gemv_calls"] == gpu_calls * steps
        median = lambda key: statistics.median(sample[key] for sample in samples)
        row = {"variant": name, "decode_tps_median": median("decode_tps"),
               "decode_ms_per_token": statistics.median(s["decode_ms"] / s["completion_tokens"] for s in samples),
               "prefill_ms_median": median("prefill_ms")}
        for stage in STAGES:
            row[stage + "_ms_per_token"] = statistics.median(
                s["decode_stages"][stage]["elapsed_ns"] / s["completion_tokens"] / 1e6 for s in samples)
        row["other_ms_per_token"] = row["decode_ms_per_token"] - sum(row[s + "_ms_per_token"] for s in STAGES)
        rows.append(row)
        records.append(record)
    benchmark = json.loads(Path("docs/benchmarks/2026-10-03/results.json").read_text(encoding="utf-8"))
    reference = next(r for r in benchmark["rows"] if r["model"] == "llama32-1b" and r["engine"] == "llama.cpp" and r["backend"] == "cpu")
    reference_sample = next(s for s in reference["samples"] if s["fixture"] == "throughput-1")
    assert reference_sample["text"].startswith(records[0]["measurements"][0]["text"])
    result = {"format": "rbitnet-performance-diagnosis-v1", "archived_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": benchmark["environment"], "summary": rows, "records": records,
              "validation": {"same_greedy_ids_all_samples": True,
                             "sample_count": sum(len(r["measurements"]) for r in records),
                             "reference_text_prefix_matches": True,
                             "stage_and_counter_counts_match": True},
              "counter_note": "gpu_cublas_gemv_calls excludes native quant GEMV; native quant copy traffic is not included in perf GPU byte counters.",
              "protocol_note": "Direct real forward, 36 prompt tokens from throughput-1, 16 generated tokens, two repeats, three-token warmup excluded, F32 dense KV/max_seq 8192; includes final unused forward to match current runtime. HTTP, tokenization and sampling excluded; CPU-attention and F32 CUDA variants are diagnostics, not shipped optimizations."}
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    labels = ["CPU actuel", "CUDA actuel", "Sortie sur GPU", "Attention CPU", "Sortie GPU + attention CPU", "cuBLAS F32 + attention CPU"]
    colors = ["#809bbb", "#adc4de", "#39759a", "#255473", "#e6a857", "#b65e50", "#dddddd"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1, 4]}, layout="constrained")
    for ax, indices in zip(axes, [[0], list(range(1, 6))]):
        left = [0.0] * len(indices)
        for stage, color in zip(STAGES + ["other"], colors):
            values = [rows[i][stage + "_ms_per_token"] for i in indices]
            ax.barh(range(len(indices)), values, left=left, label=stage, color=color)
            left = [a + b for a, b in zip(left, values)]
        ax.set_yticks(range(len(indices)), [labels[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlim(0, max(left) * 1.25)
        for y, (i, x) in enumerate(zip(indices, left)):
            ax.text(x + max(left) * .02, y, f"{x:.1f} ms / {rows[i]['decode_tps_median']:.1f} tok/s", va="center", fontsize=9)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Temps par token en millisecondes — échelles CPU et CUDA distinctes")
    fig.suptitle("Rbitnet : où passe le temps ?\nLlama 3.2 1B Q4_K_M · Ryzen 9800X3D / RTX 4080 SUPER · 03/10/2026", fontsize=12)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=7, frameon=False, fontsize=9)
    fig.savefig(output / "operator-times.png", dpi=160)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
