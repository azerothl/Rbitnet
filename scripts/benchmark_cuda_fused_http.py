#!/usr/bin/env python3
"""Measure fused-on/off CUDA Llama HTTP/SSE serving with client timing samples.

The runner starts a fresh server for each configuration, warms it, releases
concurrent streaming requests through a barrier, and saves every client sample
plus /metrics deltas.  It deliberately fails rather than writing a partial
success as a benchmark result.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


PROMPT = "Explain continuous batching in one short paragraph."
METRICS = (
    "rbitnet_core_gpu_llama_batch_rows_total",
    "rbitnet_core_gpu_llama_batch_waves_total",
    "rbitnet_core_gpu_llama_batch_projections_total",
    "rbitnet_core_gpu_upload_bytes_total",
    "rbitnet_core_gpu_download_bytes_total",
    "rbitnet_core_kv_write_bytes_total",
    "rbitnet_core_kv_physical_pages",
    "rbitnet_core_cuda_managed_live_bytes",
    "rbitnet_core_cuda_managed_peak_bytes",
    "rbitnet_core_cuda_managed_refusals_total",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=18980)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--kv-page-limit", type=int, default=128)
    parser.add_argument(
        "--live-sse-mux",
        action="store_true",
        help=(
            "Measure the opt-in owned CUDA live SSE worker instead of the "
            "default buffered Sarathi bridge. This runs fused-on plus a "
            "fused-off live-worker control and requires observable deltas."
        ),
    )
    parser.add_argument(
        "--capacity-page-limits",
        type=int,
        nargs="*",
        default=[],
        help="Optional page limits probed at concurrency 4 after the matrix.",
    )
    parser.add_argument(
        "--capacity-prompt-words",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Prompt word counts for every capacity page limit. Supply multiple "
            "values to run a prompt-length sweep."
        ),
    )
    parser.add_argument("--prompt", default=PROMPT)
    return parser.parse_args()


def ensure_file(path, name):
    if not path.is_file():
        raise SystemExit(f"{name} is not a readable file: {path}")
    return path.resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def gpu_snapshot():
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    query = "name,driver_version,memory.total,memory.used,memory.free"
    output = run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    return {"available": True, "csv": output}


def request_text(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def metrics(base_url):
    values = {}
    for line in request_text(f"{base_url}/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split(maxsplit=1)
        if "{" not in name:
            try:
                values[name] = float(value)
            except ValueError:
                pass
    return values


def metric_delta(before, after):
    return {name: after.get(name, 0) - before.get(name, 0) for name in METRICS}


def percentile(sorted_values, quantile):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * quantile
    low, high = int(position), min(int(position) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summary(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": min(ordered),
        "p50": percentile(ordered, 0.5),
        "p95": percentile(ordered, 0.95),
        "max": max(ordered),
        "mean": statistics.fmean(ordered),
        "sample_stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def start_server(args, fused, page_limit, log_path):
    env = os.environ.copy()
    env.update(
        {
            "RBITNET_MODEL": str(args.model),
            "RBITNET_TOKENIZER": str(args.tokenizer),
            "RBITNET_CUDA_QUANT_LIB": str(args.library),
            "RBITNET_HOST": "127.0.0.1",
            "RBITNET_PORT": str(args.port),
            "RBITNET_BIND": f"127.0.0.1:{args.port}",
            "RBITNET_BACKEND": "cuda",
            "RBITNET_CUDA_PREFILL": "1",
            "RBITNET_CUDA_KV_FORMAT": "f32",
            "RBITNET_CUDA_KV_PAGE_LIMIT": str(page_limit),
            # The default mode compares the two Sarathi bridge scheduler
            # legs. Live-mux mode deliberately selects its own owned worker,
            # while retaining the Sarathi fused flags as an explicit opt-in.
            "RBITNET_CONTINUOUS_BATCHING": "1",
            "RBITNET_FUSED_MULTI_SEQ": "1" if fused else "0",
            "RBITNET_CUDA_FUSED_DECODE_SLOTS": "8",
            "RBITNET_CUDA_CONTINUOUS": "1" if args.live_sse_mux else "0",
            "RBITNET_CUDA_LIVE_SSE_MUX": "1" if args.live_sse_mux else "0",
            "RBITNET_CUDA_CONTINUOUS_SLOTS": "8",
            "RBITNET_MAX_CONCURRENT": "8",
        }
    )
    log = log_path.open("wb")
    process = subprocess.Popen([str(args.binary)], env=env, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{args.port}"
    for _ in range(180):
        if process.poll() is not None:
            break
        try:
            request_text(f"{base_url}/ready", timeout=1)
            return process, log, base_url, env
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    log.close()
    detail = log_path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(f"server did not become ready:\n{detail}")


def stop_server(process, log):
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)
    log.close()


def stream_request(base_url, body, gate):
    gate.wait()
    started = time.perf_counter_ns()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    events, text, first_ns, previous_ns, itl_ms = 0, [], None, None, []
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                fragment = event.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if fragment:
                    now = time.perf_counter_ns()
                    if first_ns is None:
                        first_ns = now
                    elif previous_ns is not None:
                        itl_ms.append((now - previous_ns) / 1_000_000)
                    previous_ns = now
                    events += 1
                    text.append(fragment)
        ended = time.perf_counter_ns()
    except Exception as error:
        return {"error": repr(error), "started_ns": started}
    if first_ns is None:
        return {"error": "SSE completed without content delta", "started_ns": started}
    return {
        "started_ns": started,
        "ttft_ms": (first_ns - started) / 1_000_000,
        "total_latency_ms": (ended - started) / 1_000_000,
        "itl_ms": itl_ms,
        "content_events": events,
        "text": "".join(text),
    }


def run_wave(base_url, args, concurrency, prompt=None):
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt or args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    gate = threading.Barrier(concurrency)
    wall_start = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(lambda _: stream_request(base_url, body, gate), range(concurrency)))
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000
    failures = [sample for sample in samples if "error" in sample]
    if failures:
        raise RuntimeError(f"{len(failures)} streaming requests failed: {failures}")
    if any(not sample["text"] for sample in samples):
        raise RuntimeError("a streaming request returned empty content")
    if args.live_sse_mux and any(sample["content_events"] < 2 for sample in samples):
        raise RuntimeError(
            "live SSE mux did not expose multiple content deltas per request"
        )
    return {
        "concurrency": concurrency,
        "wall_ms": wall_ms,
        "aggregate_requested_tokens_per_sec": concurrency * args.max_tokens / (wall_ms / 1000),
        "per_request_requested_tokens_per_sec": args.max_tokens / (wall_ms / 1000),
        "samples": samples,
    }


def append_aggregate(wave):
    samples = wave["samples"]
    ttft = [item["ttft_ms"] for item in samples]
    total = [item["total_latency_ms"] for item in samples]
    itl = [value for item in samples for value in item["itl_ms"]]
    wave["aggregate"] = {
        "ttft_ms": summary(ttft),
        "total_latency_ms": summary(total),
        "itl_ms": summary(itl) if itl else None,
        "content_events": [item["content_events"] for item in samples],
    }


def run_matrix_case(base_url, args, concurrency, prompt=None):
    before = metrics(base_url)
    waves = []
    for _ in range(args.warmups):
        run_wave(base_url, args, concurrency, prompt)
    for _ in range(args.repetitions):
        wave = run_wave(base_url, args, concurrency, prompt)
        append_aggregate(wave)
        waves.append(wave)
    after = metrics(base_url)
    return {
        "concurrency": concurrency,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "waves": waves,
        "metrics_delta": metric_delta(before, after),
    }


def execute_configuration(args, fused, page_limit, label):
    log_path = args.output_dir / f"{label}.server.log"
    gpu_before = gpu_snapshot()
    process, log, base_url, environment = start_server(args, fused, page_limit, log_path)
    try:
        return {
            "label": label,
            "fused": fused,
            "page_limit": page_limit,
            "environment": {key: environment[key] for key in sorted(environment) if key.startswith("RBITNET_")},
            "gpu_before": gpu_before,
            "cases": [run_matrix_case(base_url, args, count) for count in args.concurrency],
            "gpu_after": gpu_snapshot(),
            "server_log": log_path.name,
        }
    finally:
        stop_server(process, log)


def capacity_prompt(word_count):
    return " ".join(["capacity"] * word_count)


def execute_capacity_probe(args, page_limit, word_count):
    label = f"capacity-pages-{page_limit}-words-{word_count}"
    log_path = args.output_dir / f"{label}.server.log"
    process = log = None
    try:
        process, log, base_url, environment = start_server(args, True, page_limit, log_path)
        prompt = capacity_prompt(word_count)
        case = run_matrix_case(base_url, args, 4, prompt)
        return {
            "page_limit": page_limit,
            "prompt_words": word_count,
            "prompt_characters": len(prompt),
            "status": "success",
            "environment": {key: environment[key] for key in sorted(environment) if key.startswith("RBITNET_")},
            "case": case,
            "server_log": log_path.name,
        }
    except Exception as error:
        return {
            "page_limit": page_limit,
            "status": "failed",
            "error": repr(error),
            "server_log": log_path.name,
        }
    finally:
        if process is not None and log is not None:
            stop_server(process, log)


def main():
    args = parse_args()
    for name in ("model", "tokenizer", "binary", "library"):
        setattr(args, name, ensure_file(getattr(args, name), name))
    if args.warmups < 0 or args.repetitions < 1 or args.max_tokens < 2:
        raise SystemExit("warmups must be non-negative; repetitions >= 1; max-tokens >= 2")
    if any(value < 1 or value > 8 for value in args.concurrency):
        raise SystemExit("concurrency values must be in 1..8")
    if any(value < 1 for value in args.capacity_page_limits):
        raise SystemExit("capacity page limits must be positive")
    if any(value < 1 for value in args.capacity_prompt_words):
        raise SystemExit("capacity prompt word counts must be positive")
    if args.capacity_page_limits and not args.capacity_prompt_words:
        raise SystemExit(
            "--capacity-page-limits requires --capacity-prompt-words for a prompt-length sweep"
        )
    if args.capacity_prompt_words and not args.capacity_page_limits:
        raise SystemExit("--capacity-prompt-words requires --capacity-page-limits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 2,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "source_revision": run_command(["git", "rev-parse", "HEAD"]),
        "live_sse_mux": args.live_sse_mux,
        "fixture": {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "model": str(args.model),
            "model_sha256": sha256(args.model),
            "tokenizer": str(args.tokenizer),
            "tokenizer_sha256": sha256(args.tokenizer),
            "binary": str(args.binary),
            "binary_sha256": sha256(args.binary),
            "library": str(args.library),
            "library_sha256": sha256(args.library),
        },
        "results": [],
        "capacity_probes": [],
    }
    configurations = [(False, "fused-off"), (True, "fused-on")]
    for fused, label in configurations:
        result = execute_configuration(args, fused, args.kv_page_limit, label)
        report["results"].append(result)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for page_limit in args.capacity_page_limits:
        for word_count in args.capacity_prompt_words:
            report["capacity_probes"].append(
                execute_capacity_probe(args, page_limit, word_count)
            )
            (args.output_dir / "results.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
