#!/usr/bin/env python3
"""Sequential, real-model HTTP benchmark; never substitutes stub results for failures.

Requires Python 3.10+, requests and psutil. See docs/BENCHMARKS.md for the manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import psutil
import requests


QUALITY = [
    ("capital", [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "paris"),
    ("arithmetic", [{"role": "user", "content": "What is 7 multiplied by 8? Answer only with the number."}], "56"),
    ("memory", [
        {"role": "user", "content": "Remember the code ORION. Reply OK."},
        {"role": "assistant", "content": "OK."},
        {"role": "user", "content": "What is the code? Answer in one word."},
    ], "orion"),
]


def http(url, body=None, timeout=300):
    response = requests.get(url, timeout=timeout) if body is None else requests.post(url, json=body, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_used_mib():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


class MemorySampler:
    """Whole process tree working set; global GPU delta, not per-process VRAM."""
    def __init__(self, pid=None):
        self.pid = pid
        self.peak_rss = 0
        self.baseline_gpu = gpu_used_mib()
        self.peak_gpu = self.baseline_gpu
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            if self.pid:
                try:
                    parent = psutil.Process(self.pid)
                    processes = [parent] + parent.children(recursive=True)
                    rss = 0
                    for process in processes:
                        try:
                            rss += process.memory_info().rss
                        except psutil.Error:
                            pass
                    self.peak_rss = max(self.peak_rss, rss)
                except psutil.Error:
                    pass
            used = gpu_used_mib()
            if used is not None:
                self.peak_gpu = max(self.peak_gpu or 0, used)
            self.stop.wait(0.75)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)
        return {"peak_process_tree_rss_bytes": self.peak_rss,
                "gpu_global_baseline_mib": self.baseline_gpu,
                "gpu_global_peak_mib": self.peak_gpu,
                "gpu_global_peak_delta_mib": max(0, self.peak_gpu - self.baseline_gpu) if self.peak_gpu is not None and self.baseline_gpu is not None else None}


class Server:
    def __init__(self, config, model, engine, backend, log_dir):
        self.config, self.model, self.engine, self.backend = config, model, engine, backend
        self.proc = None
        self.logs = []
        self.log_path = log_dir / f"{model['id']}-{engine}-{backend}.log"
        self.memory = MemorySampler()
        self.base = config.get("ollama_url", "http://127.0.0.1:11434") if engine == "ollama" else f"http://127.0.0.1:{config.get('port', 18094)}"
        self.command = None
        self.env_overrides = {}
        self.owns_ollama_model = False

    def start(self):
        if self.engine == "ollama":
            loaded = http(self.base + "/api/ps")["models"]
            if loaded:
                raise RuntimeError("Ollama already has a loaded model; unload it before an isolated benchmark.")
            self.owns_ollama_model = True
            self.memory.pid = self.config.get("ollama_pid")
            if not self.memory.pid:
                candidates = [process.info["pid"] for process in psutil.process_iter(["name", "pid", "cmdline"])
                              if (process.info["name"] or "").lower() in ("ollama", "ollama.exe")
                              and "serve" in (process.info["cmdline"] or [])]
                if len(candidates) == 1:
                    self.memory.pid = candidates[0]
            if not self.memory.pid:
                raise RuntimeError("Cannot identify a unique Ollama serve PID; set ollama_pid in the manifest.")
            self.memory.thread.start()
            return
        env = os.environ.copy()
        if self.engine == "llama.cpp":
            binary = self.config["llama_cpu" if self.backend == "cpu" else "llama_gpu"]
            self.command = [binary, "-m", self.model["gguf"], "-ngl", "0" if self.backend == "cpu" else "auto",
                            "-c", str(self.config.get("context", 1024)), "-t", str(self.config.get("threads", 16)),
                            "-tb", str(self.config.get("threads", 16)), "-np", "1", "--host", "127.0.0.1",
                            "--port", str(self.config.get("port", 18094)), "--no-warmup", "-ctk", "f32", "-ctv", "f32"]
            health = "/health"
        else:
            self.env_overrides = {"RBITNET_MODEL": self.model["gguf"], "RBITNET_TOKENIZER": self.model["tokenizer"],
                                 "RBITNET_BACKEND": "cpu" if self.backend == "cpu" else "cuda",
                                 "RBITNET_BIND": "127.0.0.1:" + str(self.config.get("port", 18094)),
                                 "RBITNET_CHAT_TEMPLATE": "{user}", "RBITNET_CHAT_FORMAT": "raw",
                                 "RBITNET_LLAMA_WEIGHT_MODE": "auto", "RBITNET_LLAMA_ENCODE_ADD_SPECIAL": "1",
                                 "RBITNET_STUB": "0", "RBITNET_TOY": "0", "RBITNET_PREFIX_KV": "0",
                                 "RBITNET_KV_POOL": "0", "RBITNET_KV_QUANT": "f32", "RBITNET_MAX_CONCURRENT": "1",
                                 "RBITNET_INFERENCE_TIMEOUT_SECS": "600", "RAYON_NUM_THREADS": str(self.config.get("threads", 16))}
            if self.config.get("cuda_quant_library"):
                self.env_overrides["RBITNET_CUDA_QUANT_LIB"] = self.config["cuda_quant_library"]
            if self.model.get("rbitnet_architecture"):
                self.env_overrides["RBITNET_ARCHITECTURE"] = self.model["rbitnet_architecture"]
            custom_env = self.config.get("rbitnet_env", {})
            if not isinstance(custom_env, dict) or any(not key.startswith("RBITNET_") or not isinstance(value, str) for key, value in custom_env.items()):
                raise ValueError("rbitnet_env must map RBITNET_ option names to strings")
            self.env_overrides.update(custom_env)
            # Avoid inheriting an architecture override or a registry from another experiment.
            for key in list(env):
                if key.startswith("RBITNET_"):
                    del env[key]
            env.update(self.env_overrides)
            self.command = [self.config["rbitnet"], "serve"]
            health = "/ready"
        self.logs = [open(self.log_path, "w", encoding="utf-8")]
        self.proc = subprocess.Popen(self.command, cwd=self.config.get("cwd"), env=env,
                                     stdout=self.logs[0], stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.memory.pid = self.proc.pid
        self.memory.thread.start()
        deadline = time.monotonic() + self.config.get("startup_timeout", 180)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Server exited {self.proc.returncode}: " + self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
            try:
                response = requests.get(self.base + health, timeout=1)
                if self.engine == "rbitnet" and response.status_code == 503 and "LoadFailed" in response.text:
                    raise RuntimeError("Model load failed: " + response.text.strip())
                response.raise_for_status()
                return
            except requests.RequestException:
                time.sleep(0.3)
        raise RuntimeError("Server readiness timed out")

    def close(self):
        memory = self.memory.finish() if self.memory.thread.ident is not None else {}
        if self.engine == "ollama" and self.owns_ollama_model:
            try:
                http(self.base + "/api/generate", {"model": self.model["ollama"], "keep_alive": 0})
                deadline = time.monotonic() + 30
                while http(self.base + "/api/ps")["models"] and time.monotonic() < deadline:
                    time.sleep(0.3)
            except (RuntimeError, requests.RequestException) as error:
                # Preserve the failed measurement (e.g. a missing model) instead
                # of aborting the report while attempting to unload it.
                memory["cleanup_error"] = str(error)
        elif self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for stream in self.logs:
            stream.close()
        return memory

    def metrics(self):
        response = requests.get(self.base + "/metrics", timeout=5)
        response.raise_for_status()
        return {match[0]: float(match[1]) for match in re.findall(r"^(rbitnet_\w+) ([0-9.]+)$", response.text, re.M)}

    def unload_and_preload(self):
        # Ollama reuses prefix KV automatically. A warm model with an empty KV
        # avoids comparing its cached prefill with other engines' uncached prefill.
        http(self.base + "/api/generate", {"model": self.model["ollama"], "keep_alive": 0})
        deadline = time.monotonic() + 30
        while http(self.base + "/api/ps")["models"] and time.monotonic() < deadline:
            time.sleep(0.2)
        http(self.base + "/api/generate", {"model": self.model["ollama"], "stream": False,
             "keep_alive": "5m", "options": self.ollama_options(0)}, timeout=300)

    def ollama_options(self, count):
        options = {"num_predict": count, "temperature": 0, "seed": 0, "top_k": 1, "top_p": 1,
                   "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
                   "num_ctx": self.config.get("context", 1024), "num_thread": self.config.get("threads", 16)}
        if self.backend == "cpu":
            options["num_gpu"] = 0
        return options

    def request(self, fixture, count, stream=False):
        before = self.metrics() if self.engine == "rbitnet" and not stream else {}
        if self.engine == "llama.cpp":
            endpoint = "/completion"
            body = {"prompt": fixture["token_ids"], "n_predict": count, "temperature": 0, "top_k": 1, "top_p": 1,
                    "repeat_penalty": 1, "seed": 0, "cache_prompt": False, "stream": stream}
        elif self.engine == "ollama":
            endpoint = "/api/generate"
            body = {"model": self.model["ollama"], "prompt": fixture["prompt"], "raw": True,
                    "stream": stream, "keep_alive": "5m", "options": self.ollama_options(count)}
        else:
            endpoint = "/v1/chat/completions"
            body = {"model": self.model["id"], "messages": [{"role": "user", "content": fixture["prompt"]}],
                    "max_tokens": count, "temperature": 0, "stream": stream}
        start = time.perf_counter()
        if stream:
            first = None
            fragments = []
            with requests.post(self.base + endpoint, json=body, stream=True, timeout=(10, 600)) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    line = line.decode("utf-8")
                    if not line or line == "data: [DONE]":
                        continue
                    item = json.loads(line[6:] if line.startswith("data: ") else line)
                    if "error" in item:
                        raise RuntimeError(str(item["error"]))
                    piece = (item.get("response", "") + item.get("thinking", "")) if self.engine == "ollama" else (item.get("content", "") if self.engine == "llama.cpp" else item.get("choices", [{}])[0].get("delta", {}).get("content", ""))
                    if piece:
                        if first is None:
                            first = time.perf_counter() - start
                        fragments.append(piece)
            return {"first_content_ms": first * 1000 if first is not None else None,
                    "wall_ms": (time.perf_counter() - start) * 1000, "text": "".join(fragments)}
        raw = http(self.base + endpoint, body, timeout=600)
        wall = (time.perf_counter() - start) * 1000
        if "error" in raw:
            raise RuntimeError(str(raw["error"]))
        if self.engine == "llama.cpp":
            timing = raw["timings"]
            result = {"text": raw["content"], "prompt_tokens": raw["tokens_evaluated"],
                      "completion_tokens": timing["predicted_n"], "prefill_ms": timing["prompt_ms"],
                      "decode_ms": timing["predicted_ms"], "cached_prompt_tokens": raw.get("tokens_cached"),
                      "stopped_eos": raw.get("stop_type") == "eos", "raw_timing": timing}
        elif self.engine == "ollama":
            result = {"text": raw.get("response", ""), "thinking": raw.get("thinking", ""),
                      "prompt_tokens": raw["prompt_eval_count"], "completion_tokens": raw["eval_count"],
                      "prefill_ms": raw["prompt_eval_duration"] / 1e6, "decode_ms": raw["eval_duration"] / 1e6,
                      "load_ms": raw["load_duration"] / 1e6, "cached_prompt_tokens": raw.get("prompt_eval_cached_count"),
                      "done_reason": raw.get("done_reason")}
        else:
            after = self.metrics()
            delta = lambda key: after.get(key, 0) - before.get(key, 0)
            result = {"text": raw["choices"][0]["message"]["content"], "prompt_tokens": raw["usage"]["prompt_tokens"],
                      "completion_tokens": raw["usage"]["completion_tokens"],
                      "prefill_ms": delta("rbitnet_inference_prefill_ms_sum"),
                      "decode_ms": delta("rbitnet_inference_decode_ms_sum"),
                      "engine_ttft_ms": delta("rbitnet_inference_ttft_ms_sum")}
        result.update(wall_ms=wall, expected_prompt_tokens=len(fixture["token_ids"]))
        result["prompt_count_matches"] = result["prompt_tokens"] == result["expected_prompt_tokens"]
        result["decode_tokens_per_second"] = result["completion_tokens"] * 1000 / result["decode_ms"] if result["decode_ms"] > 0 else None
        result["wall_tokens_per_second"] = result["completion_tokens"] * 1000 / wall
        return result


def prepare_fixtures(server, repeats, long_notes=0):
    cases = [(name, messages, "quality", expected) for name, messages, expected in QUALITY]
    for index in range(repeats + 1):
        cases.append((f"throughput-{index}", [{"role": "user", "content":
            f"Case {index}. Write a detailed story of at least 300 words about a robot exploring a forest. Start the story immediately."}], "throughput", None))
    fixtures = []
    for name, messages, kind, expected in cases:
        if long_notes and kind == "throughput":
            notes = 'Tu es un assistant précis. Voici des notes communes à cette conversation.\n' + '\n'.join(
                f'Note {i}: Les villes ont des bibliothèques, des jardins et des musées.' for i in range(long_notes))
            messages = [{"role":"system", "content":notes}, *messages]
        if server.model["architecture"] == "llama":
            prompt = "".join("<|start_header_id|>" + m["role"] + "<|end_header_id|>\n\n" + m["content"] + "<|eot_id|>" for m in messages)
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = http(server.base + "/apply-template", {"messages": messages, "add_generation_prompt": True,
                "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}})["prompt"]
        ids = http(server.base + "/tokenize", {"content": prompt, "add_special": True})["tokens"]
        fixtures.append({"id": name, "kind": kind, "messages": messages, "expected": expected,
                         "prompt": prompt, "token_ids": ids})
    return fixtures


def visible_answer(text):
    # Preserve raw outputs in JSON; score only a completed final answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>", 1)[1]
    elif "<|channel|>analysis" in text or ("<think>" in text and "</think>" not in text):
        return ""
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    return text


def score(text, expected):
    answer = visible_answer(text)
    return bool(re.fullmatch(r"[\s\W]*" + re.escape(expected) + r"[\s\W]*", answer, flags=re.I)), answer


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engines", nargs="+", default=["llama.cpp", "ollama", "rbitnet"], choices=["llama.cpp", "ollama", "rbitnet"])
    parser.add_argument("--backends", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--long-notes", type=int, default=0, help="Add common system notes to throughput prompts; keep quality probes unchanged")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.tokens < 1:
        parser.error("repeats and tokens must be positive")
    if not 0 <= args.long_notes <= 500: parser.error("--long-notes must be between 0 and 500")
    config = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists() else {
        "schema_version": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {"repeats": args.repeats, "max_output_tokens": args.tokens, "quality_max_output_tokens": 256,
                     "stream_probe_tokens": 16, "temperature": 0,
                     "long_note_lines": args.long_notes,
                     "seed": {"llama.cpp": 0, "ollama": 0, "rbitnet": None}, "threads": config.get("threads", 16),
                     "context_requested": config.get("context", 1024),
                     "kv_dtype": {"rbitnet": "f32", "llama.cpp": "f32", "ollama": "default f16"}, "concurrency": 1,
                     "warmup_requests": 1, "ollama_empty_kv_reload_before_performance_requests": True},
        "environment": config.get("environment", {}), "models": [], "rows": []}
    if report["protocol"]["repeats"] != args.repeats or report["protocol"]["max_output_tokens"] != args.tokens:
        parser.error("Resume requires the original repetitions and output-token limit")
    if report["protocol"].get("long_note_lines", 0) != args.long_notes: parser.error("Resume requires the original long-note count")
    report["rows"] = [row for row in report["rows"] if row["status"] != "running"]
    for model in config["models"]:
        if args.models and model["id"] not in args.models:
            continue
        if not any(item["id"] == model["id"] for item in report["models"]):
            emit({"hashing": model["id"]})
            report["models"].append({**model, "sha256": sha256(model["gguf"]), "bytes": Path(model["gguf"]).stat().st_size,
                                     "tokenizer_sha256": sha256(model["tokenizer"])})
        fixture_path = args.output.parent / f"{model['id']}-prompts.json"
        if fixture_path.exists():
            fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
            if any(f.get("long_note_lines", 0) != (args.long_notes if f["kind"] == "throughput" else 0) for f in fixtures):
                parser.error("Fixture directory contains a different long-note protocol; choose a fresh output directory")
        else:
            ref = Server(config, model, "llama.cpp", "cpu", log_dir)
            try:
                ref.start()
                fixtures = prepare_fixtures(ref, args.repeats, args.long_notes)
                for fixture in fixtures: fixture["long_note_lines"] = args.long_notes if fixture["kind"] == "throughput" else 0
                fixture_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
            except (RuntimeError, requests.RequestException, KeyError, ValueError, OSError) as error:
                report.setdefault("fixture_errors", []).append({"model": model["id"], "error": str(error)})
                save_report(args.output, report)
                emit({"fixture_preparation_failed": model["id"], "error": str(error)[:1000]})
                continue
            finally:
                ref.close()
        for backend in args.backends:
            for engine in args.engines:
                if args.resume and any(r["model"] == model["id"] and r["engine"] == engine and r["backend"] == backend for r in report["rows"]):
                    continue
                emit({"starting": [model["id"], engine, backend]})
                row = {"model": model["id"], "engine": engine, "backend": backend, "status": "running"}
                server = Server(config, model, engine, backend, log_dir)
                try:
                    start = time.perf_counter()
                    server.start()
                    row["server_ready_ms"] = (time.perf_counter() - start) * 1000
                    row["command"] = server.command
                    row["environment_overrides"] = server.env_overrides
                    perf = [f for f in fixtures if f["kind"] == "throughput"]
                    row["warmup"] = server.request(perf[0], args.tokens)
                    row["samples"] = []
                    for fixture in perf[1:args.repeats + 1]:
                        if engine == "ollama":
                            server.unload_and_preload()
                        sample = server.request(fixture, args.tokens)
                        sample["fixture"] = fixture["id"]
                        row["samples"].append(sample)
                        emit({"sample": [model["id"], engine, backend, fixture["id"]],
                              "decode_tps": sample["decode_tokens_per_second"], "wall_ms": sample["wall_ms"], "prompt_count_matches": sample["prompt_count_matches"]})
                    if engine == "ollama":
                        server.unload_and_preload()
                    row["stream_probe"] = server.request(perf[0], 16, stream=True)
                    row["quality"] = []
                    for fixture in [f for f in fixtures if f["kind"] == "quality"]:
                        if engine == "ollama":
                            server.unload_and_preload()
                        result = server.request(fixture, 256)
                        passed, answer = score(result["text"], fixture["expected"])
                        row["quality"].append({"fixture": fixture["id"], "strict_answer_pass": passed,
                                               "visible_answer": answer, **result})
                    if engine == "ollama":
                        row["loaded_model"] = http(server.base + "/api/ps")
                    elif engine == "rbitnet":
                        row["loaded_model"] = http(server.base + "/v1/models")
                        row["metrics"] = server.metrics()
                    row["median_decode_tokens_per_second"] = statistics.median(s["decode_tokens_per_second"] for s in row["samples"] if s["decode_tokens_per_second"] is not None)
                    row["median_wall_ms"] = statistics.median(s["wall_ms"] for s in row["samples"])
                    row["quality_passes"] = sum(q["strict_answer_pass"] for q in row["quality"])
                    row["prompt_alignment_verified"] = all(s["prompt_count_matches"] for s in row["samples"])
                    row["status"] = "ok" if row["prompt_alignment_verified"] else "prompt_mismatch"
                except (RuntimeError, requests.RequestException, KeyError, ValueError, OSError) as error:
                    row["status"] = "error"
                    row["error"] = str(error)
                    emit({"failed": [model["id"], engine, backend], "error": str(error)[:1000]})
                finally:
                    row["memory"] = server.close()
                    if "cleanup_error" in row["memory"]:
                        row["status"] = "error"
                        row.setdefault("error", row["memory"]["cleanup_error"])
                    if server.log_path.exists():
                        log = server.log_path.read_text(encoding="utf-8", errors="replace")
                        row["server_log_tail"] = log[-6000:]
                        row["offload_evidence"] = [line for line in log.splitlines()
                                                   if re.search(r"offload.*layers|model buffer size|KV buffer size|tensor overrides to CPU", line)]
                    report["rows"].append(row)
                    save_report(args.output, report)
                emit({"completed": [model["id"], engine, backend], "status": row["status"], "decode_tps": row.get("median_decode_tokens_per_second"), "quality_passes": row.get("quality_passes")})
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_report(args.output, report)


if __name__ == "__main__":
    main()
