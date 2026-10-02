"""Check accounting and strict answer gates independently of live engine tests."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

spec = importlib.util.spec_from_file_location("benchmark_engines", Path(__file__).resolve().parents[1] / "benchmark_engines.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def test_only_completed_final_answer_passes(self):
        self.assertEqual(bench.score("Thinking about Paris", "paris")[0], False)
        self.assertEqual(bench.score("<think>Paris is likely", "paris"), (False, ""))
        text = "Reason about the question.<|end|><|start|>assistant<|channel|>final<|message|>Paris.<|return|>"
        self.assertEqual(bench.score(text, "paris"), (True, "Paris."))
        self.assertEqual(bench.score("<think>Compute 7 times 8.</think>56", "56"), (True, "56"))

    def test_http_token_accounting_and_prompt_mismatch(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.received = body
                payload = json.dumps({"response": "one visible word", "prompt_eval_count": 7,
                    "eval_count": 12, "prompt_eval_duration": 20_000_000,
                    "eval_duration": 240_000_000, "load_duration": 3_000_000}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            server = bench.Server({"ollama_url": f"http://127.0.0.1:{httpd.server_port}"},
                                  {"id": "fixture", "ollama": "fixture"}, "ollama", "cpu", Path("."))
            result = server.request({"prompt": "some text", "token_ids": list(range(8))}, 12)
            self.assertEqual(result["completion_tokens"], 12)
            self.assertEqual(result["decode_ms"], 240)
            self.assertEqual(result["decode_tokens_per_second"], 50)
            self.assertFalse(result["prompt_count_matches"])
            self.assertEqual(httpd.received["options"]["num_gpu"], 0)
            self.assertTrue(httpd.received["raw"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()

    def test_unowned_ollama_model_is_never_unloaded(self):
        server = bench.Server({}, {"id": "fixture", "ollama": "fixture"}, "ollama", "cpu", Path("."))
        # No daemon exists at this unused address. close() must perform no HTTP call.
        server.base = "http://127.0.0.1:1"
        self.assertEqual(server.close(), {})


if __name__ == "__main__":
    unittest.main()
