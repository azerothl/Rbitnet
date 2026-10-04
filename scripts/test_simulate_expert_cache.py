import unittest
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from simulate_expert_cache import simulate


class PlacementReplayTests(unittest.TestCase):
    def test_invalid_policy_and_negative_budget_are_refused(self):
        with self.assertRaises(ValueError):
            simulate([], 1, 'typo')
        with self.assertRaises(ValueError):
            simulate([], -1, 'lru')

    def test_cli_cached_replays_equal_uncached_and_reuse_only_counters(self):
        rows = [dict(layer=0, selected=[0], group_bytes=1, phase='decode', position=i, **{'pass': i+1}) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trace = root / 'trace.jsonl'
            trace.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            source = pathlib.Path(__file__).with_name('simulate_expert_cache.py')
            command = [sys.executable, str(source), str(trace), '--budgets-mib', '0', '1']
            original = root / 'uncached.json'
            subprocess.run(command + ['--output', str(original)], check=True, capture_output=True)
            cached = root / 'cached.json'
            cache = root / 'replay-cache'
            subprocess.run(command + ['--output', str(cached), '--cache-dir', str(cache)], check=True, capture_output=True)
            self.assertEqual(json.loads(cached.read_text()), json.loads(original.read_text()))
            objects = list(cache.glob('*.json'))
            self.assertEqual(len(objects), 6)
            signatures = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in objects}
            subprocess.run(command + ['--output', str(cached), '--cache-dir', str(cache)], check=True, capture_output=True)
            self.assertEqual(json.loads(cached.read_text()), json.loads(original.read_text()))
            self.assertEqual(signatures, {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in cache.glob('*.json')})

    def test_selected_leases_and_byte_budgets(self):
        rows = [dict(layer=0, selected=[0, 1], group_bytes=5, phase='decode', **{'pass': 1}),
                dict(layer=0, selected=[0, 2], group_bytes=5, phase='decode', **{'pass': 2}),
                dict(layer=1, selected=[0, 1], group_bytes=6, phase='prefill', **{'pass': 3})]
        result = simulate(rows, 10, 'lru')
        self.assertEqual((result['hits'], result['misses'], result['evictions']), (1, 3, 1))
        self.assertEqual(result['upload_bytes'], 15)
        self.assertEqual(result['peak_bytes'], 10)
        self.assertEqual(result['cpu_fallback_layers'], 1)

    def test_later_selected_ready_expert_survives_first_demand_miss(self):
        def row(selected, epoch):
            return dict(layer=0, selected=selected, group_bytes=1, phase='decode', **{'pass': epoch})
        rows = [row([0], 1), row([1], 1), row([1], 1), row([2, 0], 2)]
        for policy in ['lru', 'lfu', 'least-stale']:
            result = simulate(rows, 2, policy)
            self.assertEqual((result['hits'], result['misses'], result['evictions']), (2, 3, 1), policy)
            self.assertEqual(result['upload_bytes'], 3, policy)
            self.assertEqual(result['peak_bytes'], 2, policy)

    def test_stale_layer_order_differs_from_recency_and_frequency(self):
        def row(layer, expert, epoch):
            return dict(layer=layer, selected=[expert], group_bytes=1, phase='decode', **{'pass': epoch})
        rows = [row(8, 0, 1), row(8, 0, 1), row(1, 0, 1), row(4, 0, 2), row(8, 0, 2)]
        self.assertEqual(simulate(rows, 2, 'lru')['misses'], 4)
        self.assertEqual(simulate(rows, 2, 'lfu')['misses'], 3)
        self.assertEqual(simulate(rows, 2, 'least-stale')['misses'], 3)


if __name__ == '__main__': unittest.main()
