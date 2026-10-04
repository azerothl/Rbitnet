import unittest
from simulate_expert_cache import simulate


class PlacementReplayTests(unittest.TestCase):
    def test_selected_leases_and_byte_budgets(self):
        rows = [dict(layer=0, selected=[0, 1], group_bytes=5, phase='decode', **{'pass': 1}),
                dict(layer=0, selected=[0, 2], group_bytes=5, phase='decode', **{'pass': 2}),
                dict(layer=1, selected=[0, 1], group_bytes=6, phase='prefill', **{'pass': 3})]
        result = simulate(rows, 10, 'lru')
        self.assertEqual((result['hits'], result['misses'], result['evictions']), (1, 3, 1))
        self.assertEqual(result['upload_bytes'], 15)
        self.assertEqual(result['peak_bytes'], 10)
        self.assertEqual(result['cpu_fallback_layers'], 1)

    def test_stale_layer_order_differs_from_recency_and_frequency(self):
        def row(layer, expert, epoch):
            return dict(layer=layer, selected=[expert], group_bytes=1, phase='decode', **{'pass': epoch})
        rows = [row(8, 0, 1), row(8, 0, 1), row(1, 0, 1), row(4, 0, 2), row(8, 0, 2)]
        self.assertEqual(simulate(rows, 2, 'lru')['misses'], 4)
        self.assertEqual(simulate(rows, 2, 'lfu')['misses'], 3)
        self.assertEqual(simulate(rows, 2, 'least-stale')['misses'], 3)


if __name__ == '__main__': unittest.main()
