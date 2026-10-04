"""Summarize completed quiet captures without running new inference."""
from pathlib import Path
import hashlib, json, statistics
base = Path('E:/devs/Rbitnet/target/performance-cache/expert-arena-proof')
rows = []
captures = {}
for family in ['gpt-oss-20b', 'glm47-flash']:
    for budget in [512, 8192]:
        path = base / 'quiet' / f'{family}-{budget}' / 'results.json'
        data = json.loads(path.read_text())
        assert (len(data['rows']), len(data['sse']), len(data['stops'])) == (27, 9, 3)
        assert all(r['matches_baseline'] for r in data['rows'])
        assert all(r['matches_baseline'] and r['done'] and r['text'] == r['sse_text'] for r in data['sse'])
        captures[f'{family}-{budget}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        for mode in ['sync', 'async-demand', 'arena-demand']:
            record = dict(model=family, expert_budget_mib=budget, mode=mode,
                          memory=data['memory'][mode], tasks={})
            managed = [r['managed_metrics'] for r in data['rows'] if r['mode'] == mode]
            record['managed'] = {key: max(r[key] for r in managed) for key in [
                'rbitnet_core_cuda_managed_peak_bytes', 'rbitnet_core_cuda_managed_allocations_total']}
            for prompt, task in [(0, 'story'), (1, 'code')]:
                measurements = [r for r in data['rows'] if r['mode'] == mode and r['cycle'] > 0 and r['prompt'] == prompt]
                assert len(measurements) == 2
                assert all(r['metrics_delta']['rbitnet_completion_tokens_total'] == r['response']['usage']['completion_tokens'] for r in measurements)
                rates = [r['metrics_delta']['rbitnet_completion_tokens_total'] * 1000 /
                         r['metrics_delta']['rbitnet_inference_decode_ms_sum'] for r in measurements]
                record['tasks'][task] = dict(median_tps=statistics.median(rates), measured_tps=rates,
                                            wall_ms=[r['wall_ms'] for r in measurements])
            rows.append(record)
summary = dict(capture_sha256=captures, rows=rows,
               protocol='Same CLI/library; one warm cycle plus two measured cycles; 128 tokens, four system notes, ctx2048; all modes demand-only.',
               limits=['Global sampled GPU memory is not process VRAM; OS/driver allocations affect it.',
                       'Arena reduces physical allocation calls, without dynamic compaction or changed expert admission.',
                       'No statistical confidence from two measured cycles; no independent engine-parity claim.',
                       'This is a quiet-capture analysis, not the full numerical/network proof manifest.'])
output = base / 'quiet-analysis.json'
output.write_text(json.dumps(summary, indent=2) + '\n')
for r in rows:
    print(r['model'], r['expert_budget_mib'], r['mode'],
          'story', round(r['tasks']['story']['median_tps'], 2),
          'code', round(r['tasks']['code']['median_tps'], 2),
          'GPU delta MiB', r['memory']['gpu_global_peak_delta_mib'])
