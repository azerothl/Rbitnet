#!/usr/bin/env python3
"""Summarize measured long-prompt rows from benchmark_cache_stack (exclude warmup/control)."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    report = json.loads(args.input.read_text(encoding='utf-8'))
    groups = {}
    for row in report['rows']:
        if row['cycle'] > report['warmup_cycle'] and row['prompt'] < 2:
            groups.setdefault(row['mode'], []).append(row)
    def stats(values):
        return dict(median=statistics.median(values), min=min(values), max=max(values))
    result = dict(warmup_excluded=True, control_prompt_excluded=True, modes={})
    for mode, rows in groups.items():
        keys = dict(prefill_ms='rbitnet_inference_prefill_ms_sum', decode_ms='rbitnet_inference_decode_ms_sum',
                    upload_bytes='rbitnet_core_gpu_upload_bytes_total', download_bytes='rbitnet_core_gpu_download_bytes_total')
        result['modes'][mode] = dict(rows=len(rows), wall_ms=stats([r['wall_ms'] for r in rows]),
            **{key: stats([r['metrics_delta'][name] for r in rows]) for key, name in keys.items()},
            completion_tokens=stats([r['response']['usage']['completion_tokens'] for r in rows]),
            decode_tokens_per_second=stats([1000*r['response']['usage']['completion_tokens']/r['metrics_delta'][keys['decode_ms']] for r in rows]))
    result.update(response_count=len(report['rows']), response_matches=sum(r['matches_baseline'] for r in report['rows']),
                  sse_pairs=len(report['sse']), sse_matches=sum(r['text']==r['sse_text'] and r['done'] for r in report['sse']), memory=report['memory'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
