#!/usr/bin/env python3
"""Replay unchanged router traces at identical byte budgets for cache policies."""
import argparse
import hashlib
import gzip
import json
import pathlib
from collections import defaultdict


def simulate(rows, budget, policy):
    entries, clock, occupied = {}, 0, 0
    result = dict(hits=0, misses=0, evictions=0, upload_bytes=0, cpu_fallback_layers=0,
                  collision_misses=0, peak_bytes=0)
    evicted_pass = {}
    phases = defaultdict(lambda: dict(hits=0, misses=0, upload_bytes=0))
    for row in rows:
        selected = row['selected']; size = row['group_bytes']; layer = row['layer']; epoch = row['pass']
        if len(selected) * size > budget:
            result['cpu_fallback_layers'] += 1
            continue
        protected = set()
        for expert in selected:
            clock += 1; key = (layer, expert); phase = phases[row['phase']]
            if key in entries:
                result['hits'] += 1; phase['hits'] += 1
                entries[key].update(touched=clock, frequency=entries[key]['frequency']+1, epoch=epoch)
            else:
                result['misses'] += 1; phase['misses'] += 1
                if evicted_pass.get(key) == epoch: result['collision_misses'] += 1
                while occupied + size > budget:
                    eligible = [k for k in entries if k not in protected]
                    if not eligible: raise ValueError('selected expert group does not fit budget')
                    def rank(k):
                        e = entries[k]
                        if policy == 'lru': return e['touched'], *k, 0
                        if policy == 'lfu': return e['frequency'], e['touched'], *k
                        return int(e['epoch']==epoch), *k, e['touched']
                    victim = min(eligible, key=rank)
                    occupied -= entries.pop(victim)['bytes']; result['evictions'] += 1
                    evicted_pass[victim] = epoch
                entries[key] = dict(bytes=size, touched=clock, frequency=1, epoch=epoch)
                occupied += size; result['upload_bytes'] += size; phase['upload_bytes'] += size
                result['peak_bytes'] = max(result['peak_bytes'], occupied)
            protected.add(key)
    result['phases'] = dict(phases)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=pathlib.Path)
    parser.add_argument('--budgets-mib', type=int, nargs='+', default=[512, 2048, 4096, 8192])
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--events', type=int, help='Replay only the first N router events (for measured-counter validation).')
    args = parser.parse_args()
    payload = gzip.decompress(args.trace.read_bytes()) if args.trace.suffix == '.gz' else args.trace.read_bytes()
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    total = len(rows)
    if args.events is not None:
        if args.events <= 0 or args.events > total: parser.error('--events must be between 1 and trace length')
        rows = rows[:args.events]
    report = {'trace':str(args.trace), 'trace_sha256':hashlib.sha256(payload).hexdigest(),
              'total_router_events':total, 'router_events':len(rows), 'results':[
        {'budget_mib':mib, 'policy':policy, **simulate(rows, mib*1024*1024, policy)}
        for mib in args.budgets_mib for policy in ['lru','lfu','least-stale']]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
