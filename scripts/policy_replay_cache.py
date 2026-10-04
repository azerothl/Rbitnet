"""Persist deterministic counter replays, separately from inference timings."""
from pathlib import Path
import hashlib
import json
import os
import time

FIELDS = ('pass', 'position', 'phase', 'layer', 'selected', 'group_bytes')


def canonical_sha(rows):
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps({key: row[key] for key in FIELDS},
                             sort_keys=True, separators=(',', ':'))
        digest.update((encoded + '\n').encode())
    return digest.hexdigest()


class ReplayCache:
    def __init__(self, directory, simulator, source_sha256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.simulator = simulator
        self.source_sha256 = source_sha256
        self.cache_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def replay(self, rows, budget, policy, trace_sha256):
        identity = dict(version=2, cache_source_sha256=self.cache_source_sha256,
                        simulator_sha256=self.source_sha256,
                        canonical_trace_sha256=trace_sha256,
                        budget_bytes=budget, policy=policy)
        identity_bytes = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
        token = hashlib.sha256(identity_bytes).hexdigest()
        path = self.directory / (token + '.json')
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
                result_bytes = json.dumps(cached['result'], sort_keys=True,
                                          separators=(',', ':')).encode()
                if (cached['identity'] == identity
                        and hashlib.sha256(result_bytes).hexdigest() == cached['result_sha256']):
                    return cached['result']
            except (ValueError, KeyError, TypeError):
                pass

        result = self.simulator(rows, budget, policy)
        result_bytes = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
        record = dict(identity=identity, result=result,
                      result_sha256=hashlib.sha256(result_bytes).hexdigest())
        temporary = path.with_name(path.name + f'.{os.getpid()}-{time.time_ns()}.tmp')
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as output:
                output.write(json.dumps(record, indent=2) + '\n')
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return result
