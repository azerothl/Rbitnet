"""Run only as the third successor stage after preceding serial hardware proofs."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

from atomic_journal import write_json

root = Path.cwd()
base = root / 'target/performance-cache'
out = base / 'expert-arena-proof'
out.mkdir(exist_ok=True)
state = json.loads((base / 'serial-experiments.json').read_text(encoding='utf-8'))
post = json.loads((base / 'post-serial-experiments.json').read_text(encoding='utf-8'))
assert state['complete'] and all(row['status'] == 'passed' for row in state['stages'])
assert len(post['stages']) >= 3 and all(row['status'] == 'passed' for row in post['stages'][:2])
assert post['stages'][2]['script'] == Path(__file__).name and post['stages'][2]['pid'] == os.getpid()
production = json.loads((root / 'target/async-expert-cache/production-proof/manifest.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
library = root / 'target/paged-kv/cuda/rbitnet_cuda_quant64.dll'
assert sha(library) == production['library_sha256']
assert all(sha(root / name) == digest for name, digest in production['native_source_sha256'].items())
env = {key: value for key, value in os.environ.items() if not key.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', RAYON_NUM_THREADS='16',
           RBITNET_CUDA='0', CARGO_INCREMENTAL='0', CARGO_TARGET_DIR=str(base / 'check-build'))


def run(name, command, cwd=root, extra=None, marker=None):
    print('Private expert arena:', name, flush=True)
    path = out / (name + '.log')
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=cwd, env=env | (extra or {}), stdout=log,
                                stderr=subprocess.STDOUT, timeout=7200)
    text = path.read_text(encoding='utf-8')
    assert result.returncode == 0, (name, result.returncode, text[-6000:])
    if marker:
        assert marker in text and 'running 0 tests' not in text, (name, marker, text[-3000:])


run('prepare', [sys.executable, str(base / 'prepare_expert_arena.py')])
run('prepare-harness', [sys.executable, str(base / 'prepare_expert_arena_harness.py')])
crate = base / 'check-expert-arena'
run('check', ['cargo', 'check', '--workspace', '--all-targets'], crate)
run('clippy', ['cargo', 'clippy', '--workspace', '--all-targets'], crate)
run('workspace', ['cargo', 'test', '--workspace', '--', '--test-threads=1'], crate)
env['CARGO_TARGET_DIR'] = str(base / 'check-release')
gpu = dict(RBITNET_CUDA_QUANT_LIB=str(library), RBITNET_CUDA_DEVICE_BUDGET_MB='12288',
           RBITNET_CUDA_DEVICE_MARGIN_MB='256', RBITNET_EXPERT_ARENA_TEST='1',
           RBITNET_MOE_ARENA='1', RBITNET_MOE_EXECUTION='cache')
run('actual-owner', ['cargo', 'test', '-p', 'bitnet-core', '--release', '--lib',
                    'optional_actual_expert_arena_views', '--', '--nocapture', '--test-threads=1'], crate, gpu,
    'EXPERT_ARENA_OWNER_DONE physical_allocations=1 groups=4 views=12')
models = [('gpt-oss-20b', 'gptoss', 'gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf', 'gpt-oss-20b'),
          ('glm47-flash', 'mla', 'GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf', 'GLM-4.7-Flash')]
for name, family, relative, tokenizer in models:
    gguf = str(Path('D:/Rbitnet-benchmark-models') / relative)
    run('actual-cache-' + name, ['cargo', 'test', '-p', 'bitnet-core', '--release', '--lib',
        'optional_actual_expert_arena_cache', '--', '--nocapture', '--test-threads=1'], crate,
        gpu | dict(RBITNET_TEST_GGUF=gguf), 'physical_allocations=1 refill_allocations=0 poison_and_last_lease=true')
    for budget in ['512', '8192']:
        actual = gpu | dict(RBITNET_MOE_POLICY_TEST='1', RBITNET_MOE_POLICY_FAMILY=family,
            RBITNET_MOE_POLICY_GGUF=gguf, RBITNET_MOE_POLICY_CACHE=budget,
            RBITNET_MOE_POLICY_TOKENIZER=str(root / 'target/engine-benchmark/tokenizers' / tokenizer / 'tokenizer.json'),
            RBITNET_MOE_REQUIRE_PREFETCH='1' if budget == '512' else '0')
        run('actual-runtime-' + name + '-' + budget,
            ['cargo', 'test', '-p', 'bitnet-core', '--release', '--lib',
             'real_moe_async_cache_preserves_logits_generations_prefix_and_model_lifetime', '--', '--nocapture', '--test-threads=1'],
            crate, actual, 'MoE async real worst KL=')
run('release', ['cargo', 'build', '-p', 'rbitnet-cli', '--release'], crate)
binary = out / 'rbitnet.exe'
shutil.copy2(base / 'check-release/release/rbitnet.exe', binary)
for name, _, _, _ in models:
    options = ['--gpt-full', '--gpt-segmented'] if name == 'gpt-oss-20b' else ['--mla-full', '--split-kv']
    for budget in ['512', '8192']:
        dest = out / 'quiet' / (name + '-' + budget)
        run('quiet-' + name + '-' + budget, [sys.executable, str(base / 'expert-arena-harness/benchmark.py'),
            '--config', 'docs/benchmarks/2026-10-03-parity-round2/manifest.json', '--binary', str(binary), '--library', str(library),
            '--backend', 'gpu', '--model', name, '--async', *options, '--moe-cache', budget, '--cycles', '3', '--notes', '4',
            '--max-tokens', '128', '--device-mib', '12288', '--port', '18138', '--output-dir', str(dest)])
        capture = json.loads((dest / 'results.json').read_text(encoding='utf-8'))
        assert len(capture['rows']) == 27 and len(capture['sse']) == 9 and len(capture['stops']) == 3
        assert all(row['matches_baseline'] for row in capture['rows'])
        assert all(row['matches_baseline'] and row['done'] and row['text'] == row['sse_text'] for row in capture['sse'])
        assert {row['mode'] for row in capture['rows']} == {'sync', 'async-demand', 'arena-demand'}
        for row in capture['rows']:
            assert row['env']['RBITNET_MOE_PREFETCH'] == 'off'
            assert row['env']['RBITNET_MOE_ARENA'] == ('1' if row['mode'] == 'arena-demand' else '0')
    run('live-' + name, [sys.executable, str(base / 'expert-arena-harness/live.py'),
        '--config', 'docs/benchmarks/2026-10-03-parity-round2/manifest.json', '--binary', str(binary), '--library', str(library),
        '--async', '--split-kv', *options, '--moe-cache', '512', '--device-mib', '12288', '--port', '18138',
        '--output-dir', str(out / 'live' / name)], marker='stop and concurrency passed')
prepared = json.loads((crate / 'arena-prepared.json').read_text(encoding='utf-8'))
assert all(sha(crate / name) == digest for name, digest in prepared['source_sha256'].items())
write_json(out / 'manifest.json', dict(binary_sha256=sha(binary), library_sha256=sha(library), prepared=prepared,
    harness_sha256={p.name: sha(p) for p in (base / 'expert-arena-harness').glob('*.py')},
    scope='Private bounded asynchronous expert arena, one physical allocation with leased disjoint views. No Native kernel modification.',
    limits=['No production adoption or claim of faster steady-state decode before assessing the quiet ablation.',
           'Retaining one external view keeps the whole arena alive; physical padding counts toward the budget.',
           'Arena default disabled; currently restricted to the asynchronous GPT-OSS/GLM cache.',
           'No dynamic compaction; most useful expected benefit is startup allocation overhead.']))
print('Private expert arena passed actual single-allocation, mixed-format refill, poison/last-lease, full-model numerical and quiet/network suites; adoption and performance decision pending.', flush=True)
