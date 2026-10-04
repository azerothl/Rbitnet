"""Actual serving, RAM eviction, process restart and corrupt-object replay.

Run only from the serialized context owner, after all earlier hardware suites.
"""
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import requests

root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from benchmark_engines import Server

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
config = json.loads((root / 'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text())
config.update(rbitnet=str(args.binary.resolve()), cuda_quant_library=str(args.library.resolve()),
              cwd=str(root), port=18138, startup_timeout=600)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = dict(binary_sha256=sha(args.binary), library_sha256=sha(args.library),
              harness_sha256=sha(Path(__file__)), cases=[], memory={}, namespaces={})

def save():
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def identity(row):
    choice = row['choices'][0]
    return choice['message']['content'], choice['finish_reason'], row['usage']['completion_tokens']

def raw_chat(model, prompt):
    if model['id'] == 'llama32-1b':
        return '<|start_header_id|>user<|end_header_id|>\n\n' + prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    return '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'

def unary(server, body):
    before = server.metrics()
    started = time.perf_counter()
    response = requests.post(server.base + '/v1/chat/completions', json=body, timeout=600)
    response.raise_for_status()
    row = response.json()
    assert 'error' not in row and '\ufffd' not in identity(row)[0], row
    return dict(request=body, response=row, wall_ms=(time.perf_counter()-started)*1000,
                metrics_before=before, metrics_after=server.metrics())

def stream(server, body, disconnect=False):
    pieces, terminals, times, events = [], [], [], []
    done, first_metrics = 0, None
    started = time.perf_counter()
    with requests.post(server.base + '/v1/chat/completions', json=body | dict(stream=True), stream=True, timeout=600) as response:
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=1):
            if not raw.startswith(b'data: '): continue
            if raw == b'data: [DONE]': done += 1; continue
            event = json.loads(raw[6:]); events.append(event)
            assert 'error' not in event, event
            choice = event['choices'][0]
            if choice.get('finish_reason') is not None: terminals.append(choice['finish_reason'])
            text = choice.get('delta', {}).get('content', '')
            if text:
                pieces.append(text); times.append((time.perf_counter()-started)*1000)
                if first_metrics is None: first_metrics = server.metrics()
                if disconnect and len(pieces) == 3: break
    after = server.metrics()
    if disconnect:
        assert len(pieces) == 3 and done == 0
    else:
        assert done == 1 and len(terminals) == 1
        assert first_metrics is not None, 'stream must expose actual content'
        # All context transfers are complete before the first visible text.
        # This metrics probe is a correctness check, not a quiet throughput measure.
        for name in ['captures', 'writes', 'read_ns', 'write_ns', 'restored_tokens']:
            key = 'rbitnet_core_context_' + name + '_total'
            assert first_metrics[key] == after[key], (key, first_metrics, after)
    return dict(text=''.join(pieces), finish_reason=terminals[-1] if terminals else None,
                ttft_ms=times[0] if times else None, wall_ms=(time.perf_counter()-started)*1000,
                first_text_metrics=first_metrics, metrics_after=after, events=events, disconnected=disconnect)

def delta(case, name):
    key = 'rbitnet_core_context_' + name + '_total'
    return case['metrics_after'].get(key, 0) - case['metrics_before'].get(key, 0)

for model in config['models']:
    if model['id'] not in ['llama32-1b', 'qwen35-2b']: continue
    identifier = model['id']
    directory = out / identifier / 'cache'
    directory.mkdir(parents=True, exist_ok=True)
    common = dict(RBITNET_MAX_SEQ='2048', RBITNET_CUDA_KV_FORMAT='f32', RBITNET_CUDA_KV_PAGE_LIMIT='0',
                  RBITNET_CUDA_DEVICE_BUDGET_MB='12288', RBITNET_CUDA_DEVICE_MARGIN_MB='256',
                  RBITNET_CUDA_PREFILL='1', RBITNET_CUDA_PREFILL_TOKENS='128', RBITNET_CUDA_SPLIT_KV='0',
                  RBITNET_CUDA_PREFILL_TF32X3='0', RBITNET_CUDA_RESIDENT_GRAPH='1', RBITNET_PREFIX_KV='0',
                  RBITNET_CUDA_QWEN_FULL='1' if identifier == 'qwen35-2b' else '0',
                  RBITNET_REQUIRE_QWEN_FULL='1' if identifier == 'qwen35-2b' else '0',
                  RBITNET_CUDA_QWEN_PREFILL='1' if identifier == 'qwen35-2b' else '0',
                  RBITNET_QWEN_SPECULATIVE='0', RBITNET_SPECULATIVE_PLD='0', RBITNET_CUDA_CONTINUOUS='0',
                  RBITNET_CONTEXT_DIR=str(directory), RBITNET_CONTEXT_DISK_MB='512', RBITNET_CONTEXT_DISK_GLOBAL_MB='512',
                  RBITNET_CONTEXT_ENTRIES='8', RBITNET_CONTEXT_TTL_SECS='1800')
    def create(label, tiers, ram=128):
        cfg = config | dict(rbitnet_env=common | dict(RBITNET_CONTEXT_TIERS=str(int(tiers)), RBITNET_CONTEXT_RAM_MB=str(ram)))
        server = Server(cfg, model, 'rbitnet', 'gpu', out)
        server.log_path = out / (identifier + '-' + label + '.log')
        return server
    prompts = [('Notes on the garden and the library.\n' * 24) + 'Write a detailed story about a robot exploring this place.',
               ('Notes on the river and the observatory.\n' * 24) + 'Write a detailed story about a robot exploring this place.']
    bodies = [dict(model=identifier, messages=[dict(role='user', content=raw_chat(model, prompt))],
                   max_tokens=65, temperature=.7, top_p=.9, seed=42+i, frequency_penalty=.2, presence_penalty=.1)
              for i, prompt in enumerate(prompts)]
    reference = create('reference', False)
    try:
        reference.start()
        refs = [unary(reference, body) for body in bodies]
        assert all(identity(row['response'])[0].strip() for row in refs)
        report['cases'].append(dict(model=identifier, kind='reference', records=refs)); save()
    finally: report['memory'][identifier+'-reference'] = reference.close(); save()
    first = create('initial', True)
    try:
        first.start()
        for index, body in enumerate(bodies):
            row = unary(first, body)
            assert identity(row['response']) == identity(refs[index]['response'])
            assert delta(row, 'captures') == 1 and delta(row, 'writes') == 1
            report['cases'].append(dict(model=identifier, kind='initial-capture', record=row)); save()
        hit = unary(first, bodies[0]); assert delta(hit, 'ram_hits') == 1
        assert identity(hit['response']) == identity(refs[0]['response'])
        streamed = stream(first, bodies[0])
        assert (streamed['text'], streamed['finish_reason']) == identity(refs[0]['response'])[:2]
        interrupted = stream(first, bodies[0], True)
        resumed = unary(first, bodies[1]); assert identity(resumed['response']) == identity(refs[1]['response'])
        report['cases'].append(dict(model=identifier, kind='ram-stream-disconnect', hit=hit, stream=streamed, interrupted=interrupted, survivor=resumed)); save()
    finally: report['memory'][identifier+'-initial'] = first.close(); save()
    objects = list(directory.rglob('*.state')); assert len(objects) == 2
    sizes = []
    for path in objects:
        with path.open('rb') as file:
            sizes.append(int.from_bytes(file.read(20)[12:20], 'little'))
    ram = math.ceil(max(sizes)/1048576)
    assert min(sizes)*2 > ram*1048576, ('fixture must force host eviction', sizes, ram)
    report['namespaces'][identifier] = dict(ram_mib=ram, payload_bytes=sizes,
                                           sealed_files={str(path.relative_to(directory)):sha(path) for path in objects})
    restarted = create('restarted-small-ram', True, ram)
    try:
        restarted.start()
        for index in [0, 1, 0]:
            row = unary(restarted, bodies[index]); assert delta(row, 'disk_hits') == 1
            assert identity(row['response']) == identity(refs[index]['response'])
            report['cases'].append(dict(model=identifier, kind='restart-and-eviction', record=row)); save()
        stopped = unary(restarted, bodies[0] | dict(stop=identity(refs[0]['response'])[0][:2]))
        assert identity(stopped['response'])[:2] == ('', 'stop')
        report['cases'].append(dict(model=identifier, kind='stop-after-restore', record=stopped)); save()
    finally: report['memory'][identifier+'-restarted'] = restarted.close(); save()
    # Payload corruption preserves a valid header, so full seal verification,
    # rather than header indexing alone, must refuse it before Native import.
    for path in directory.rglob('*.state'):
        with path.open('r+b') as file:
            fixed = file.read(20); offset = 20 + int.from_bytes(fixed[8:12], 'little')
            file.seek(offset); value = file.read(1); file.seek(offset); file.write(bytes([value[0] ^ 1]))
    corrupt = create('corrupt-replay', True, ram)
    try:
        corrupt.start()
        row = unary(corrupt, bodies[0]); assert delta(row, 'read_failures') == 1 and delta(row, 'disk_hits') == 0
        assert identity(row['response']) == identity(refs[0]['response'])
        assert delta(row, 'captures') == 1 and delta(row, 'writes') == 1
        report['cases'].append(dict(model=identifier, kind='corrupt-state-replay', record=row)); save()
    finally: report['memory'][identifier+'-corrupt'] = corrupt.close(); save()

# Two actual model processes share a cap sized for only the larger checkpoint.
# Requests remain serial; these are correctness captures, not throughput runs.
selected = {m['id']: m for m in config['models'] if m['id'] in ['llama32-1b', 'qwen35-2b']}
references = {row['model']: row['records'] for row in report['cases'] if row['kind'] == 'reference'}
cap_mib = math.ceil((max(max(row['payload_bytes']) for row in report['namespaces'].values()) + 65536) / 1048576)
cap = cap_mib * 1048576
shared = out / 'shared-global-root'; shared.mkdir()
def quota_server(identifier, port):
    env = common | dict(RBITNET_CONTEXT_DIR=str(shared), RBITNET_CONTEXT_DISK_GLOBAL_MB=str(cap_mib),
        RBITNET_CONTEXT_TIERS='1', RBITNET_CONTEXT_RAM_MB='128',
        RBITNET_CUDA_QWEN_FULL='1' if identifier == 'qwen35-2b' else '0',
        RBITNET_REQUIRE_QWEN_FULL='1' if identifier == 'qwen35-2b' else '0',
        RBITNET_CUDA_QWEN_PREFILL='1' if identifier == 'qwen35-2b' else '0')
    server = Server(config | dict(port=port, rbitnet_env=env), selected[identifier], 'rbitnet', 'gpu', out)
    server.log_path = out / (identifier + '-global-active.log')
    return server
def physical():
    objects = list(shared.rglob('*.state'))
    temporary = list(shared.rglob('.*.tmp'))
    total = sum(p.stat().st_size for p in objects + temporary)
    assert total <= cap, ('global quota exceeded', total, cap)
    return dict(bytes=total, cap_bytes=cap, objects={str(p.relative_to(shared)):sha(p) for p in objects})
holder = quota_server('qwen35-2b', 18138)
other = quota_server('llama32-1b', 18148)
holder_closed = False
try:
    holder.start()
    first = unary(holder, references['qwen35-2b'][0]['request'])
    assert identity(first['response']) == identity(references['qwen35-2b'][0]['response'])
    assert delta(first, 'writes') == 1
    initial = physical(); assert len(initial['objects']) == 1
    other.start()
    blocked = unary(other, references['llama32-1b'][0]['request'])
    assert identity(blocked['response']) == identity(references['llama32-1b'][0]['response'])
    assert delta(blocked, 'captures') == 1 and delta(blocked, 'write_failures') == 1 and delta(blocked, 'writes') == 0
    assert physical() == initial
    warm = unary(other, references['llama32-1b'][0]['request'])
    assert delta(warm, 'ram_hits') == 1
    assert identity(warm['response']) == identity(references['llama32-1b'][0]['response'])
    report['memory']['global-holder'] = holder.close(); holder_closed = True
    admitted = unary(other, references['llama32-1b'][1]['request'])
    assert identity(admitted['response']) == identity(references['llama32-1b'][1]['response'])
    assert delta(admitted, 'writes') == 1
    final = physical(); assert len(final['objects']) == 1
    assert set(initial['objects']).isdisjoint(final['objects'])
    report['global_quota'] = dict(initial=initial, blocked=blocked, warm=warm, admitted=admitted, final=final,
        limits=['Serial requests to two live model processes; no throughput claim.', 'Model holder closes normally; abrupt exit separately exercised by filesystem process fixture.'])
    save()
finally:
    report['memory']['global-other'] = other.close()
    if not holder_closed: report['memory']['global-holder'] = holder.close()
    save()
print('GLOBAL_QUOTA_MODELS_DONE active_foreign_preserved=true ram_fallback=true idle_reclaimed=true exact_outputs=true physical_cap=true', flush=True)

report['complete'] = True; save()
print('CONTEXT_TIERS_HTTP_DONE models=2 process_restart=true ram_eviction=true corruption_replay=true token_loop_io=false', flush=True)
