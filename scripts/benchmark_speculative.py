#!/usr/bin/env python3
"""Ablate native Llama token-lookup verification, keeping every raw response."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
from unittest.mock import patch
import requests
from benchmark_engines import Server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=pathlib.Path, required=True)
    p.add_argument('--binary', type=pathlib.Path, required=True)
    p.add_argument('--library', type=pathlib.Path, required=True)
    p.add_argument('--output-dir', type=pathlib.Path, required=True)
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--port', type=int, default=18107)
    args = p.parse_args()
    root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    binary = root/'rbitnet.exe'; binary.write_bytes(args.binary.read_bytes())
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config.update(rbitnet=str(binary.resolve()), cuda_quant_library=str(args.library.resolve()), port=args.port)
    report = dict(config=config, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_dirty=True, warmup_cycle=0, rows=[], streaming=[], memory={})
    report['baseline_manifest_environment'] = config.pop('environment')
    model = config['models'][0]
    phrases = [
        'Repeat the following sequence eight times without commentary: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.',
        'Return only this Python code, repeated six times:\ndef add(a, b):\n    return a + b\n',
        'Write a long story about a robot who explores a library.',
        'What is the capital of France? Answer with one word.',
    ]
    def body(question, **options):
        return dict(model=model['id'], messages=[dict(role='user', content=question)], max_tokens=256, **options)
    original = subprocess.Popen
    baseline = {}
    def save():
        (root/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for width in [0, 4, 8, 15]:
        label = 'baseline' if width == 0 else f'pld{width}'
        overrides = dict(RBITNET_CUDA_PREFILL='1', RBITNET_SPECULATIVE='0', RBITNET_SPECULATIVE_PLD=str(int(width > 0)),
                         RBITNET_SPECULATIVE_TOKENS=str(width or 8), RBITNET_SPECULATIVE_ADAPTIVE='1', RBITNET_PREFIX_KV='0', RBITNET_REQUIRE_RESIDENT='1')
        def popen(*a, **kw):
            if kw.get('env'):
                kw['env'] = {**kw['env'], **overrides}
                for key in ['RBITNET_CHAT_TEMPLATE', 'RBITNET_CHAT_FORMAT']: kw['env'].pop(key, None)
            return original(*a, **kw)
        server = Server(config, model, 'rbitnet', 'gpu', root)
        server.log_path = root/(label+'.log')
        try:
            with patch('subprocess.Popen', popen): server.start()
            for cycle in range(args.cycles):
                for index, question in enumerate(phrases):
                    request = body(question, temperature=0)
                    before = server.metrics(); start = time.perf_counter()
                    response = requests.post(server.base+'/v1/chat/completions', json=request, timeout=300)
                    response.raise_for_status(); raw = response.json(); wall = 1000*(time.perf_counter()-start)
                    after = server.metrics(); delta = {k: after.get(k, 0)-before.get(k, 0) for k in after}
                    text = raw['choices'][0]['message']['content']
                    if not width: baseline[cycle, index] = text
                    equal = text == baseline[cycle, index]
                    report['rows'].append(dict(mode=label, env=overrides, cycle=cycle, prompt=index, request=request, response=raw,
                                               wall_ms=wall, metrics_delta=delta, matches_baseline=equal)); save()
                    print(json.dumps(dict(mode=label, cycle=cycle, prompt=index, decode_ms=delta.get('rbitnet_inference_decode_ms_sum'),
                                          blocks=delta.get('rbitnet_core_speculative_verify_blocks_total'),
                                          draft=delta.get('rbitnet_core_speculative_draft_tokens_total'),
                                          accepted=delta.get('rbitnet_core_speculative_accepted_tokens_total'), matches=equal)), flush=True)
                    assert equal and text and '\ufffd' not in text, (label, cycle, index)
            for sampling, options in [('greedy', dict(temperature=0)), ('seed', dict(temperature=0.7, seed=735, top_p=0.9)),
                                      ('penalties', dict(temperature=0.7, seed=735, top_p=0.9, frequency_penalty=0.25, presence_penalty=0.15))]:
                request = body(phrases[0], **options)
                response = requests.post(server.base+'/v1/chat/completions', json=request, timeout=300); response.raise_for_status()
                raw = response.json(); text = raw['choices'][0]['message']['content']
                key = 'sampling', sampling
                if not width: baseline[key] = text
                stream = requests.post(server.base+'/v1/chat/completions', json={**request, 'stream':True}, timeout=300); stream.raise_for_status()
                parts, done = [], False
                for line in stream.content.decode('utf-8').splitlines():
                    if line == 'data: [DONE]': done = True
                    elif line.startswith('data: '): parts.append(json.loads(line[6:]).get('choices',[{}])[0].get('delta',{}).get('content',''))
                joined = ''.join(parts)
                report['streaming'].append(dict(mode=label, kind='http_sse', sampling=sampling, request=request, response=raw,
                                                sse_text=joined, done=done, matches_baseline=text==baseline[key])); save()
                assert done and joined == text == baseline[key], (label, sampling)
            # Cancel after repeated text is visible, then use the same runtime again.
            request = body(phrases[0], temperature=0)
            stream = requests.post(server.base+'/v1/chat/completions', json={**request, 'max_tokens':512, 'stream':True}, stream=True, timeout=300)
            stream.raise_for_status(); pieces=[]
            try:
                for line in stream.iter_lines(chunk_size=1):
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        part = json.loads(line[6:].decode('utf-8')).get('choices',[{}])[0].get('delta',{}).get('content','')
                        if part: pieces.append(part)
                        if len(pieces) >= 12: break
            finally: stream.close()
            assert len(pieces) >= 12
            response = requests.post(server.base+'/v1/chat/completions', json=request, timeout=300); response.raise_for_status()
            raw = response.json()
            equal = raw['choices'][0]['message']['content'] == baseline[0, 0]
            report['streaming'].append(dict(mode=label, kind='disconnect_resume', request=request, fragments=pieces, response=raw, matches_baseline=equal)); save()
            assert equal
            request = body(phrases[0], temperature=0, stop=['gamma'])
            response = requests.post(server.base+'/v1/chat/completions', json=request, timeout=300); response.raise_for_status()
            raw = response.json()
            stream = requests.post(server.base+'/v1/chat/completions', json={**request,'stream':True}, timeout=300); stream.raise_for_status()
            parts=[]; done=False
            for line in stream.content.decode('utf-8').splitlines():
                if line == 'data: [DONE]': done=True
                elif line.startswith('data: '): parts.append(json.loads(line[6:]).get('choices',[{}])[0].get('delta',{}).get('content',''))
            text=''.join(parts)
            report['streaming'].append(dict(mode=label, kind='stop', request=request, response=raw, sse_text=text, done=done)); save()
            assert done and text == raw['choices'][0]['message']['content'] and 'gamma' not in text
        finally:
            report['memory'][label] = server.close(); save()
    assert all(any(r['mode']==f'pld{width}' and r['metrics_delta'].get('rbitnet_core_speculative_verify_blocks_total',0)>0 for r in report['rows']) for width in [4,8,15])


if __name__ == '__main__': main()
