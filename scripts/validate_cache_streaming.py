#!/usr/bin/env python3
"""Live cancellation/resume, stop and concurrent-request checks for native caches."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
from unittest.mock import patch
import requests
from benchmark_engines import Server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=pathlib.Path, required=True)
    p.add_argument('--binary', type=pathlib.Path, required=True)
    p.add_argument('--library', type=pathlib.Path, required=True)
    p.add_argument('--output-dir', type=pathlib.Path, required=True)
    p.add_argument('--port', type=int, default=18106)
    p.add_argument('--speculative', action='store_true', help='Validate native Llama PLD together with prefix reuse')
    p.add_argument('--qwen-full', action='store_true', help='Validate complete dense Qwen GPU pipeline with checkpoints')
    p.add_argument('--split-kv', action='store_true', help='Validate exact split-KV with runtime graphs and cached prefixes')
    args = p.parse_args(); root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    binary = root/'rbitnet.exe'; binary.write_bytes(args.binary.read_bytes())
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config.update(rbitnet=str(binary.resolve()), cuda_quant_library=str(args.library.resolve()), port=args.port)
    report = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), speculative=args.speculative, qwen_full=args.qwen_full, split_kv=args.split_kv, cases=[])
    original = subprocess.Popen
    def save(): (root/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    modes = [(config['models'][0], '1')] if args.speculative else [(config['models'][0], '0'), (config['models'][0], '1'), (config['models'][1], '0')]
    if args.qwen_full:
        if args.speculative: p.error('select only one architecture experiment')
        modes = [(next(m for m in config['models'] if m['id']=='qwen35-2b'), '0')]
    for model, block in modes:
        label = model['id']+'-block'+block
        def popen(*a, **kw):
            if kw.get('env'):
                kw['env'] = kw['env'].copy()
                kw['env'].update(RBITNET_PREFIX_KV='1', RBITNET_CUDA_PREFIX_MB='256', RBITNET_CUDA_PREFIX_ENTRIES='8',
                                 RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS='32', RBITNET_CUDA_PREFILL=block,
                                 RBITNET_SPECULATIVE_PLD='1' if args.speculative else '0', RBITNET_SPECULATIVE='0', RBITNET_SPECULATIVE_TOKENS='15',
                                 RBITNET_CUDA_SPLIT_KV='1' if args.split_kv else '0',
                                 RBITNET_MAX_CONCURRENT='4')
                kw['env'].update(RBITNET_CUDA_QWEN_FULL='1' if args.qwen_full else '0', RBITNET_REQUIRE_QWEN_FULL='1' if args.qwen_full else '0')
                for k in ['RBITNET_CHAT_TEMPLATE', 'RBITNET_CHAT_FORMAT']: kw['env'].pop(k, None)
            return original(*a, **kw)
        server = Server(config, model, 'rbitnet', 'gpu', root); server.log_path = root/(label+'.log')
        try:
            with patch('subprocess.Popen', popen): server.start()
            def complete(body):
                r = requests.post(server.base+'/v1/chat/completions', json=body, timeout=300); r.raise_for_status(); return r.json()
            def request(question, **opts):
                return dict(model=model['id'], messages=[{'role':'system', 'content':'Tu es un assistant précis. Réponds en français et commence directement ta réponse.'},
                                                       {'role':'user', 'content':question}], max_tokens=128, **opts)
            for mode, options in [('greedy', dict(temperature=0)), ('sampling', dict(temperature=0.7, seed=42)),
                                  ('penalties', dict(temperature=0, frequency_penalty=0.2, presence_penalty=0.1))]:
                body = request('Écris un long récit sur un robot qui explore une bibliothèque.', **options)
                expected = complete(body)
                stream = requests.post(server.base+'/v1/chat/completions', json={**body, 'max_tokens':512, 'stream':True}, stream=True, timeout=300)
                stream.raise_for_status(); fragments = []
                try:
                    for line in stream.iter_lines(chunk_size=1):
                        if line.startswith(b'data: ') and line != b'data: [DONE]':
                            text = json.loads(line[6:].decode('utf-8')).get('choices',[{}])[0].get('delta',{}).get('content','')
                            if text: fragments.append(text)
                            if len(fragments) >= 2: break
                finally: stream.close()
                assert len(fragments) >= 2, 'stream ended before cancellation check'
                resumed = complete(body)
                a = expected['choices'][0]['message']['content']; b = resumed['choices'][0]['message']['content']
                report['cases'].append(dict(config=label, mode=mode, kind='client_disconnect_then_resume',
                                            request=body, cancelled_fragments=fragments, expected=expected, resumed=resumed, equal=a==b)); save()
                assert a == b and a and '\ufffd' not in b, (label, mode)
                print(label, mode, 'disconnect/resume passed', flush=True)
            body = request('Quelle est la capitale de la France ? Réponds en un mot.', temperature=0, stop=['Paris'])
            expected = complete(body)
            stream = requests.post(server.base+'/v1/chat/completions', json={**body, 'stream':True}, timeout=300); stream.raise_for_status()
            pieces, done = [], False
            for line in stream.content.decode('utf-8').splitlines():
                if line == 'data: [DONE]': done = True
                elif line.startswith('data: '): pieces.append(json.loads(line[6:]).get('choices',[{}])[0].get('delta',{}).get('content',''))
            text = ''.join(pieces); expected_text = expected['choices'][0]['message']['content']
            report['cases'].append(dict(config=label, kind='explicit_stop_http_sse', request=body, response=expected, sse_text=text, done=done)); save()
            assert done and text == expected_text and 'Paris' not in text
            questions = ['Quelle est la capitale de la France ?', 'Calcule 13 + 29.',
                         'Quelle est la capitale de l’Italie ?', 'Répète : été, café, résumé, 🙂.']
            if args.speculative:
                questions[1] = 'Repeat the following sequence eight times without commentary: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.'
                questions[2] = 'Return only this Python code, repeated six times:\ndef add(a, b):\n    return a + b\n'
            bodies = [request(q, temperature=0) for q in questions]
            expected = [complete(body) for body in bodies]
            before = server.metrics()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: actual = list(pool.map(complete, bodies))
            after = server.metrics(); delta = {k: after.get(k, 0)-before.get(k, 0) for k in after}
            equal = all(a['choices'][0]['message']['content'] == b['choices'][0]['message']['content'] for a,b in zip(expected, actual))
            report['cases'].append(dict(config=label, kind='four_concurrent_requests_serialized_runtime', requests=bodies, expected=expected, actual=actual, equal=equal, metrics_delta=delta)); save()
            assert equal, label
            if args.speculative: assert delta.get('rbitnet_core_speculative_verify_blocks_total', 0) > 0
            if args.qwen_full: assert delta.get('rbitnet_core_gpu_qwen_full_tokens_total', 0) > 0
            if args.split_kv:
                used=delta.get('rbitnet_core_gpu_split_attention_queries_total', 0)
                if model['id']=='llama32-1b' or args.qwen_full: assert used > 0
                else: assert used == 0, 'partial Qwen must not claim the full-attention split kernels'
            print(label, 'stop and concurrency passed', flush=True)
        finally: server.close(); save()


if __name__ == '__main__': main()
