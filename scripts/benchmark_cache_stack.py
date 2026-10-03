#!/usr/bin/env python3
"""Same-revision HTTP/SSE ablation of prefix checkpoints and CUDA block prefill."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--model', choices=['llama32-1b', 'qwen35-2b', 'gpt-oss-20b'], required=True)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--binary', type=pathlib.Path, required=True)
    parser.add_argument('--library', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--cycles', type=int, default=4)
    parser.add_argument('--port', type=int, default=18105)
    parser.add_argument('--qwen-full', action='store_true', help='Compare complete dense Qwen GPU pipeline and prefix cache')
    parser.add_argument('--split-kv', action='store_true', help='Compare exact split-KV attention with its previous resident path')
    parser.add_argument('--tf32x3', action='store_true', help='Compare Llama block prefill with compensated Tensor Core projections')
    parser.add_argument('--qwen-prefill', action='store_true', help='Compare full Qwen serial/SIMT block/Tensor Core block prefill')
    parser.add_argument('--gpt-full', action='store_true', help='Compare GPT-OSS partial, resident, and resident split-KV paths')
    args = parser.parse_args()
    if args.cycles < 2: parser.error('use at least one warmup and one measured cycle')
    if args.qwen_full and (args.model != 'qwen35-2b' or args.backend != 'gpu'): parser.error('--qwen-full requires dense Qwen GPU')
    if args.gpt_full and (args.model != 'gpt-oss-20b' or args.backend != 'gpu' or args.qwen_full or args.qwen_prefill or args.tf32x3): parser.error('--gpt-full requires GPT-OSS GPU')
    if args.model == 'gpt-oss-20b' and not args.gpt_full: parser.error('GPT-OSS ablation requires --gpt-full')
    if args.split_kv and (args.backend != 'gpu' or (args.model == 'qwen35-2b' and not args.qwen_full)): parser.error('--split-kv requires resident Llama, Qwen, or GPT-OSS GPU')
    if args.qwen_prefill and (args.model != 'qwen35-2b' or args.backend != 'gpu' or not args.qwen_full): parser.error('--qwen-prefill requires full dense Qwen GPU')
    if args.tf32x3 and not args.qwen_prefill and (args.model != 'llama32-1b' or args.backend != 'gpu' or args.qwen_full): parser.error('--tf32x3 requires resident Llama GPU or --qwen-prefill')
    root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    binary = root/'rbitnet.exe'; binary.write_bytes(args.binary.read_bytes())
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config.update(rbitnet=str(binary.resolve()), cuda_quant_library=str(args.library.resolve()), port=args.port)
    model = next(m for m in config['models'] if m['id'] == args.model)
    report = dict(config=config, backend=args.backend, warmup_cycle=0, rows=[], sse=[], stops=[], memory={},
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()))
    # The input manifest describes a previous comparison. Preserve it as context,
    # while making the tested binary/library provenance authoritative here.
    report['baseline_manifest_environment'] = config.get('environment', {}).copy()
    config.setdefault('environment', {}).update(
        rbitnet_commit=report['source_commit'] + (' + local cache-stack changes' if report['source_dirty'] else ''),
        rbitnet_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        rbitnet_exe_sha256=report['binary_sha256'], cuda_quant_library_sha256=report['library_sha256'],
        cuda_execution='per-mode cache/block/full-Qwen ablation; see row env and execution counters')
    system = 'Tu es un assistant précis. Voici des notes communes à cette conversation.\n' + '\n'.join(
        f'Note {i}: Les villes ont des bibliothèques, des jardins et des musées.' for i in range(72))
    prompts = [
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Écris un récit de 150 mots sur un robot qui explore une bibliothèque.'}],
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Écris un récit de 150 mots sur un robot qui explore un musée.'}],
        [{'role': 'user', 'content': 'Quelle est la capitale de la France ? Réponds en un mot.'}],
    ]
    modes = [('baseline', '0', '0', '0'), ('prefix', '1', '0', '0')]
    if args.model == 'llama32-1b' and args.backend == 'gpu': modes += [('block', '0', '1', '0'), ('combined', '1', '1', '0')]
    if args.qwen_full: modes += [('full', '0', '0', '1'), ('full-prefix', '1', '0', '1')]
    if args.split_kv:
        modes = [('baseline', '0', '0', '1' if args.qwen_full else '0', '0'),
                 ('split', '0', '0', '1' if args.qwen_full else '0', '1'),
                 ('split-prefix', '1', '0', '1' if args.qwen_full else '0', '1')]
        if not args.qwen_full: modes += [('block', '0', '1', '0', '0'), ('split-block', '0', '1', '0', '1')]
    else: modes = [(*mode, '0') for mode in modes]
    modes = [(*mode, '0') for mode in modes]
    if args.tf32x3: modes = [('baseline','0','1','0','1','0'), ('tensor','0','1','0','1','1'), ('tensor-prefix','1','1','0','1','1')]
    modes = [(*mode,'0') for mode in modes]
    if args.qwen_prefill:
        split='1' if args.split_kv else '0'
        modes=[('baseline','0','0','1',split,'0','0'), ('block','0','0','1',split,'0','1')]
        if args.tf32x3: modes += [('tensor','0','0','1',split,'1','1'), ('tensor-prefix','1','0','1',split,'1','1')]
        else: modes += [('block-prefix','1','0','1',split,'0','1')]
    if args.gpt_full:
        modes=[('baseline','0','0','0','0','0','0'),('full','0','0','0','0','0','0'),('full-split','0','0','0','1','0','0')]
    original = subprocess.Popen
    baseline = {}
    baseline_sse = {}
    def save():
        (root/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for mode, prefix, block, full, split, tensor, qwen_block in modes:
        overrides = dict(RBITNET_PREFIX_KV=prefix, RBITNET_CUDA_PREFIX_MB='256', RBITNET_CUDA_PREFIX_ENTRIES='8',
                         RBITNET_CUDA_QWEN_FULL=full, RBITNET_REQUIRE_QWEN_FULL=full,
                         RBITNET_CUDA_GPT_FULL='1' if args.gpt_full and mode != 'baseline' else '0',
                         RBITNET_REQUIRE_GPT_FULL='1' if args.gpt_full and mode != 'baseline' else '0',
                         RBITNET_MOE_CACHE_MB='0',
                         RBITNET_CUDA_SPLIT_KV=split,
                         RBITNET_CUDA_PREFILL_TF32X3=tensor,
                         RBITNET_CUDA_QWEN_PREFILL=qwen_block,
                         RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS='128', RBITNET_CUDA_PREFILL=block,
                         RBITNET_CUDA_PREFILL_TOKENS='128', RBITNET_REQUIRE_RESIDENT='1' if args.backend == 'gpu' and args.model == 'llama32-1b' else '0')
        def popen(*a, **kw):
            if kw.get('env'):
                kw['env'] = kw['env'].copy(); kw['env'].update(overrides)
                for key in ['RBITNET_CHAT_TEMPLATE', 'RBITNET_CHAT_FORMAT']: kw['env'].pop(key, None)
            return original(*a, **kw)
        server = Server(config, model, 'rbitnet', args.backend, root); server.log_path = root/(mode+'.log')
        try:
            with patch('subprocess.Popen', popen): server.start()
            for cycle in range(args.cycles):
                for index, messages in enumerate(prompts):
                    body = dict(model=model['id'], messages=messages, max_tokens=128, temperature=0)
                    before = server.metrics(); start = time.perf_counter()
                    response = requests.post(server.base+'/v1/chat/completions', json=body, timeout=600); response.raise_for_status()
                    raw = response.json(); wall = 1000*(time.perf_counter()-start); after = server.metrics()
                    delta = {k: after.get(k, 0)-before.get(k, 0) for k in after}
                    text = raw['choices'][0]['message']['content']
                    if mode == 'baseline': baseline[cycle, index] = text
                    row = dict(mode=mode, env=overrides, cycle=cycle, prompt=index, request=body, response=raw,
                               wall_ms=wall, metrics_delta=delta, matches_baseline=text == baseline[cycle, index])
                    report['rows'].append(row); save()
                    print(json.dumps(dict(mode=mode, cycle=cycle, prompt=index, prefill_ms=delta.get('rbitnet_inference_prefill_ms_sum'),
                                          decode_ms=delta.get('rbitnet_inference_decode_ms_sum'),
                                          hits=delta.get('rbitnet_core_prefix_cache_hits_total'),
                                          blocks=delta.get('rbitnet_core_gpu_prefill_blocks_total'), matches=row['matches_baseline'])), flush=True)
                    assert text and '\ufffd' not in text and row['matches_baseline'], (mode, cycle, index, text)
                    if block == '1': assert delta.get('rbitnet_core_gpu_prefill_blocks_total', 0) > 0, 'native block prefill was not used'
                    if qwen_block == '1' and index < 2 and prefix == '0': assert delta.get('rbitnet_core_gpu_prefill_blocks_total', 0) > 0, 'native Qwen block prefill was not used'
                    if full == '1': assert delta.get('rbitnet_core_gpu_qwen_full_tokens_total', 0) > 0, 'full Qwen pipeline was not used'
                    if args.gpt_full and mode != 'baseline': assert delta.get('rbitnet_core_gpu_gpt_full_tokens_total', 0) > 0, 'full GPT-OSS pipeline was not used'
                    if split == '1': assert delta.get('rbitnet_core_gpu_split_attention_queries_total', 0) > 0, 'split-KV kernels were not used'
                    if tensor == '1' and index < 2 and prefix == '0': assert delta.get('rbitnet_core_gpu_tensor_gemm_calls_total', 0) > 0, 'Tensor Core projections were not used'
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Répète exactement : été, café, résumé, 🙂.'}]
            for label, options in [('greedy', dict(temperature=0)), ('sampling', dict(temperature=0.7, seed=42)),
                                   ('penalties', dict(temperature=0, frequency_penalty=0.2, presence_penalty=0.1))]:
                body = dict(model=model['id'], messages=messages, max_tokens=128, **options)
                response = requests.post(server.base+'/v1/chat/completions', json=body, timeout=600); response.raise_for_status()
                text = response.json()['choices'][0]['message']['content']
                stream = requests.post(server.base+'/v1/chat/completions', json={**body, 'stream': True}, timeout=600); stream.raise_for_status()
                parts, done = [], False
                for line in stream.content.decode('utf-8').splitlines():
                    if line == 'data: [DONE]': done = True
                    elif line.startswith('data: '): parts.append(json.loads(line[6:]).get('choices', [{}])[0].get('delta', {}).get('content', ''))
                joined = ''.join(parts)
                if mode == 'baseline': baseline_sse[label] = text
                report['sse'].append(dict(mode=mode, sampling=label, request=body, text=text, sse_text=joined, done=done, matches_baseline=text == baseline_sse[label])); save()
                assert done and joined == text and text and '\ufffd' not in text, (mode, label, text, joined)
                assert text == baseline_sse[label], (mode, label, 'output differs from baseline')
            stop_body = dict(model=model['id'], messages=prompts[2], max_tokens=128, temperature=0, stop=['Paris'])
            response = requests.post(server.base+'/v1/chat/completions', json=stop_body, timeout=600); response.raise_for_status()
            raw = response.json(); report['stops'].append(dict(mode=mode, request=stop_body, response=raw)); save()
            assert 'Paris' not in raw['choices'][0]['message']['content']
        finally:
            report['memory'][mode] = server.close(); save()


if __name__ == '__main__': main()
