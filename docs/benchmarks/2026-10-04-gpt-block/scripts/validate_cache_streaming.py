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
    p.add_argument('--tf32x3', action='store_true', help='Validate Llama compensated block prefill with prefix reuse')
    p.add_argument('--qwen-prefill', action='store_true', help='Validate full Qwen block prefill with checkpoints')
    p.add_argument('--gpt-full', action='store_true', help='Validate fixed-bank resident GPT-OSS resets, disconnects and serialization')
    p.add_argument('--moe-cache', type=int, help='Validate actual dynamic MoE leases, disconnects and serialized concurrency with this MiB budget')
    p.add_argument('--moe-model', choices=['gpt-oss-20b','glm47-flash'], default='gpt-oss-20b')
    p.add_argument('--moe-execution', choices=['cache', 'cpu', 'adaptive'], default='cache')
    p.add_argument('--device-mib', type=int, default=12288)
    p.add_argument('--mla-full', action='store_true', help='Validate compressed MLA segmented graphs, dynamic expert admission and prefix restore')
    p.add_argument('--gpt-segmented', action='store_true', help='Validate GPT host-admitted FFNs, prefixes and cache/CPU recovery; requires --gpt-full')
    p.add_argument('--expect-cpu-routed', action='store_true', help='Require resident MLA/GPT routed FFNs to fall back to CPU under an explicitly too-small expert cache')
    p.add_argument('--gpt-prefill',type=int,choices=[16,32])
    p.add_argument('--moe-fused',action='store_true')
    args = p.parse_args(); root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    if args.qwen_prefill and not args.qwen_full: p.error('--qwen-prefill requires --qwen-full')
    if args.tf32x3 and args.qwen_full and not args.qwen_prefill: p.error('select --qwen-prefill for Qwen Tensor Core validation')
    if args.gpt_full and (args.qwen_full or args.tf32x3 or args.speculative): p.error('select only one architecture experiment')
    if args.gpt_segmented and not args.gpt_full:p.error('--gpt-segmented requires --gpt-full')
    if args.moe_cache is not None and ((args.gpt_full and not args.gpt_segmented) or args.qwen_full or args.tf32x3 or args.speculative or args.moe_cache<=0): p.error('select a positive MoE cache budget with MLA or segmented GPT')
    if args.mla_full and (args.gpt_full or args.qwen_full or args.tf32x3 or args.speculative):p.error('select only MLA')
    if args.expect_cpu_routed and (not (args.mla_full or args.gpt_segmented) or args.moe_cache is None):p.error('--expect-cpu-routed requires MLA/segmented GPT and an explicit cache budget')
    if args.gpt_segmented:args.moe_model='gpt-oss-20b'
    if args.device_mib<1:p.error('device budget must be positive')
    if args.mla_full:args.moe_model='glm47-flash'
    binary = root/'rbitnet.exe'; binary.write_bytes(args.binary.read_bytes())
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config.update(rbitnet=str(binary.resolve()), cuda_quant_library=str(args.library.resolve()), port=args.port)
    report = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), speculative=args.speculative, qwen_full=args.qwen_full, gpt_full=args.gpt_full, split_kv=args.split_kv, tf32x3=args.tf32x3, moe_cache_mib=args.moe_cache, moe_execution=args.moe_execution, mla_full=args.mla_full, gpt_segmented=args.gpt_segmented, cases=[])
    original = subprocess.Popen
    def save(): (root/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    modes = [(config['models'][0], '1')] if args.speculative else [(config['models'][0], '0'), (config['models'][0], '1'), (config['models'][1], '0')]
    if args.qwen_full:
        if args.speculative: p.error('select only one architecture experiment')
        modes = [(next(m for m in config['models'] if m['id']=='qwen35-2b'), '0')]
    if args.tf32x3 and not args.qwen_prefill: modes = [(next(m for m in config['models'] if m['id']=='llama32-1b'), '1')]
    if args.gpt_full: modes = [(next(m for m in config['models'] if m['id']=='gpt-oss-20b'), '0')]
    if args.moe_cache is not None: modes = [(next(m for m in config['models'] if m['id']==args.moe_model), '0')]
    if args.mla_full: modes=[(next(m for m in config['models'] if m['id']=='glm47-flash'),'0')]
    for model, block in modes:
        label = model['id']+'-block'+block
        def popen(*a, **kw):
            if kw.get('env'):
                kw['env'] = kw['env'].copy()
                kw['env'].update(RBITNET_PREFIX_KV='1', RBITNET_CUDA_PREFIX_MB='256', RBITNET_CUDA_PREFIX_ENTRIES='8',
                                 RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS='128' if args.qwen_prefill else '32', RBITNET_CUDA_PREFILL=block,
                                 RBITNET_CUDA_QWEN_PREFILL='1' if args.qwen_prefill else '0',
                                 RBITNET_SPECULATIVE_PLD='1' if args.speculative else '0', RBITNET_SPECULATIVE='0', RBITNET_SPECULATIVE_TOKENS='15',
                                 RBITNET_CUDA_SPLIT_KV='1' if args.split_kv else '0',
                                 RBITNET_CUDA_PREFILL_TF32X3='1' if args.tf32x3 else '0',
                                 RBITNET_MAX_CONCURRENT='4')
                kw['env'].update(RBITNET_CUDA_QWEN_FULL='1' if args.qwen_full else '0', RBITNET_REQUIRE_QWEN_FULL='1' if args.qwen_full else '0')
                kw['env'].update(RBITNET_CUDA_GPT_FULL='1' if args.gpt_full else '0',RBITNET_REQUIRE_GPT_FULL='1' if args.gpt_full else '0',RBITNET_MOE_CACHE_MB='0')
                kw['env']['RBITNET_CUDA_GPT_SEGMENTED']='1' if args.gpt_segmented else '0'
                kw['env']['RBITNET_MOE_EXECUTION']=args.moe_execution
                if args.moe_cache is not None or args.mla_full or args.gpt_segmented:
                    kw['env'].update(RBITNET_MOE_CACHE_MB=str(args.moe_cache or 0),RBITNET_MAX_SEQ='2048',RBITNET_CUDA_DEVICE_BUDGET_MB=str(args.device_mib),RBITNET_CUDA_DEVICE_MARGIN_MB='256')
                kw['env'].update(RBITNET_CUDA_MLA_FULL='1' if args.mla_full else '0',RBITNET_REQUIRE_MLA_FULL='1' if args.mla_full else '0')
                if args.gpt_prefill or args.moe_fused:
                    kw['env'].update(RBITNET_MAX_SEQ='2048',RBITNET_CUDA_DEVICE_BUDGET_MB=str(args.device_mib),RBITNET_CUDA_DEVICE_MARGIN_MB='256',
                        RBITNET_CUDA_GPT_PREFILL='1' if args.gpt_prefill else '0',RBITNET_REQUIRE_GPT_PREFILL='1' if args.gpt_prefill else '0',
                        RBITNET_CUDA_GPT_PREFILL_TOKENS=str(args.gpt_prefill or 16),RBITNET_CUDA_GPT_PREFILL_TILE='1' if args.gpt_prefill==32 else '0',
                        RBITNET_CUDA_MOE_FUSED='1' if args.moe_fused else '0',RBITNET_REQUIRE_FUSED_MOE='1' if args.moe_fused else '0')
                for k in ['RBITNET_CHAT_TEMPLATE', 'RBITNET_CHAT_FORMAT']: kw['env'].pop(k, None)
            return original(*a, **kw)
        server = Server(config, model, 'rbitnet', 'gpu', root); server.log_path = root/(label+'.log')
        try:
            with patch('subprocess.Popen', popen): server.start()
            initial_metrics=server.metrics()
            if args.mla_full:
                loaded=requests.get(server.base+'/v1/models',timeout=10).json()
                assert 'resident MLA attention/router/head: true' in json.dumps(loaded),loaded
            if args.gpt_segmented:
                loaded=requests.get(server.base+'/v1/models',timeout=10).json()
                assert 'resident GPT-OSS attention/router/head: true' in json.dumps(loaded),loaded
                assert 'fully resident GPT-OSS token pipeline: false' in json.dumps(loaded),loaded
            def complete(body):
                r = requests.post(server.base+'/v1/chat/completions', json=body, timeout=300); r.raise_for_status(); return r.json()
            def request(question, **opts):
                system='Tu es un assistant précis. Réponds en français et commence directement ta réponse.'
                if args.tf32x3: system+='\n'+'\n'.join(f'Note {i}: Les villes ont des bibliothèques, des jardins et des musées.' for i in range(12))
                return dict(model=model['id'], messages=[{'role':'system', 'content':system},
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
            if args.gpt_full: assert delta.get('rbitnet_core_gpu_gpt_full_tokens_total', 0) > 0
            if args.split_kv:
                used=delta.get('rbitnet_core_gpu_split_attention_queries_total', 0)
                if model['id']=='llama32-1b' or args.qwen_full or args.gpt_full or args.mla_full: assert used > 0
                else: assert used == 0, 'partial Qwen must not claim the full-attention split kernels'
            if args.tf32x3 or args.qwen_prefill:
                usage={k:after.get(k,0)-initial_metrics.get(k,0) for k in after}
                report['cases'][-1]['all_cases_metrics_delta']=usage;save()
                # The four requests above are warm prefix hits: correctly skip
                # the large prefill GEMMs. Require actual use on the cold cases.
                if args.tf32x3: assert usage.get('rbitnet_core_gpu_tensor_gemm_calls_total', 0)>0, 'compensated projections were not used'
                if args.qwen_prefill: assert usage.get('rbitnet_core_gpu_prefill_blocks_total', 0)>0, 'Qwen block prefill was not used'
            if args.moe_cache is not None:
                usage={k:after.get(k,0)-initial_metrics.get(k,0) for k in after}
                report['cases'][-1]['all_cases_metrics_delta']=usage;save()
                if args.mla_full:assert usage.get('rbitnet_core_gpu_mla_full_tokens_total',0)>0
                if args.gpt_segmented:assert usage.get('rbitnet_core_gpu_gpt_full_tokens_total',0)>0
                if args.expect_cpu_routed:
                    assert usage.get('rbitnet_core_native_moe_fallback_layers_total',0)>0, 'CPU routed fallback was not exercised'
                    assert usage.get('rbitnet_core_native_moe_resident_layers_total',0)==0, 'tiny-cache test must route all FFNs on CPU'
                elif args.moe_execution == 'cache':
                    assert usage.get('rbitnet_core_expert_cache_evictions_total',0)>0, 'test must exercise actual slot reuse'
                    assert usage.get('rbitnet_core_native_moe_resident_layers_total',0)>0
                assert after['rbitnet_core_cuda_managed_live_bytes']<=after['rbitnet_core_cuda_managed_limit_bytes']
                assert after['rbitnet_core_cuda_managed_peak_bytes']<=after['rbitnet_core_cuda_managed_limit_bytes']
            if args.gpt_prefill:
                usage={k:after.get(k,0)-initial_metrics.get(k,0)for k in after}
                assert usage.get('rbitnet_core_gpu_prefill_blocks_total',0)>0
                assert after['rbitnet_core_cuda_managed_live_bytes']<=after['rbitnet_core_cuda_managed_peak_bytes']<=after['rbitnet_core_cuda_managed_limit_bytes']<=args.device_mib*2**20
                report['cases'][-1]['managed_metrics']={k:v for k,v in after.items()if 'cuda_managed'in k}
                response=requests.get(server.base+'/metrics',timeout=10);response.raise_for_status()
                report['cases'][-1]['scoped_metrics_after']=response.text;save()
            print(label, 'stop and concurrency passed', flush=True)
        finally: server.close(); save()


if __name__ == '__main__': main()
