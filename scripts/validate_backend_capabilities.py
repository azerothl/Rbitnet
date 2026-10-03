#!/usr/bin/env python3
"""Live Windows HTTP checks: real CUDA auto, CPU stub metadata, rejected GPU prototypes, OS RSS."""
import argparse
import hashlib
import json
import pathlib
import subprocess
from unittest.mock import patch

import psutil
import requests
from benchmark_engines import Server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=pathlib.Path, required=True)
    p.add_argument('--binary', type=pathlib.Path, required=True)
    p.add_argument('--library', type=pathlib.Path, required=True)
    p.add_argument('--output-dir', type=pathlib.Path, required=True)
    args = p.parse_args()
    root = args.output_dir; root.mkdir(parents=True, exist_ok=True)
    binary = root/'rbitnet.exe'; binary.write_bytes(args.binary.read_bytes())
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    config.update(rbitnet=str(binary.resolve()), cuda_quant_library=str(args.library.resolve()), port=18108)
    config.pop('rbitnet_env', None)
    model = next(m for m in config['models'] if m['id']=='qwen35-2b')
    report = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), cases=[])
    original = subprocess.Popen
    def save(): (root/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for mode in ['stub-cuda-requested', 'auto-cuda-qwen', 'vulkan-refused', 'metal-refused']:
        def popen(*a, **kw):
            if kw.get('env'):
                kw['env'] = kw['env'].copy()
                kw['env'].update(RBITNET_CUDA_QWEN_FULL='1', RBITNET_REQUIRE_QWEN_FULL='1', RBITNET_BACKEND='auto', RBITNET_ADMIN_TOKEN='backend-validation-local')
                for key in ['RBITNET_CHAT_TEMPLATE', 'RBITNET_CHAT_FORMAT']: kw['env'].pop(key, None)
                if mode=='stub-cuda-requested':
                    kw['env'].update(RBITNET_STUB='1', RBITNET_BACKEND='cuda')
                    kw['env'].pop('RBITNET_MODEL', None)
                if mode.endswith('-refused'): kw['env']['RBITNET_BACKEND'] = mode.split('-')[0]
            return original(*a, **kw)
        server = Server(config, model, 'rbitnet', 'gpu', root); server.log_path = root/(mode+'.log')
        try:
            error = None
            try:
                with patch('subprocess.Popen', popen): server.start()
            except RuntimeError as e: error = str(e)
            ready = requests.get(server.base+'/ready', timeout=5)
            models = requests.get(server.base+'/v1/models', timeout=5); models.raise_for_status()
            raw_models = models.json()
            body = dict(model=model['id'], messages=[dict(role='user',content='Quelle est la capitale de la France ? Réponds en un mot.')], max_tokens=16, temperature=0)
            response = requests.post(server.base+'/v1/chat/completions', json=body, timeout=60)
            metrics = server.metrics()
            rss = metrics.get('rbitnet_process_rss_bytes')
            os_rss = psutil.Process(server.proc.pid).memory_info().rss
            row = dict(mode=mode, startup_error=error, ready_status=ready.status_code, ready_text=ready.text,
                       models=raw_models, response_status=response.status_code, response=response.json(), metrics=metrics,
                       metrics_rss=rss, psutil_rss=os_rss)
            report['cases'].append(row); save()
            assert 'rbitnet_process_vram_bytes' not in metrics
            assert metrics.get('rbitnet_process_vram_measurement_available')==0
            assert rss and abs(rss-os_rss) < max(32*1024*1024,os_rss*0.05), (mode,rss,os_rss)
            if mode.endswith('-refused'):
                assert error and ready.status_code==503
                assert response.status_code==503, 'LoadFailed must not return a fake stub completion'
                row['blocked_routes'] = probe_unavailable(server, body, 'LoadFailed')
            else:
                assert error is None and ready.status_code==200 and response.status_code==200
                text=response.json()['choices'][0]['message']['content']
                metadata=raw_models['data'][0]['metadata']
                if mode=='stub-cuda-requested':
                    assert metadata['backend']=='cpu' and not metadata['backend_accelerated']
                    assert 'stub' in text
                else:
                    assert metadata['backend']=='cuda' and metadata['backend_accelerated']
                    assert metrics.get('rbitnet_core_gpu_qwen_full_tokens_total',0)>0
                    assert 'Paris' in text and 'stub' not in text
                    headers={'x-rbitnet-admin-token':'backend-validation-local'}
                    failed=requests.post(server.base+'/v1/admin/reload',headers=headers,
                                         json={'model':'missing-validation-model.gguf'},timeout=30)
                    assert failed.status_code==500
                    assert requests.get(server.base+'/ready',timeout=5).status_code==200
                    retained=requests.post(server.base+'/v1/chat/completions',json=body,timeout=60)
                    assert retained.status_code==200 and 'Paris' in retained.json()['choices'][0]['message']['content']
                    row['retained_model_after_failed_reload']=retained.json()
                    unloaded=requests.post(server.base+'/v1/admin/unload',headers=headers,timeout=30)
                    assert unloaded.status_code==200
                    assert requests.get(server.base+'/ready',timeout=5).status_code==503
                    row['blocked_after_unload']=probe_unavailable(server,body,'ModelUnloaded')
                    reloaded=requests.post(server.base+'/v1/admin/reload',headers=headers,json={},timeout=120)
                    assert reloaded.status_code==200, reloaded.text
                    resumed=requests.post(server.base+'/v1/chat/completions',json=body,timeout=60)
                    assert resumed.status_code==200 and 'Paris' in resumed.json()['choices'][0]['message']['content']
                    row['resumed_response']=resumed.json()
            save()
            print(mode, 'passed', flush=True)
        finally: server.close(); save()


def probe_unavailable(server, body, expected):
    rows=[]
    for route in ['/v1/chat/completions','/v1/completions','/v1/messages']:
        for stream in [False,True]:
            request=dict(body,stream=stream,prompt='hello')
            response=requests.post(server.base+route,json=request,timeout=10)
            error=response.json()
            assert response.status_code==503, (route,stream,response.status_code,error)
            assert error['error']['code']==expected, error
            if route=='/v1/messages': assert error['type']=='error'
            rows.append(dict(route=route,stream=stream,status=response.status_code,error=error))
    return rows


if __name__=='__main__': main()
