#!/usr/bin/env python3
"""Live fail-closed checks for insufficient caps and an older native DLL."""
import argparse
import hashlib
import json
import os
import pathlib
import requests
from benchmark_engines import Server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=pathlib.Path,required=True)
    p.add_argument('--binary',type=pathlib.Path,required=True)
    p.add_argument('--library',type=pathlib.Path,required=True)
    p.add_argument('--legacy-library',type=pathlib.Path,required=True)
    p.add_argument('--output-dir',type=pathlib.Path,required=True)
    args=p.parse_args()
    os.environ['RUST_LOG']='info'
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    binary=root/'rbitnet.exe';binary.write_bytes(args.binary.read_bytes())
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    config.update(rbitnet=str(binary.resolve()),port=18114)
    model=next(m for m in config['models'] if m['id']=='gpt-oss-20b')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cases=[])
    def save():(root/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for label,library,budget in [('insufficient-state',args.library,64),('missing-memory-abi',args.legacy_library,12288)]:
        config['cuda_quant_library']=str(library.resolve())
        config['rbitnet_env']={
            'RBITNET_CUDA_DEVICE_BUDGET_MB':str(budget),'RBITNET_CUDA_DEVICE_MARGIN_MB':'256',
            'RBITNET_MAX_SEQ':'2048','RBITNET_CUDA_GPT_FULL':'0','RBITNET_REQUIRE_GPT_FULL':'0',
            'RBITNET_MOE_CACHE_MB':'512','RBITNET_ADMIN_TOKEN':'memory-validation-local'}
        server=Server(config,model,'rbitnet','gpu',root);server.log_path=root/(label+'.log')
        case=dict(label=label,config=json.loads(json.dumps(config)),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),routes=[])
        try:
            try:server.start()
            except RuntimeError as error:case['startup_error']=str(error)
            else:raise AssertionError('explicit unsupported/insufficient cap unexpectedly loaded')
            ready=requests.get(server.base+'/ready',timeout=10)
            assert ready.status_code==503 and 'LoadFailed' in ready.text
            case['ready']=dict(status=ready.status_code,text=ready.text)
            chat=dict(model=model['id'],messages=[{'role':'user','content':'Bonjour'}],max_tokens=8)
            completion=dict(model=model['id'],prompt='Bonjour',max_tokens=8)
            anthropic=dict(model=model['id'],messages=chat['messages'],max_tokens=8)
            for route,body in [('/v1/chat/completions',chat),('/v1/completions',completion),('/v1/messages',anthropic)]:
                for stream in [False,True]:
                    r=requests.post(server.base+route,json={**body,'stream':stream},timeout=30)
                    assert r.status_code==503,(label,route,stream,r.text)
                    raw=r.json();assert raw['error']['code']=='LoadFailed'
                    case['routes'].append(dict(path=route,stream=stream,status=r.status_code,response=raw))
            metrics=server.metrics();case['metrics']=metrics
            assert metrics.get('rbitnet_core_cuda_managed_live_bytes',0)==0
            log=server.log_path.read_text(encoding='utf-8',errors='replace');case['log']=log
            if label=='insufficient-state':assert 'state bytes with' in (log+ready.text)
            else:assert 'requires the native memory ABI' in log
            print(label,'ready and all six HTTP routes reject; no managed model allocation',flush=True)
        finally:
            case['memory']=server.close();report['cases'].append(case);save()


if __name__=='__main__':main()
