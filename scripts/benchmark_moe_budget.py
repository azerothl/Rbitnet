#!/usr/bin/env python3
"""Same-binary fixed/expert-cache ablation with a shared native memory cap."""
import argparse
import hashlib
import json
import pathlib
import re
import time
from unittest.mock import patch

import requests
from benchmark_engines import Server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=pathlib.Path,required=True)
    p.add_argument('--binary',type=pathlib.Path,required=True)
    p.add_argument('--library',type=pathlib.Path,required=True)
    p.add_argument('--output-dir',type=pathlib.Path,required=True)
    p.add_argument('--models',nargs='+',choices=['gpt-oss-20b','glm47-flash'],default=['gpt-oss-20b','glm47-flash'])
    p.add_argument('--cache-mib',nargs='+',type=int,default=[0,16,512,8192])
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--device-mib',type=int,default=12288)
    p.add_argument('--policy',choices=['lru','lfu','least-stale'],default='lru')
    args=p.parse_args()
    if args.cycles<2 or args.device_mib<1 or any(n<0 for n in args.cache_mib):p.error('invalid cycles or memory budget')
    if not args.cache_mib or args.cache_mib[0]!=0:p.error('first mode must be the fixed-placement reference (0)')
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    binary=root/'rbitnet.exe';binary.write_bytes(args.binary.read_bytes())
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    config.update(rbitnet=str(binary.resolve()),cuda_quant_library=str(args.library.resolve()),port=18113)
    config['rbitnet_env']={
        'RBITNET_CUDA_DEVICE_BUDGET_MB':str(args.device_mib),'RBITNET_CUDA_DEVICE_MARGIN_MB':'256',
        'RBITNET_MAX_SEQ':'2048','RBITNET_HYBRID_MAX_VRAM_MB':'12288','RBITNET_CUDA_GPT_FULL':'0',
        'RBITNET_REQUIRE_GPT_FULL':'0','RBITNET_CUDA_QWEN_FULL':'0','RBITNET_REQUIRE_QWEN_FULL':'0',
        'RBITNET_CUDA_SPLIT_KV':'0','RBITNET_PREFIX_KV':'0','RBITNET_CUDA_HEAD':'0',
        'RBITNET_CUDA_PREFILL':'0','RBITNET_MOE_CACHE_POLICY':args.policy,
        'RBITNET_ADMIN_TOKEN':'memory-validation-local'}
    report=dict(config=config,warmup_cycle=0,rows=[],sse=[],stops=[],unload=[],reload=[],memory={},
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    def save():(root/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    system='Tu es un assistant précis. Réponds en français.\n'+'\n'.join(
        f'Note {i}: Les villes ont des bibliothèques, des jardins et des musées.' for i in range(12))
    prompts=[
        [{'role':'system','content':system},{'role':'user','content':'Écris un récit sur un robot qui explore une bibliothèque.'}],
        [{'role':'system','content':system},{'role':'user','content':'Écris une fonction Python qui calcule une factorielle.'}],
        [{'role':'user','content':'Quelle est la capitale de la France ? Réponds en un mot.'}]]
    for model in (m for m in config['models'] if m['id'] in args.models):
        baseline={}
        for cache in args.cache_mib:
            mode=f"{model['id']}-cache{cache}-{args.policy}"
            config['rbitnet_env']['RBITNET_MOE_CACHE_MB']=str(cache)
            server=Server(config,model,'rbitnet','gpu',root);server.log_path=root/(mode+'.log')
            original=__import__('subprocess').Popen
            def natural(*a,**kw):
                if kw.get('env'):
                    kw['env']=kw['env'].copy()
                    for k in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:kw['env'].pop(k,None)
                return original(*a,**kw)
            def managed():
                raw=requests.get(server.base+'/metrics',timeout=10);raw.raise_for_status()
                values=server.metrics()
                cats={k:int(v) for k,v in re.findall(r'^rbitnet_core_cuda_managed_category_bytes\{category="([^"]+)"\} ([0-9]+)$',raw.text,re.M)}
                assert values.get('rbitnet_core_cuda_managed_memory_available')==1
                live=values['rbitnet_core_cuda_managed_live_bytes'];limit=values['rbitnet_core_cuda_managed_limit_bytes']
                assert sum(cats.values())==live and 0<limit<=args.device_mib*1024*1024
                assert live<=limit and values['rbitnet_core_cuda_managed_peak_bytes']<=limit
                assert cats['experts']<=cache*1024*1024
                return dict(metrics=values,categories=cats)
            def complete(body):
                r=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);r.raise_for_status();return r.json()
            try:
                with patch('subprocess.Popen',natural):server.start()
                loaded=requests.get(server.base+'/v1/models',timeout=10).json()
                reservation=int(re.search(r'native state reservation bytes: ([0-9]+)',json.dumps(loaded)).group(1))
                assert reservation>0
                for cycle in range(args.cycles):
                    for index,messages in enumerate(prompts):
                        body=dict(model=model['id'],messages=messages,temperature=0,max_tokens=64 if index<2 else 32)
                        before=managed();start=time.perf_counter();raw=complete(body);wall=1000*(time.perf_counter()-start);after=managed()
                        text=raw['choices'][0]['message']['content']
                        assert text and '\ufffd' not in text
                        if index==2:assert 'Paris' in text
                        if cache==0:baseline[cycle,index]=text
                        native=sum(after['categories'][k] for k in ['kv_state','activations','scratch','other'])
                        assert native<=reservation,(mode,native,reservation)
                        delta={k:after['metrics'].get(k,0)-v for k,v in before['metrics'].items()}
                        row=dict(mode=mode,model=model['id'],cache_mib=cache,policy=args.policy,env=config['rbitnet_env'].copy(),cycle=cycle,prompt=index,
                            request=body,response=raw,wall_ms=wall,metrics_delta=delta,managed=after,loaded=loaded,
                            reservation_bytes=reservation,matches_fixed_text=text==baseline[cycle,index])
                        report['rows'].append(row);save()
                        print(json.dumps(dict(mode=mode,cycle=cycle,prompt=index,prefill_ms=delta.get('rbitnet_inference_prefill_ms_sum'),
                            decode_ms=delta.get('rbitnet_inference_decode_ms_sum'),hits=delta.get('rbitnet_core_expert_cache_hits_total'),
                            misses=delta.get('rbitnet_core_expert_cache_misses_total'),matches=row['matches_fixed_text'],live=after['metrics']['rbitnet_core_cuda_managed_live_bytes'])),flush=True)
                for label,opts in [('greedy',dict(temperature=0)),('seed',dict(temperature=.7,seed=42)),('penalties',dict(temperature=0,frequency_penalty=.2,presence_penalty=.1))]:
                    body=dict(model=model['id'],messages=[{'role':'user','content':'Répète exactement : été, café, résumé, 🙂.'}],max_tokens=64,**opts)
                    raw=complete(body);text=raw['choices'][0]['message']['content']
                    r=requests.post(server.base+'/v1/chat/completions',json={**body,'stream':True},timeout=600);r.raise_for_status()
                    parts=[];done=False
                    for line in r.content.decode('utf-8').splitlines():
                        if line=='data: [DONE]':done=True
                        elif line.startswith('data: '):parts.append(json.loads(line[6:]).get('choices',[{}])[0].get('delta',{}).get('content',''))
                    joined=''.join(parts);assert done and joined==text and text and '\ufffd' not in text
                    report['sse'].append(dict(mode=mode,sampling=label,request=body,response=raw,text=text,sse_text=joined,done=done));save()
                body=dict(model=model['id'],messages=prompts[2],max_tokens=32,temperature=0,stop=['Paris'])
                raw=complete(body);assert 'Paris' not in raw['choices'][0]['message']['content']
                report['stops'].append(dict(mode=mode,request=body,response=raw));save()
                before=managed()
                r=requests.post(server.base+'/v1/admin/unload',headers={'x-rbitnet-admin-token':'memory-validation-local'},timeout=30);r.raise_for_status()
                after=managed()
                assert all(v==0 for k,v in after['categories'].items() if k!='scratch'),after
                unavailable=requests.post(server.base+'/v1/chat/completions',json=body,timeout=30)
                assert unavailable.status_code==503
                report['unload'].append(dict(mode=mode,before=before,after=after,unavailable_status=unavailable.status_code));save()
                # Reload in the same process: the ledger and its retained scratch
                # outlive the old model. Required state must still fit the cap.
                r=requests.post(server.base+'/v1/admin/reload',json={},headers={'x-rbitnet-admin-token':'memory-validation-local'},timeout=120)
                r.raise_for_status()
                ready=requests.get(server.base+'/ready',timeout=10);ready.raise_for_status()
                paris=complete(dict(model=model['id'],messages=prompts[2],max_tokens=32,temperature=0))
                assert 'Paris' in paris['choices'][0]['message']['content']
                loaded_again=managed()
                r=requests.post(server.base+'/v1/admin/unload',headers={'x-rbitnet-admin-token':'memory-validation-local'},timeout=30);r.raise_for_status()
                released_again=managed()
                assert all(v==0 for k,v in released_again['categories'].items() if k!='scratch'),released_again
                report['reload'].append(dict(mode=mode,response=paris,loaded=loaded_again,released=released_again));save()
            finally:report['memory'][mode]=server.close();save()


if __name__=='__main__':main()
