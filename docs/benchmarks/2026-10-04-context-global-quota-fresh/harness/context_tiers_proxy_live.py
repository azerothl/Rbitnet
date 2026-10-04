"""Two actual model runners: session routing, idle recycling and process restart."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time
import psutil, requests

root=Path.cwd(); sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import MemorySampler
p=argparse.ArgumentParser()
for name in ['proxy','runner','library','standalone','output']:p.add_argument('--'+name,type=Path,required=True)
args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text())
models=[row for row in config['models']if row['id']in ['llama32-1b','qwen35-2b']]
standalone=json.loads(args.standalone.read_text(encoding='utf-8'));assert standalone['complete']
references={row['model']:row['records'][0]for row in standalone['cases']if row['kind']=='reference'}
registry=out/'registry.json';registry.write_text(json.dumps({'default':models[0]['id'],'models':{row['id']:{'gguf':row['gguf'],'tokenizer':row['tokenizer']}for row in models}}))
report=dict(proxy_sha256=sha(args.proxy),runner_sha256=sha(args.runner),library_sha256=sha(args.library),harness_sha256=sha(Path(__file__)),cases=[],memory={})
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def identity(row):
    choice=row['choices'][0];return choice['message']['content'],choice['finish_reason'],row['usage']['completion_tokens']
common=dict(RBITNET_PROXY_BIND='127.0.0.1:18138',RBITNET_MODEL_REGISTRY=str(registry),RBITNET_RUNNER_BIN=str(args.runner.resolve()),
 RBITNET_BACKEND='cuda',RBITNET_INFERENCE_BACKEND='local',RBITNET_CUDA_QUANT_LIB=str(args.library.resolve()),
 RBITNET_PROXY_STICKY='1',RBITNET_IDLE_UNLOAD_SECS='1',RBITNET_RUNNER_READY_TIMEOUT_SECS='600',
 RBITNET_CHAT_FORMAT='raw',RBITNET_CHAT_TEMPLATE='{user}',RBITNET_LLAMA_ENCODE_ADD_SPECIAL='1',RBITNET_LLAMA_WEIGHT_MODE='auto',
 RBITNET_MAX_SEQ='2048',RBITNET_CUDA_KV_FORMAT='f32',RBITNET_CUDA_KV_PAGE_LIMIT='0',RBITNET_PREFIX_KV='0',
 RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',
 RBITNET_CUDA_PREFILL='1',RBITNET_CUDA_PREFILL_TOKENS='128',RBITNET_CUDA_SPLIT_KV='0',RBITNET_CUDA_PREFILL_TF32X3='0',
 RBITNET_CUDA_RESIDENT_GRAPH='1',RBITNET_CUDA_QWEN_FULL='1',RBITNET_REQUIRE_QWEN_FULL='1',RBITNET_CUDA_QWEN_PREFILL='1',
 RBITNET_CUDA_CONTINUOUS='0',RBITNET_QWEN_SPECULATIVE='0',RBITNET_SPECULATIVE_PLD='0',
 RBITNET_CONTEXT_DIR=str(out/'cache'),RBITNET_CONTEXT_RAM_MB='128',RBITNET_CONTEXT_DISK_MB='512', RBITNET_CONTEXT_DISK_GLOBAL_MB='512',
 RBITNET_CONTEXT_ENTRIES='8',RBITNET_CONTEXT_TTL_SECS='1800',RBITNET_MAX_CONCURRENT='1')
def children(parent):
    result={}
    for child in psutil.Process(parent.pid).children():
        if Path(child.exe()).resolve()==args.runner.resolve():
            env=child.environ();model=env['RBITNET_ACTIVE_MODEL_ID']
            result[model]=dict(pid=child.pid,base='http://'+env['RBITNET_BIND'])
    return result
def metrics(child):
    import re
    text=requests.get(child['base']+'/metrics',timeout=10);text.raise_for_status()
    return {name:float(value)for name,value in re.findall(r'^(rbitnet_\w+) ([0-9.]+)$',text.text,re.M)}
def request(parent,model,omit=False):
    body=references[model]['request'].copy()
    if omit:body.pop('model')
    started=time.perf_counter()
    response=requests.post('http://127.0.0.1:18138/v1/chat/completions',json=body,headers={'x-rbitnet-session':model+'-session'},timeout=600)
    if not response.ok:
        report.setdefault('errors',[]).append(dict(model=model,request=body,status=response.status_code,body=response.text,headers=dict(response.headers)));save()
    response.raise_for_status();row=response.json()
    assert identity(row)==identity(references[model]['response']),(model,row)
    assert response.headers.get('x-rbitnet-session')==model+'-session'
    running=children(parent);assert model in running
    catalog=requests.get(running[model]['base']+'/v1/models',timeout=10);catalog.raise_for_status()
    assert catalog.json()['data'][0]['id']==model,(model,catalog.text)
    assert row['model']==model,(model,row)
    return dict(model=model,omitted_model=omit,request=body,response=row,headers=dict(response.headers),
                wall_ms=(time.perf_counter()-started)*1000,runners=running,model_metrics=metrics(running[model]))
def phase(label,tiers,recycle):
    environment={name:value for name,value in os.environ.items()if not name.startswith('RBITNET_')}
    environment.update(common,RBITNET_CONTEXT_TIERS=str(int(tiers)),RAYON_NUM_THREADS='16')
    sampler=MemorySampler();log=(out/(label+'.log')).open('w',encoding='utf-8')
    parent=subprocess.Popen([str(args.proxy.resolve())],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    sampler.pid=parent.pid;sampler.thread.start()
    try:
        deadline=time.monotonic()+60
        while True:
            assert parent.poll()is None,('proxy exited',parent.returncode)
            try:
                response=requests.get('http://127.0.0.1:18138/health',timeout=1);response.raise_for_status();break
            except requests.RequestException:
                assert time.monotonic()<deadline;time.sleep(.1)
        for model in models:
            identifier=model['id'];row=request(parent,identifier)
            if tiers:
                name='disk_hits'if label=='process-restart'else'captures'
                assert row['model_metrics']['rbitnet_core_context_'+name+'_total']==1,(label,row)
            report['cases'].append(dict(phase=label,kind='explicit-model',record=row));save()
            sticky=request(parent,identifier,True)
            report['cases'].append(dict(phase=label,kind='session-affinity',record=sticky));save()
        if recycle:
            before=children(parent);assert len(before)==2
            time.sleep(2.1)
            for model in models:
                identifier=model['id'];row=request(parent,identifier,True)
                assert row['runners'][identifier]['pid']!=before[identifier]['pid'],'idle runner must actually be replaced'
                assert row['model_metrics']['rbitnet_core_context_disk_hits_total']==1
                report['cases'].append(dict(phase=label,kind='idle-reload',record=row,previous_pid=before[identifier]['pid']));save()
    finally:
        # Snapshot only this test's direct runner children, and terminate only
        # those whose exact executable matches the supplied runner artifact.
        owned=children(parent)if parent.poll()is None else{}
        if parent.poll()is None:parent.terminate();parent.wait(timeout=15)
        for row in owned.values():
            try:
                process=psutil.Process(row['pid'])
                if Path(process.exe()).resolve()==args.runner.resolve():process.terminate();process.wait(timeout=15)
            except psutil.NoSuchProcess:pass
        report['memory'][label]=sampler.finish();log.close();save()
phase('reference-proxy',False,False)
phase('capture-and-idle',True,True)
phase('process-restart',True,False)
report['complete']=True;save()
print('CONTEXT_TIERS_PROXY_DONE models=2 sticky_sessions=true idle_reload=true proxy_process_restart=true exact=true',flush=True)
