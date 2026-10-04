"""Actual GPT fixed/segmented/cache/hybrid/CPU lifecycle and old-ABI refusals."""
import argparse,hashlib,json,pathlib,re,sys
from unittest.mock import patch
import requests
sys.path.insert(0,str(pathlib.Path('scripts').resolve()))
from benchmark_engines import Server

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=pathlib.Path,required=True)
p.add_argument('--binary',type=pathlib.Path,required=True)
p.add_argument('--library',type=pathlib.Path,required=True)
p.add_argument('--legacy-library',type=pathlib.Path,required=True)
p.add_argument('--output-dir',type=pathlib.Path,required=True)
a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
binary=out/'rbitnet.exe';binary.write_bytes(a.binary.read_bytes())
config=json.loads(a.config.read_text(encoding='utf-8-sig'));config.update(rbitnet=str(binary.resolve()),port=18129)
model=next(m for m in config['models'] if m['id']=='gpt-oss-20b')
report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),cases=[])
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
token='gpt-segmented-lifecycle-local'
headers={'x-rbitnet-admin-token':token}
system='Tu es un assistant précis. Réponds en français.\n'+'\n'.join(f'Note {i}: Les villes ont des bibliothèques, des jardins et des musées.' for i in range(12))
body=dict(model=model['id'],messages=[dict(role='system',content=system),dict(role='user',content='Quelle est la capitale de la France ? Réponds en un mot.')],temperature=0,max_tokens=64)
# label/cache/cap/library/required/refuse/force-segments/expected-pipeline/segments/snapshots/backend
for label,cache,cap,library,required,refuse,force,available,segmented,prefix,backend in [
    ('fixed',0,12288,a.library,True,False,False,True,False,True,'cuda'),
    ('forced-segments',0,12288,a.library,True,False,True,True,True,True,'cuda'),
    ('cached',8192,12288,a.library,True,False,False,True,True,True,'cuda'),
    ('cpu-routed',16,6144,a.library,True,False,False,True,True,True,'cuda'),
    ('partial-fixed',0,6144,a.library,True,False,False,True,True,True,'cuda'),
    ('hybrid-cached',8192,12288,a.library,True,False,False,True,True,True,'hybrid'),
    ('legacy-fixed',0,12288,a.legacy_library,True,False,False,True,False,False,'cuda'),
    ('legacy-optional',8192,12288,a.legacy_library,False,False,False,False,False,False,'cuda'),
    ('legacy-required',8192,12288,a.legacy_library,True,True,False,False,False,False,'cuda'),
    ('insufficient-state',16,64,a.library,True,True,False,False,False,False,'cuda')]:
    config['cuda_quant_library']=str(library.resolve())
    config['rbitnet_env']=dict(RBITNET_BACKEND=backend,RBITNET_HYBRID_MAX_VRAM_MB='12288',
        RBITNET_MAX_SEQ='2048',RBITNET_CUDA_GPT_FULL='1',RBITNET_REQUIRE_GPT_FULL='1' if required else '0',
        RBITNET_CUDA_GPT_SEGMENTED='1' if force else '0',RBITNET_CUDA_GPT_FULL_GRAPH='1',RBITNET_CUDA_SPLIT_KV='1',RBITNET_MOE_CACHE_MB=str(cache),
        RBITNET_CUDA_DEVICE_BUDGET_MB=str(cap),RBITNET_CUDA_DEVICE_MARGIN_MB='256',
        RBITNET_PREFIX_KV='1',RBITNET_CUDA_PREFIX_MB='256',RBITNET_CUDA_PREFIX_ENTRIES='8',RBITNET_ADMIN_TOKEN=token)
    server=Server(config,model,'rbitnet','gpu',out);server.log_path=out/(label+'.log')
    case=dict(label=label,env=config['rbitnet_env'].copy(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    def managed():
        text=requests.get(server.base+'/metrics',timeout=10);text.raise_for_status()
        metrics=server.metrics();cats={k:int(v) for k,v in re.findall(r'^rbitnet_core_cuda_managed_category_bytes\{category="([^"]+)"\} ([0-9]+)$',text.text,re.M)}
        assert metrics['rbitnet_core_cuda_managed_memory_available']==1
        assert sum(cats.values())==metrics['rbitnet_core_cuda_managed_live_bytes']
        assert metrics['rbitnet_core_cuda_managed_live_bytes']<=metrics['rbitnet_core_cuda_managed_peak_bytes']<=metrics['rbitnet_core_cuda_managed_limit_bytes']<=cap*2**20
        return dict(metrics=metrics,categories=cats)
    original=__import__('subprocess').Popen
    def natural(*args,**kwargs):
        if kwargs.get('env'):
            kwargs['env']=kwargs['env'].copy()
            for k in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:kwargs['env'].pop(k,None)
        return original(*args,**kwargs)
    try:
        try:
            with patch('subprocess.Popen',natural):server.start()
        except RuntimeError as error:
            assert refuse,(label,str(error));case['startup_error']=str(error)
        else:assert not refuse,label
        if refuse:
            ready=requests.get(server.base+'/ready',timeout=10);assert ready.status_code==503 and 'LoadFailed' in ready.text
            case['ready']=dict(status=ready.status_code,text=ready.text);case['routes']=[]
            completion=dict(model=model['id'],prompt='Bonjour',max_tokens=8)
            anthropic=dict(model=model['id'],messages=body['messages'],max_tokens=8)
            for route,request in [('/v1/chat/completions',body),('/v1/completions',completion),('/v1/messages',anthropic)]:
                for stream in [False,True]:
                    response=requests.post(server.base+route,json={**request,'stream':stream},timeout=30)
                    assert response.status_code==503 and response.json()['error']['code']=='LoadFailed'
                    case['routes'].append(dict(path=route,stream=stream,status=response.status_code,response=response.json()))
            case['managed']=managed();assert all(v==0 for k,v in case['managed']['categories'].items() if k!='scratch')
        else:
            loaded=requests.get(server.base+'/v1/models',timeout=10).json();case['loaded']=loaded
            full='resident GPT-OSS attention/router/head: true' in json.dumps(loaded);assert full==available,loaded
            if full:
                assert ('fully resident GPT-OSS token pipeline: false' in json.dumps(loaded))==segmented,loaded
                assert ('prefix snapshot ABI: true' in json.dumps(loaded))==prefix,loaded
            response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status()
            case['response']=response.json();text=case['response']['choices'][0]['message']['content'];assert 'Paris' in text
            case['before_unload']=managed()
            m=case['before_unload']['metrics'];assert (m.get('rbitnet_core_gpu_gpt_full_tokens_total',0)>0)==available
            if prefix:assert case['before_unload']['categories']['prefix']>0
            if label=='cpu-routed':assert m['rbitnet_core_native_moe_fallback_layers_total']>0 and m['rbitnet_core_native_moe_resident_layers_total']==0
            if label=='partial-fixed':assert m['rbitnet_core_native_moe_fallback_layers_total']>0 and m['rbitnet_core_native_moe_resident_layers_total']>0
            response=requests.post(server.base+'/v1/admin/unload',headers=headers,timeout=60);response.raise_for_status()
            case['unloaded']=managed();assert all(v==0 for k,v in case['unloaded']['categories'].items() if k!='scratch')
            unavailable=requests.post(server.base+'/v1/chat/completions',json=body,timeout=30);assert unavailable.status_code==503
            response=requests.post(server.base+'/v1/admin/reload',json={},headers=headers,timeout=180);response.raise_for_status()
            response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status()
            case['reload_response']=response.json();assert case['reload_response']['choices'][0]['message']['content']==text
            case['reloaded']=managed()
            response=requests.post(server.base+'/v1/admin/unload',headers=headers,timeout=60);response.raise_for_status()
            case['released_again']=managed();assert all(v==0 for k,v in case['released_again']['categories'].items() if k!='scratch')
        print(label,'passed GPT ABI, output and managed memory checks',flush=True)
    finally:case['memory']=server.close();report['cases'].append(case);save()
