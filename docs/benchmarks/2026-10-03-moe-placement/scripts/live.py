"""Actual CPU/adaptive MoE lifecycle, scoped metrics and streaming proof."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
from unittest.mock import patch
import requests
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
base=root/'target/moe-placement/live';base.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'target/moe-placement/validation-manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=root/'target/release/rbitnet.exe';lib=root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'
assert sha(binary)==manifest['cli_sha256'] and sha(lib)==manifest['dll_sha256']
assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items())
frozen=base/'rbitnet.exe';frozen.write_bytes(binary.read_bytes())
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text())
config.update(rbitnet=str(frozen),cuda_quant_library=str(lib),port=18130)
report={'manifest':manifest,'cases':[]}
def save():(base/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
token='moe-placement-local';headers={'x-rbitnet-admin-token':token}
for model_id in ['gpt-oss-20b','glm47-flash']:
  model=next(m for m in config['models']if m['id']==model_id)
  for policy in ['cpu','adaptive']:
    label=model_id+'-'+policy
    config['rbitnet_env']=dict(RBITNET_MAX_SEQ='2048',RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',
      RBITNET_MOE_CACHE_MB='8192',RBITNET_MOE_EXECUTION=policy,RBITNET_CUDA_SPLIT_KV='1',
      RBITNET_CUDA_GPT_FULL='1'if model_id=='gpt-oss-20b'else'0',RBITNET_REQUIRE_GPT_FULL='1'if model_id=='gpt-oss-20b'else'0',
      RBITNET_CUDA_MLA_FULL='1'if model_id=='glm47-flash'else'0',RBITNET_REQUIRE_MLA_FULL='1'if model_id=='glm47-flash'else'0',
      RBITNET_PREFIX_KV='1',RBITNET_CUDA_PREFIX_MB='256',RBITNET_CUDA_PREFIX_ENTRIES='8',RBITNET_ADMIN_TOKEN=token)
    server=Server(config,model,'rbitnet','gpu',base);server.log_path=base/(label+'.log')
    case={'label':label,'env':config['rbitnet_env'].copy()}
    body=dict(model=model_id,messages=[dict(role='user',content='Quelle est la capitale de la France ? Réponds en un mot.')],temperature=0,max_tokens=64)
    original=subprocess.Popen
    def natural(*args,**kwargs):
      if kwargs.get('env'):
        kwargs['env']=kwargs['env'].copy()
        for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:kwargs['env'].pop(key,None)
      return original(*args,**kwargs)
    def state():
      response=requests.get(server.base+'/metrics',timeout=10);response.raise_for_status();raw=response.text
      metrics=server.metrics()
      cats={k:int(v)for k,v in re.findall(r'^rbitnet_core_cuda_managed_category_bytes\{category="([^\"]+)"\} ([0-9]+)$',raw,re.M)}
      scoped=[dict(name=n,labels=labels,value=int(v))for n,labels,v in re.findall(r'^(rbitnet_moe_layer_\w+)\{([^\n]+)\} ([0-9]+)$',raw,re.M)]
      ids=set(re.findall(r'model_id="([0-9]+)"',raw))
      assert sum(cats.values())==metrics['rbitnet_core_cuda_managed_live_bytes']
      assert metrics['rbitnet_core_cuda_managed_live_bytes']<=metrics['rbitnet_core_cuda_managed_peak_bytes']<=metrics['rbitnet_core_cuda_managed_limit_bytes']<=12288*2**20
      return dict(metrics=metrics,categories=cats,scoped=scoped,model_ids=sorted(ids),prometheus=raw)
    def complete():
      response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status();return response.json()
    try:
      with patch('subprocess.Popen',natural):server.start()
      case['models']=requests.get(server.base+'/v1/models',timeout=10).json()
      case['response']=complete();text=case['response']['choices'][0]['message']['content'];assert 'Paris'in text
      case['before_unload']=state();s=case['before_unload'];assert len(s['model_ids'])==1
      totals={name:sum(row['value']for row in s['scoped']if row['name']==name)for name in {row['name']for row in s['scoped']}}
      assert totals['rbitnet_moe_layer_fallback_ffns_total']>0
      if policy=='cpu':
        assert totals['rbitnet_moe_layer_gpu_ffns_total']==0
        assert totals['rbitnet_moe_layer_cache_ready_bytes']==0 and s['categories']['experts']==0
      else:
        assert totals['rbitnet_moe_layer_gpu_ffns_total']>0 and totals['rbitnet_moe_layer_gpu_decisions_total']>0
        assert totals['rbitnet_moe_layer_cache_ready_bytes']<=s['categories']['experts']<=8192*2**20
      for stage in ['unload','reload','unload']:
        response=requests.post(server.base+'/v1/admin/'+stage,json={}if stage=='reload'else None,headers=headers,timeout=180);response.raise_for_status()
        if stage=='reload':
          case['reload_response']=complete();assert case['reload_response']['choices'][0]['message']['content']==text
          case['reloaded']=state();assert len(case['reloaded']['model_ids'])==1 and case['reloaded']['model_ids']!=s['model_ids']
        else:
          released=state();assert not released['scoped'] and not released['model_ids']
          assert all(v==0 for k,v in released['categories'].items()if k!='scratch')
          case.setdefault('released',[]).append(released)
          response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=30);assert response.status_code==503
      print(label,'scoped cache, placement, unload and reload passed',flush=True)
    finally:case['memory']=server.close();report['cases'].append(case);save()
    folder=base/(label+'-streaming')
    command=[sys.executable,'scripts/validate_cache_streaming.py','--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
      '--binary',str(frozen),'--library',str(lib),'--output-dir',str(folder),'--port','18130','--split-kv','--moe-cache','8192',
      '--moe-execution',policy,'--device-mib','12288']
    if model_id=='gpt-oss-20b':command+=['--gpt-full','--gpt-segmented']
    else:command+=['--mla-full']
    with(base/(label+'-streaming.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,(label,r.returncode)
    print(label,'disconnect, sampling, penalties, stop and serialized concurrent requests passed',flush=True)
assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items())
print('All four MoE live policy/metrics/lifecycle/streaming suites passed.',flush=True)
