"""Actual frozen async cache unload/reload and scoped pool/pinned accounting."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
from unittest.mock import patch
import requests
root=Path.cwd();base=root/'target/performance-cache';gate=base/'followups-chain.log';deadline=time.monotonic()+28800
while not gate.exists()or 'All actual context/page/async prototype network and quiet measurements passed; adoption and production revalidation still pending.'not in gate.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'quiet/live experiments not complete; no competing lifecycle inference started'
 time.sleep(15)
sys.path.insert(0,str(root/'scripts'));from benchmark_engines import Server
out=base/'async-lifecycle';out.mkdir(exist_ok=True);proof=base/'async-current-proof'
m=json.loads((proof/'manifest.json').read_text(encoding='utf-8'));binary=proof/'rbitnet.exe';lib=root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(binary)==m['cli_sha256']and sha(lib)==m['library_sha256']
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'));config.update(rbitnet=str(binary),cuda_quant_library=str(lib),port=18138)
report={'binary_sha256':sha(binary),'library_sha256':sha(lib),'budget_mib':512,'cases':[]}
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
token='async-cache-local-proof';headers={'x-rbitnet-admin-token':token}
for model_id in ['gpt-oss-20b','glm47-flash']:
 model=next(x for x in config['models']if x['id']==model_id)
 config['rbitnet_env']={'RBITNET_MAX_SEQ':'2048','RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_MOE_CACHE_MB':'512','RBITNET_MOE_EXECUTION':'cache',
  'RBITNET_MOE_ASYNC':'1','RBITNET_MOE_PREFETCH':'previous-pass','RBITNET_MOE_PINNED_SLOTS':'2','RBITNET_CUDA_SPLIT_KV':'1','RBITNET_PREFIX_KV':'1','RBITNET_CUDA_PREFIX_MB':'256','RBITNET_CUDA_PREFIX_ENTRIES':'8','RBITNET_ADMIN_TOKEN':token,
  'RBITNET_CUDA_GPT_FULL':'1'if model_id=='gpt-oss-20b'else'0','RBITNET_REQUIRE_GPT_FULL':'1'if model_id=='gpt-oss-20b'else'0','RBITNET_CUDA_GPT_SEGMENTED':'1'if model_id=='gpt-oss-20b'else'0',
  'RBITNET_CUDA_MLA_FULL':'1'if model_id=='glm47-flash'else'0','RBITNET_REQUIRE_MLA_FULL':'1'if model_id=='glm47-flash'else'0'}
 server=Server(config,model,'rbitnet','gpu',out);server.log_path=out/(model_id+'.log');case={'model':model_id,'env':config['rbitnet_env'].copy()};report['cases'].append(case);save()
 body={'model':model_id,'messages':[{'role':'user','content':'Quelle est la capitale de la France ? Réponds en un mot.'}],'max_tokens':32,'temperature':0}
 original=subprocess.Popen
 def natural(*args,**kwargs):
  if kwargs.get('env'):
   kwargs['env']=kwargs['env'].copy()
   for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:kwargs['env'].pop(key,None)
  return original(*args,**kwargs)
 def state():
  response=requests.get(server.base+'/metrics',timeout=10);response.raise_for_status();raw=response.text;metrics=server.metrics()
  cats={k:int(v)for k,v in re.findall(r'^rbitnet_core_cuda_managed_category_bytes\{category="([^\"]+)"\} ([0-9]+)$',raw,re.M)}
  rows=[{'name':n,'labels':labels,'value':int(v)}for n,labels,v in re.findall(r'^(rbitnet_moe_(?:layer|model)_\w+)\{([^\n]+)\} ([0-9]+)$',raw,re.M)]
  ids=sorted(set(re.findall(r'model_id="([0-9]+)"',raw)))
  assert sum(cats.values())==metrics['rbitnet_core_cuda_managed_live_bytes']
  assert metrics['rbitnet_core_cuda_managed_live_bytes']<=metrics['rbitnet_core_cuda_managed_peak_bytes']<=metrics['rbitnet_core_cuda_managed_limit_bytes']<=12288*2**20
  totals={name:sum(x['value']for x in rows if x['name']==name)for name in {x['name']for x in rows}}
  return {'metrics':metrics,'categories':cats,'scoped':rows,'model_ids':ids,'totals':totals,'prometheus':raw}
 def complete():
  r=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);r.raise_for_status();return r.json()
 try:
  with patch('subprocess.Popen',natural):server.start()
  case['response']=complete();text=case['response']['choices'][0]['message']['content'];assert 'Paris'in text and '\ufffd'not in text
  case['before_unload']=s=state();t=s['totals'];save()
  assert len(s['model_ids'])==1 and t['rbitnet_moe_model_async_enabled']==1 and t['rbitnet_moe_model_async_failed']==0
  assert t['rbitnet_moe_model_pinned_slots']==2 and t['rbitnet_moe_model_pinned_bytes']>0
  assert t['rbitnet_moe_layer_gpu_ffns_total']>0 and t['rbitnet_moe_layer_copy_dma_ns_total']>0
  ready=t['rbitnet_moe_layer_cache_ready_bytes'];pending=t['rbitnet_moe_layer_pending_bytes'];pool=t['rbitnet_moe_model_async_pool_bytes']
  assert ready+pending<=pool<=s['categories']['experts']<=512*2**20
  for stage in ['unload','reload','unload']:
   r=requests.post(server.base+'/v1/admin/'+stage,json={}if stage=='reload'else None,headers=headers,timeout=180);r.raise_for_status()
   if stage=='reload':
    case['reload_response']=complete();assert case['reload_response']['choices'][0]['message']['content']==text
    case['reloaded']=reloaded=state();assert len(reloaded['model_ids'])==1 and reloaded['model_ids']!=s['model_ids']and reloaded['totals']['rbitnet_moe_model_async_failed']==0
   else:
    released=state();assert not released['scoped']and not released['model_ids']and all(v==0 for k,v in released['categories'].items()if k!='scratch')
    case.setdefault('released',[]).append(released)
    r=requests.post(server.base+'/v1/chat/completions',json=body,timeout=30);assert r.status_code==503
   save()
  print('Actual async bounded pool/pinned/scoped unload and reload:',model_id,'passed.',flush=True)
 finally:case['memory']=server.close();save()
print('Actual async GPT/GLM scoped READY/PENDING/pinned budget and unload/reload lifecycle suites passed.',flush=True)
