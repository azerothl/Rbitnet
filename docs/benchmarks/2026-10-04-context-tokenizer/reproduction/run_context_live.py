"""Actual network routes using the frozen context/SentencePiece prototype."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,shutil,sys
import requests
root=Path.cwd();base=root/'target/performance-cache';sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
proof=Path(os.environ.get('RBITNET_CONTEXT_PROOF_DIR',str(base/'context-real-proof')))
out=Path(os.environ.get('RBITNET_CONTEXT_LIVE_DIR',str(base/'context-live-proof')));out.mkdir(parents=True,exist_ok=True)
m=json.loads((proof/'manifest.json').read_text(encoding='utf-8'));binary=proof/'rbitnet.exe'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==m['binary_sha256']
libraries=[root/'target/gpt-block/cuda/rbitnet_cuda_quant64.dll',root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll']
library=next((p for p in libraries if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==m['library_sha256']),None)
assert library is not None,'no native library matches the frozen proof manifest'
source=Path('D:/Rbitnet-benchmark-models/mistral7b-sp');tok=out/'tokenizer.model';shutil.copy2(source/'tokenizer.model',tok)
model={'id':'mistral7b-sp','gguf':str(source/'mistral-7b-instruct-v0.1.Q4_K_M.gguf'),'tokenizer':str(tok.resolve())}
report={'binary_sha256':m['binary_sha256'],'library_sha256':m['library_sha256'],'library_path':str(library),'capacity':64,'cases':[]}
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for backend in ['cpu','gpu']:
 config={'rbitnet':str(binary.resolve()),'port':18138,'cwd':str(root),'threads':16,'cuda_quant_library':str(library),
  'rbitnet_env':{'RBITNET_MAX_SEQ':'64','RBITNET_MAX_CONCURRENT':'4','RBITNET_PREFIX_KV':'1','RBITNET_CUDA_SPLIT_KV':'1','RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256'}}
 server=Server(config,model,'rbitnet',backend,out)
 try:
  server.start();loaded=requests.get(server.base+'/v1/models',timeout=10).json()
  assert '"context_capacity": 64'in json.dumps(loaded),loaded
  record={'backend':backend,'loaded_model':loaded,'errors':[]};report['cases'].append(record);save()
  for path,body in [('/v1/chat/completions',{'messages':[{'role':'user','content':'Bonjour'}]}),('/v1/completions',{'prompt':'Bonjour'}),('/v1/messages',{'messages':[{'role':'user','content':'Bonjour'}]})]:
   for stream in [False,True]:
    req=dict(model=model['id'],max_tokens=65,temperature=0,stream=stream,**body)
    response=requests.post(server.base+path,json=req,timeout=60)
    assert response.status_code==400 and 'application/json'in response.headers['Content-Type'],(path,response.status_code,response.text)
    raw=response.json();assert 'capacity 64'in json.dumps(raw),raw
    record['errors'].append({'path':path,'request':req,'status':response.status_code,'response':raw});save()
  # Move only the test-owned file before lazy weight initialization. The server
  # must use its frozen tokenizer owner for both counting and generation.
  backup=out/'tokenizer.moved.model';tok.rename(backup)
  try:
   def complete(body):
    r=requests.post(server.base+'/v1/chat/completions',json=body,timeout=120);r.raise_for_status();return r.json()
   body={'model':model['id'],'messages':[{'role':'user','content':'[INST]Quelle est la capitale de la France ? Réponds en un mot.[/INST]'}],'max_tokens':12,'temperature':0}
   unary=complete(body);text=unary['choices'][0]['message']['content'];assert 'Paris'in text and '\ufffd'not in text
   count=unary['usage']['prompt_tokens'];assert 0<count<64
   exact=complete(body|{'max_tokens':64-count})
   overflow=requests.post(server.base+'/v1/chat/completions',json=body|{'max_tokens':65-count,'stream':True},timeout=60)
   assert overflow.status_code==400 and 'application/json'in overflow.headers['Content-Type']
   stream=requests.post(server.base+'/v1/chat/completions',json=body|{'stream':True},timeout=120);stream.raise_for_status();parts=[];done=False
   for line in stream.text.splitlines():
    if line=='data: [DONE]':done=True
    elif line.startswith('data: '):parts.append(json.loads(line[6:]).get('choices',[{}])[0].get('delta',{}).get('content',''))
   assert done and ''.join(parts)==text
   with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:parallel=list(pool.map(complete,[body]*4))
   assert all(x['choices'][0]['message']['content']==text for x in parallel)
   record.update(tokenizer_file_moved_before_first_generation=True,unary=unary,exact_capacity_response=exact,sse_text=''.join(parts),sse_done=done,concurrent_requests=parallel,metrics=server.metrics());save()
  finally:backup.rename(tok)
 finally:record_memory=server.close();report.setdefault('memory',{})[backend]=record_memory;save()
 print('Actual Mistral network capacity/tokenizer/error/SSE/concurrency:',backend,'passed.',flush=True)
print('Actual CPU/CUDA Mistral HTTP context, frozen tokenizer and response suites passed.',flush=True)
