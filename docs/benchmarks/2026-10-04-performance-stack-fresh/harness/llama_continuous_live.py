"""Actual loopback requests and quiet arrival waves for the private Llama worker."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,statistics,sys,threading,time
import requests
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
p.add_argument('--reference-binary',type=Path,required=True);p.add_argument('--reference-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
model=next(row for row in config['models']if row['id']=='llama32-1b')
config.update(port=18138,cwd=str(root),startup_timeout=300)
common=dict(RBITNET_MAX_SEQ='2048',RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_KV_FORMAT='f32',
 RBITNET_CUDA_PREFILL='1',RBITNET_CUDA_PREFILL_TOKENS='128',RBITNET_CUDA_SPLIT_KV='0',RBITNET_CUDA_PREFILL_TF32X3='0',RBITNET_CUDA_RESIDENT_GRAPH='1',
 RBITNET_PREFIX_KV='0',RBITNET_MTP_K='1',RBITNET_SPECULATIVE='0',RBITNET_SPECULATIVE_PLD='0',RBITNET_MAX_CONCURRENT='16',RBITNET_INFERENCE_TIMEOUT_SECS='600')
prompts=['Write a detailed story about a robot exploring a forest. Begin the story immediately.',
 ('The library has gardens, museums and visitors.\n'*36)+'Return a Python checked_sum function and a worked example.',
 'Quelle est la capitale de la France ? Réponds uniquement Paris.']
bodies=[]
def chat(question):
 return '<|start_header_id|>user<|end_header_id|>\n\n'+question+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
for i in range(8):
 sampling=[dict(temperature=0,seed=42),dict(temperature=.7,top_p=.9,seed=53),dict(temperature=0,seed=64,frequency_penalty=.2,presence_penalty=.1)][i%3]
 bodies.append(dict(model=model['id'],messages=[dict(role='user',content=chat(prompts[i%3]))],max_tokens=128 if i%3!=2 else 33,**sampling))
bodies.append(dict(model=model['id'],messages=[dict(role='user',content=chat('Le jardin.'))],max_tokens=0,temperature=0))
report=dict(binary_sha256=sha(args.binary),library_sha256=sha(args.library),reference_binary_sha256=sha(args.reference_binary),
 reference_library_sha256=sha(args.reference_library),protocol=dict(repeated_wave_cycles=3,warmup_cycles=1,arrival_delay_ms=20,
 max_tokens=128,context=2048,kv='f32',graphs=True,prefix=False,split_kv=False,notes='Client SSE intervals concern text chunks; no equivalence to individual token intervals is assumed.'),reference=[],cases=[],waves=[],memory={})
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def identity(row):
 choice=row['choices'][0];return choice['message']['content'],choice['finish_reason'],row['usage']['completion_tokens']
def complete(server,body):
 start=time.perf_counter();r=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);r.raise_for_status();row=r.json()
 assert 'error'not in row,row
 return row,(time.perf_counter()-start)*1000
def stream(server,body,disconnect=False):
 start=time.perf_counter();times=[];pieces=[];events=[];terminal=[];done=0
 with requests.post(server.base+'/v1/chat/completions',json=body|dict(stream=True),stream=True,timeout=600)as response:
  response.raise_for_status()
  for raw in response.iter_lines(chunk_size=1):
   if not raw.startswith(b'data: '):continue
   raw=raw[6:]
   if raw==b'[DONE]':done+=1;continue
   event=json.loads(raw);assert 'error'not in event,event;events.append(event);choice=event['choices'][0]
   if choice.get('finish_reason')is not None:terminal.append(choice['finish_reason'])
   piece=choice.get('delta',{}).get('content','')
   if piece:
    pieces.append(piece);times.append((time.perf_counter()-start)*1000)
    if disconnect and len(pieces)==3:break
 text=''.join(pieces);assert '\ufffd'not in text,text
 if disconnect:assert len(pieces)==3 and done==0
 else:assert done==1 and len(terminal)==1,(done,terminal)
 return dict(text=text,finish_reason=terminal[0]if terminal else None,ttft_ms=times[0]if times else None,
  chunk_intervals_ms=[b-a for a,b in zip(times,times[1:])],wall_ms=(time.perf_counter()-start)*1000,events=events,client_disconnected=disconnect)
def server_for(label,binary,library,slots,pages):
 settings=common|dict(RBITNET_CUDA_CONTINUOUS='0'if slots is None else'1',RBITNET_CUDA_CONTINUOUS_SLOTS=str(slots or 1),
  RBITNET_CUDA_CONTINUOUS_QUEUE='32',RBITNET_CUDA_CONTINUOUS_TOKEN_BUDGET='256',RBITNET_CUDA_CONTINUOUS_ORDERING='0',RBITNET_CUDA_KV_PAGE_LIMIT='0'if pages is None else str(pages))
 cfg=config|dict(rbitnet=str(binary.resolve()),cuda_quant_library=str(library.resolve()),rbitnet_env=settings)
 server=Server(cfg,model,'rbitnet','gpu',out);server.log_path=out/(label+'.log');return server
reference=server_for('reference',args.reference_binary,args.reference_library,None,None)
try:
 reference.start()
 for index,body in enumerate(bodies):
  row,wall=complete(reference,body);report['reference'].append(dict(index=index,request=body,response=row,wall_ms=wall));save()
 for cycle in range(3):
  before=reference.metrics();started=time.perf_counter();barrier=threading.Barrier(8)
  def reference_task(index):
   barrier.wait()
   if index%2:time.sleep(.02)
   result=stream(reference,bodies[index])
   assert (result['text'],result['finish_reason'])==identity(report['reference'][index]['response'])[:2]
   return dict(index=index,request=bodies[index],stream=result)
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:rows=list(pool.map(reference_task,range(8)))
  wall=(time.perf_counter()-started)*1000;after=reference.metrics()
  count=sum(identity(report['reference'][row['index']]['response'])[2]for row in rows)
  report['waves'].append(dict(config='reference',cycle=cycle,warmup=cycle==0,wall_ms=wall,completion_tokens=count,aggregate_tokens_per_second=count*1000/wall,
    actual_native_metrics_delta={key:after.get(key,0)-before.get(key,0)for key in set(after)|set(before)},metrics_after=after,requests=rows));save()
finally:report['memory']['reference']=reference.close();save()
assert any(identity(row['response'])[1]=='stop'and identity(row['response'])[0].strip()for row in report['reference']),'a genuine nonempty EOS reference is required'
for pages in [None,512]:
 for slots in [1,4,8]:
  label=('dense'if pages is None else'paged')+'-slots'+str(slots)
  candidate=server_for(label,args.binary,args.library,slots,pages)
  try:
   candidate.start()
   for index,body in enumerate(bodies):
    row,wall=complete(candidate,body);sse=stream(candidate,body)
    assert identity(row)==identity(report['reference'][index]['response']),(label,index,row)
    assert (sse['text'],sse['finish_reason'])==identity(row)[:2],(label,index,sse)
    report['cases'].append(dict(config=label,kind='serial-identity',index=index,request=body,response=row,stream=sse,wall_ms=wall));save()
   for cycle in range(3):
    before=candidate.metrics();started=time.perf_counter();barrier=threading.Barrier(8)
    def task(index):
     barrier.wait()
     if index%2:time.sleep(.02)
     body=bodies[index];sse=stream(candidate,body)
     assert (sse['text'],sse['finish_reason'])==identity(report['reference'][index]['response'])[:2],(label,index,sse)
     return dict(index=index,request=body,stream=sse)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:rows=list(pool.map(task,range(8)))
    wall=(time.perf_counter()-started)*1000;after=candidate.metrics()
    delta={key:after.get(key,0)-before.get(key,0)for key in set(after)|set(before)}
    assert delta['rbitnet_core_gpu_llama_batch_waves_total']>0
    if slots>1:assert delta['rbitnet_core_gpu_llama_batch_rows_total']>delta['rbitnet_core_gpu_llama_batch_waves_total'],(label,delta)
    count=sum(identity(report['reference'][row['index']]['response'])[2]for row in rows)
    report['waves'].append(dict(config=label,cycle=cycle,warmup=cycle==0,wall_ms=wall,completion_tokens=count,aggregate_tokens_per_second=count*1000/wall,
      actual_native_metrics_delta=delta,metrics_after=after,requests=rows));save()
   # One disconnected HTTP owner runs concurrently with a surviving stream.
   with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
    stopped=pool.submit(stream,candidate,bodies[0],True);survivor=pool.submit(stream,candidate,bodies[1])
    interrupted=stopped.result();survived=survivor.result()
   assert (survived['text'],survived['finish_reason'])==identity(report['reference'][1]['response'])[:2]
   resumed,_=complete(candidate,bodies[0]);assert identity(resumed)==identity(report['reference'][0]['response'])
   report['cases'].append(dict(config=label,kind='owned-disconnect-survivor',disconnected=interrupted,survivor=survived,resumed=resumed));save()
   visible=identity(report['reference'][2]['response'])[0];assert visible.strip()
   stop_body=bodies[2]|dict(stop=visible[:2]);stopped,_=complete(candidate,stop_body);sse=stream(candidate,stop_body)
   assert identity(stopped)[:2]==(sse['text'],sse['finish_reason'])==('','stop')
   report['cases'].append(dict(config=label,kind='client-stop',request=stop_body,response=stopped,stream=sse));save()
  finally:report['memory'][label]=candidate.close();save()
report['complete']=True;save()
print('LLAMA_CONTINUOUS_HTTP_DONE layouts=2 capacities=3 serial_cases=54 waves=21 owned_disconnects=6 explicit_stops=6',flush=True)
