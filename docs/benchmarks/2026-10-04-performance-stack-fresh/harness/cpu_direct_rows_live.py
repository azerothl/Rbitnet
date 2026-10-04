"""Quiet same-binary CPU old/direct rows, actual JSON/SSE and seeded sampling."""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import subprocess
import sys
import time

import requests

sys.path.insert(0,str(Path.cwd()/'scripts'))
from benchmark_engines import Server
from atomic_journal import write_json

p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(exist_ok=True)
config=json.loads(Path('docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
config.update(rbitnet=str(args.binary.resolve()),port=18138)
report=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            protocol=dict(backend='cpu',cycles=3,warmup_cycle=0,threads=16,max_tokens=128,context_capacity=512,
                          same_binary=True,hardware_tests_serial=True),rows=[],sse=[],stops=[],memory={},complete=False)
original=subprocess.Popen
prompts=[
 [{'role':'user','content':'Écris un récit de 150 mots sur un robot qui explore une bibliothèque.'}],
 [{'role':'user','content':'Écris une fonction Python qui calcule une moyenne en ignorant les valeurs None et explique le cas d’une liste vide.'}],
 [{'role':'user','content':'Quelle est la capitale de la France ? Réponds en un mot.'}],
]
for model in config['models']:
 reference={};reference_sse={}
 for mode in ['original','direct']:
  folder=args.output/(model['id']+'-'+mode);folder.mkdir(exist_ok=True)
  overrides=dict(RBITNET_CPU_DIRECT_ROWS='1'if mode=='direct'else'0',RBITNET_MAX_SEQ='512',RBITNET_PREFIX_KV='0',
    RBITNET_BACKEND='cpu',RBITNET_QUANT_KERNEL='auto',RBITNET_CUDA='0',RBITNET_CPU_SIMD_QUANT='1',RBITNET_CPU_AVX512='1',
    RBITNET_MOE_CACHE_MB='0',RBITNET_MOE_ASYNC='0',RBITNET_MOE_ARENA='0',RBITNET_SPECULATIVE='0',RBITNET_SPECULATIVE_PLD='0',
    RBITNET_MAX_CONCURRENT='1',RBITNET_INFERENCE_TIMEOUT_SECS='1800')
  def popen(*a,**kw):
   if kw.get('env'):
    kw['env']=kw['env'].copy();kw['env'].update(overrides)
    for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:kw['env'].pop(key,None)
   return original(*a,**kw)
  server=Server(config,model,'rbitnet','cpu',folder)
  try:
   with patch('subprocess.Popen',popen):server.start()
   for cycle in range(3):
    for index,messages in enumerate(prompts):
     body=dict(model=model['id'],messages=messages,max_tokens=128,temperature=0)
     before=server.metrics();started=time.perf_counter()
     response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=1800);response.raise_for_status()
     raw=response.json();wall=1000*(time.perf_counter()-started);after=server.metrics()
     text=raw['choices'][0]['message']['content'];assert text and '\ufffd'not in text
     if mode=='original':reference[cycle,index]=text
     assert text==reference[cycle,index],(model['id'],mode,cycle,index,'different actual output')
     delta={key:after.get(key,0)-before.get(key,0)for key in set(after)|set(before)}
     calls=delta.get('rbitnet_core_cpu_direct_row_calls_total',0)
     assert (calls>0)if mode=='direct'else(calls==0),(model['id'],mode,'new CPU path was not actually exercised',calls)
     report['rows'].append(dict(model=model['id'],mode=mode,cycle=cycle,prompt=index,request=body,response=raw,
         wall_ms=wall,metrics_delta=delta,env=overrides,matches_reference=True))
     write_json(args.output/'results.json',report)
     print('CPU_DIRECT_QUIET',model['id'],mode,cycle,index,'row_calls',calls,flush=True)
   for label,options in [('greedy',dict(temperature=0)),('sampled',dict(temperature=.7,top_p=.9,seed=42,frequency_penalty=.2,presence_penalty=.1))]:
    body=dict(model=model['id'],messages=[{'role':'user','content':'Répète exactement : été, café, résumé, 🙂.'}],max_tokens=128,**options)
    response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=1800);response.raise_for_status()
    raw=response.json();text=raw['choices'][0]['message']['content'];assert text and '\ufffd'not in text
    if mode=='original':reference_sse[label]=text
    assert text==reference_sse[label],(model['id'],mode,label,'sampled output differs')
    parts=[];done=False;first_ms=None;started=time.perf_counter()
    stream=requests.post(server.base+'/v1/chat/completions',json=body|dict(stream=True),stream=True,timeout=1800);stream.raise_for_status()
    try:
     for line in stream.iter_lines(chunk_size=1):
      line=line.decode('utf-8')
      if line=='data: [DONE]':done=True
      elif line.startswith('data: '):
       content=json.loads(line[6:]).get('choices',[{}])[0].get('delta',{}).get('content','')
       if content and first_ms is None:first_ms=1000*(time.perf_counter()-started)
       parts.append(content)
    finally:stream.close()
    assert done and ''.join(parts)==text and first_ms is not None
    report['sse'].append(dict(model=model['id'],mode=mode,sampling=label,text=text,sse_text=''.join(parts),
                              first_content_ms=first_ms,done=done,matches_reference=True))
    write_json(args.output/'results.json',report)
   body=dict(model=model['id'],messages=prompts[2],max_tokens=128,temperature=0,stop=['Paris'])
   response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=1800);response.raise_for_status()
   raw=response.json();assert 'Paris'not in raw['choices'][0]['message']['content']
   report['stops'].append(dict(model=model['id'],mode=mode,response=raw))
  finally:
   report['memory'][model['id']+'-'+mode]=server.close();write_json(args.output/'results.json',report)
assert len(report['rows'])==72 and len(report['sse'])==16 and len(report['stops'])==8
report['complete']=True;write_json(args.output/'results.json',report)
print('CPU_DIRECT_HTTP_DONE actual_models=4 rows=72 sse=16 stops=8',flush=True)
