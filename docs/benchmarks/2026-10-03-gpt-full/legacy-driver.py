import json,pathlib,sys,requests,hashlib
sys.path.insert(0,str(pathlib.Path('scripts').resolve()))
from benchmark_engines import Server
root=pathlib.Path('target/gpt-full/legacy');root.mkdir(parents=True,exist_ok=True)
c=json.loads(pathlib.Path('target/gpt-full/comparison/manifest-prelaunch.json').read_text())
c['cuda_quant_library']=str(pathlib.Path('target/qwen-block/cuda/rbitnet_cuda_quant64.dll').resolve());c['port']=18111
c['environment']['cuda_quant_library_sha256']=hashlib.sha256(pathlib.Path(c['cuda_quant_library']).read_bytes()).hexdigest()
c['rbitnet_env']['RBITNET_REQUIRE_GPT_FULL']='0'
m=next(x for x in c['models'] if x['id']=='gpt-oss-20b')
s=Server(c,m,'rbitnet','gpu',root)
try:
 s.start()
 body={'model':m['id'],'messages':[{'role':'user','content':'Quelle est la capitale de la France ? Réponds en un mot.'}],'temperature':0,'max_tokens':32}
 r=requests.post(s.base+'/v1/chat/completions',json=body,timeout=120);r.raise_for_status();answer=r.json()
 loaded=requests.get(s.base+'/v1/models',timeout=5).json();metrics=s.metrics()
 assert 'Paris' in answer['choices'][0]['message']['content']
 assert 'fully resident GPT-OSS token pipeline: false' in json.dumps(loaded)
 assert 'resident routed expert layers: 24' in json.dumps(loaded)
 assert metrics.get('rbitnet_core_gpu_gpt_full_tokens_total',0)==0
 assert metrics.get('rbitnet_core_gpu_upload_bytes_total',0)>0
 (root/'results.json').write_text(json.dumps(dict(config=c,request=body,response=answer,loaded=loaded,metrics=metrics),ensure_ascii=False,indent=2)+'\n')
 print('Missing GPT ABI: real Paris response, partial CUDA path, 24 expert layers, full-token counter zero')
finally:s.close()
