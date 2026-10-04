"""Actual four-model CPU/CUDA refusals without opening SSE or doing inference."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from unittest.mock import patch
import requests
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'));config.update(rbitnet=str(args.binary.resolve()),cuda_quant_library=str(args.library.resolve()),port=18138,cwd=str(root),startup_timeout=300)
report=dict(binary_sha256=sha(args.binary),library_sha256=sha(args.library),harness_sha256=sha(Path(__file__)),effective_options={},controls=[],refusals=[])
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def identity(row):return(row['choices'][0]['message']['content'],row['choices'][0]['finish_reason'],row['usage'])
def protected(metrics):return {key:value for key,value in metrics.items()if key.startswith('rbitnet_core_gpu_')and 'available'not in key or key in ['rbitnet_inference_calls_total','rbitnet_core_speculative_draft_tokens_total','rbitnet_core_cuda_managed_allocations_total','rbitnet_core_cuda_managed_live_bytes']}
for backend in ['cpu','gpu']:
 for model in config['models']:
  if model['id']not in ['llama32-1b','qwen35-2b','gpt-oss-20b','glm47-flash']:continue
  for grammar in ['off','json','tool']:
   enabled=backend=='gpu';name=model['id'];label=backend+'-'+name+'-'+grammar
   opts=dict(RBITNET_STRUCTURED_OUTPUT=grammar,RBITNET_MAX_SEQ='2048',RBITNET_PREFIX_KV='0',RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_KV_FORMAT='f32',RBITNET_CUDA_PREFILL='1',RBITNET_CUDA_PREFILL_TF32X3='0',RBITNET_CUDA_SPLIT_KV='1',RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off',RBITNET_MOE_CACHE_POLICY='lru',RBITNET_MOE_CACHE_MB='0',RBITNET_MOE_EXECUTION='cache',RBITNET_REQUIRE_RESIDENT='1'if enabled and name=='llama32-1b'else'0')
   for key,wanted in [('QWEN',name=='qwen35-2b'),('GPT',name=='gpt-oss-20b'),('MLA',name=='glm47-flash')]:
    opts['RBITNET_CUDA_'+key+'_FULL']=opts['RBITNET_REQUIRE_'+key+'_FULL']='1'if enabled and wanted else'0'
   opts.update(RBITNET_CUDA_QWEN_PREFILL='1'if enabled and name=='qwen35-2b'else'0',RBITNET_CUDA_GPT_PREFILL='1'if enabled and name=='gpt-oss-20b'else'0',RBITNET_REQUIRE_GPT_PREFILL='1'if enabled and name=='gpt-oss-20b'else'0',RBITNET_CUDA_GPT_PREFILL_TOKENS='16')
   config['rbitnet_env']=opts;server=Server(config,model,'rbitnet',backend,out);server.log_path=out/(label+'.log');original=subprocess.Popen
   def child(*values,**options):
    if options.get('env'):
     options['env']=options['env'].copy()
     for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:options['env'].pop(key,None)
     report['effective_options'][label]={key:value for key,value in options['env'].items()if key.startswith('RBITNET_')}
    return original(*values,**options)
   body=dict(model=name,messages=[dict(role='user',content='Quelle est la capitale de la France ? Réponds uniquement par le nom de la ville.')],max_tokens=64 if name=='gpt-oss-20b'else 8,temperature=0,seed=42)
   def plain():
    response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status();return response.json()
   def stream():
    response=requests.post(server.base+'/v1/chat/completions',json=body|dict(stream=True),timeout=600);response.raise_for_status();events=[];done=0
    for line in response.text.splitlines():
     if line=='data: [DONE]':done+=1
     elif line.startswith('data: '):events.append(json.loads(line[6:]))
    reasons=[choice['finish_reason']for event in events for choice in event.get('choices',[])if choice.get('finish_reason')is not None]
    text=''.join(choice.get('delta',{}).get('content','')for event in events for choice in event.get('choices',[]));assert done==1 and len(reasons)==1,(events,done)
    return dict(text=text,finish_reason=reasons[0],events=events)
   try:
    with patch('subprocess.Popen',child):server.start()
    if grammar=='off':
     reference=plain();reference_sse=stream();assert identity(reference)[:2]==(reference_sse['text'],reference_sse['finish_reason'])and identity(reference)[0].strip()
     report['controls'].append(dict(config=label,when='before',request=body,response=reference,stream=reference_sse));save()
     requests_to_refuse=[]
     for kind in ['json_object','json_schema']:
      for streaming in [False,True]:requests_to_refuse.append(('/v1/chat/completions',body|dict(response_format={'type':kind},stream=streaming),'structured_output_not_supported'))
     for fields in [dict(tools=[dict(type='function',function=dict(name='lookup'))]),dict(tool_choice='required'),dict(functions=[dict(name='lookup')])]:
      for streaming in [False,True]:requests_to_refuse.append(('/v1/chat/completions',body|fields|dict(stream=streaming),'tool_generation_not_supported'))
     for fields in [dict(tools=[dict(name='lookup',input_schema={'type':'object'})]),dict(messages=[dict(role='user',content=[dict(type='tool_result',tool_use_id='a',content='result')])])]:
      for streaming in [False,True]:requests_to_refuse.append(('/v1/messages',body|fields|dict(stream=streaming),'tool_generation_not_supported'))
    else:
     requests_to_refuse=[(endpoint,body|dict(prompt='hello',stream=streaming),'structured_output_not_supported')for endpoint in ['/v1/chat/completions','/v1/completions','/v1/messages']for streaming in [False,True]]
    before=server.metrics()
    for endpoint,request,code in requests_to_refuse:
     response=requests.post(server.base+endpoint,json=request,timeout=600)
     assert response.status_code==501 and response.headers['content-type'].startswith('application/json'),(label,endpoint,response.status_code,response.text)
     result=response.json();assert result['error']['code']==code,(label,endpoint,result)
     report['refusals'].append(dict(config=label,endpoint=endpoint,request=request,response=result,status=response.status_code));save()
    after=server.metrics();assert protected(before)==protected(after),(label,protected(before),protected(after))
    report.setdefault('metrics',{})[label]=dict(before=before,after=after,forward_counters_unchanged=True);save()
    if grammar=='off':
     resumed=plain();resumed_sse=stream();assert identity(resumed)==identity(reference)and(resumed_sse['text'],resumed_sse['finish_reason'])==identity(reference)[:2]
     report['controls'].append(dict(config=label,when='after',request=body,response=resumed,stream=resumed_sse));save()
    print('STRUCTURED_GUARD_ACTUAL',label,'refusals',len(requests_to_refuse),'forward counters unchanged',flush=True)
   finally:server.close()
assert len(report['refusals'])==208 and len(report['controls'])==16,(len(report['refusals']),len(report['controls']))
report['complete']=True;save();print('STRUCTURED_GUARDS_LIVE_DONE models=4 backends=2 refusals=208 controls=16',flush=True)
