"""Refuse real structured requests while continuous/draft paths are enabled."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import requests
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
config.update(rbitnet=str(a.binary.resolve()),cuda_quant_library=str(a.library.resolve()),port=18138,cwd=str(root),startup_timeout=600)
report={'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'cases':[]}
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
def protected(m):return {k:v for k,v in m.items() if k.startswith(('rbitnet_core_gpu_','rbitnet_core_speculative_','rbitnet_core_scheduler_','rbitnet_core_cuda_managed_')) or k=='rbitnet_inference_calls_total'}
def settled_metrics(server):
    previous=server.metrics();deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        time.sleep(.05);current=server.metrics()
        if protected(previous)==protected(current):return current
        previous=current
    raise AssertionError('Native worker did not settle after completed control request')
for name in ['llama32-1b','qwen35-2b']:
    model=next(m for m in config['models'] if m['id']==name)
    opts={'RBITNET_MAX_SEQ':'2048','RBITNET_PREFIX_KV':'0','RBITNET_CONTEXT_TIERS':'0','RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256',
        'RBITNET_CUDA_KV_FORMAT':'f32','RBITNET_CUDA_PREFILL':'1','RBITNET_CUDA_PREFILL_TF32X3':'0','RBITNET_CUDA_SPLIT_KV':'0','RBITNET_STRUCTURED_OUTPUT':'off',
        'RBITNET_CUDA_CONTINUOUS':'1' if name=='llama32-1b' else '0','RBITNET_CUDA_CONTINUOUS_SLOTS':'4','RBITNET_CUDA_CONTINUOUS_TOKEN_BUDGET':'256','RBITNET_REQUIRE_RESIDENT':'1' if name=='llama32-1b' else '0',
        'RBITNET_CUDA_QWEN_FULL':'1' if name=='qwen35-2b' else '0','RBITNET_REQUIRE_QWEN_FULL':'1' if name=='qwen35-2b' else '0',
        'RBITNET_CUDA_QWEN_PREFILL':'1','RBITNET_QWEN_SPECULATIVE':'1' if name=='qwen35-2b' else '0',
        'RBITNET_QWEN_DRAFT_GGUF':'D:/Rbitnet-benchmark-models/qwen35-08b/Qwen3.5-0.8B-Q8_0.gguf','RBITNET_QWEN_SPEC_DEPTH':'4'}
    config['rbitnet_env']=opts;server=Server(config,model,'rbitnet','gpu',out);server.log_path=out/(name+'.log')
    body={'model':name,'messages':[{'role':'user','content':'Écris un récit sur un robot qui explore une bibliothèque.'}],'max_tokens':32,'temperature':0,'seed':42}
    def plain():
        response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status();return response.json()
    try:
        server.start();initial=server.metrics();before_response=plain();before=settled_metrics(server)
        report.setdefault('attempts',[]).append({'model':name,'options':opts,'initial_metrics':initial,'before_response':before_response,'metrics_before':before});save()
        used='rbitnet_core_scheduler_decode_waves_total' if name=='llama32-1b' else 'rbitnet_core_speculative_verify_blocks_total'
        assert before[used]>initial[used],(name,used,initial,before)
        refusals=[]
        for fields,code in [({'response_format':{'type':'json_object'}},'structured_output_not_supported'),
                            ({'tools':[{'type':'function','function':{'name':'lookup'}}]},'tool_generation_not_supported')]:
            for stream in [False,True]:
                request=body|fields|{'stream':stream}
                response=requests.post(server.base+'/v1/chat/completions',json=request,timeout=600)
                assert response.status_code==501 and response.headers['content-type'].startswith('application/json'),(name,response.status_code,response.text)
                result=response.json();assert result['error']['code']==code,result
                refusals.append({'request':request,'response':result,'status':response.status_code})
        after=server.metrics();assert protected(before)==protected(after),(name,protected(before),protected(after))
        after_response=plain()
        key=lambda r:(r['choices'][0]['message']['content'],r['choices'][0]['finish_reason'],r['usage'])
        assert key(before_response)==key(after_response),name
        report['cases'].append({'model':name,'options':opts,'before_response':before_response,'after_response':after_response,'refusals':refusals,
            'metrics_before':before,'metrics_after':after,'active_feature_counter':used,'active_feature_proved':True,'refusal_counters_unchanged':True});save()
    finally:server.close()
assert len(report['cases'])==2 and sum(len(r['refusals']) for r in report['cases'])==8
report['complete']=True;save();print('STACK_GUARD_HTTP_DONE active_paths=2 refusals=8 state_preserved=true',flush=True)
