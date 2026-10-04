"""Same-artifact sampler switch: actual four-model CPU/CUDA JSON and SSE equality."""
from pathlib import Path
import argparse,hashlib,json,sys,subprocess,time,statistics
from unittest.mock import patch
import requests
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output;out.mkdir(exist_ok=True);sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
config.update(rbitnet=str(a.binary.resolve()),cuda_quant_library=str(a.library.resolve()),port=18148,cwd=str(root))
report=dict(binary_sha256=sha(a.binary),library_sha256=sha(a.library),harness_sha256=sha(Path(__file__)),models=[],cases=[],timings=[],effective_options={})
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def metrics(server):
    response=requests.get(server.base+'/metrics',timeout=30);response.raise_for_status();result={}
    for line in response.text.splitlines():
        if line.startswith('#'):continue
        fields=line.split()
        if len(fields)==2:
            try:result[fields[0]]=float(fields[1])
            except ValueError:pass
    return result
def unary(server,body):
    response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status();return response.json()
def stream(server,body):
    rows=[];dones=0
    with requests.post(server.base+'/v1/chat/completions',json=body|dict(stream=True),stream=True,timeout=600)as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            if not raw.startswith(b'data: '):continue
            raw=raw[6:]
            if raw==b'[DONE]':dones+=1;continue
            row=json.loads(raw);assert 'error'not in row,row;rows.append(row)
    reasons=[r['choices'][0]['finish_reason']for r in rows if r['choices'][0]['finish_reason']is not None]
    assert dones==1 and len(reasons)==1,(dones,reasons)
    return dict(text=''.join(r['choices'][0].get('delta',{}).get('content','')for r in rows),finish_reason=reasons[0],events=rows)
def exact(row):return dict(choices=row['choices'],usage=row['usage'],model=row['model'])
for model in config['models']:
    if model['id']not in ['llama32-1b','qwen35-2b','gpt-oss-20b','glm47-flash']:continue
    report['models'].append(dict(id=model['id'],gguf_sha256=sha(Path(model['gguf'])),tokenizer_sha256=sha(Path(model['tokenizer']))));save()
    for backend in ['cpu','gpu']:
        expected=None
        for variant in ['0','1']:
            gpu=backend=='gpu'
            config['rbitnet_env']=dict(RBITNET_MAX_SEQ='2048',RBITNET_CPU_TOP_P_HEAP=variant,RBITNET_CUDA_KV_FORMAT='f32',RBITNET_CUDA_PREFILL_TF32X3='0',RBITNET_CUDA_SPLIT_KV='1',RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off',RBITNET_MOE_CACHE_MB='0',RBITNET_MOE_EXECUTION='cache',RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_PREFILL='1',
                RBITNET_CUDA_QWEN_FULL='1'if gpu and model['id']=='qwen35-2b'else'0',RBITNET_REQUIRE_QWEN_FULL='1'if gpu and model['id']=='qwen35-2b'else'0',RBITNET_CUDA_QWEN_PREFILL='1'if gpu and model['id']=='qwen35-2b'else'0',RBITNET_CUDA_GPT_FULL='1'if gpu and model['id']=='gpt-oss-20b'else'0',RBITNET_REQUIRE_GPT_FULL='1'if gpu and model['id']=='gpt-oss-20b'else'0',RBITNET_CUDA_GPT_PREFILL='1'if gpu and model['id']=='gpt-oss-20b'else'0',RBITNET_CUDA_GPT_PREFILL_TOKENS='16',RBITNET_CUDA_MLA_FULL='1'if gpu and model['id']=='glm47-flash'else'0',RBITNET_REQUIRE_MLA_FULL='1'if gpu and model['id']=='glm47-flash'else'0')
            server=Server(config,model,'rbitnet',backend,out);server.log_path=out/f"{model['id']}-{backend}-heap{variant}.log"
            original=subprocess.Popen
            def configured_child(*args,**options):
                if options.get('env'):
                    options['env']=options['env'].copy()
                    for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:options['env'].pop(key,None)
                    report['effective_options'][f"{model['id']}-{backend}-heap{variant}"]={key:value for key,value in options['env'].items()if key.startswith('RBITNET_')}
                return original(*args,**options)
            try:
                with patch('subprocess.Popen',configured_child):server.start()
                first_metrics=metrics(server)
                common=dict(model=model['id'],messages=[dict(role='system',content='Réponds brièvement et commence directement ta réponse.'),dict(role='user',content='Quelle est la capitale de la France ? Réponds uniquement par le nom de la ville.')],max_tokens=64 if model['id']=='gpt-oss-20b'else 16,seed=42)
                requests_to_run=[common|dict(temperature=.7,top_p=.9),common|dict(temperature=1.1,top_p=.99,frequency_penalty=.2,presence_penalty=.1,seed=53),common|dict(temperature=1.,top_p=0.,seed=999),common|dict(temperature=0.,top_p=.9)]
                responses=[]
                for index,body in enumerate(requests_to_run):
                    row=unary(server,body);assert row['usage']['completion_tokens']>0,row
                    report['cases'].append(dict(model=model['id'],backend=backend,variant=variant,kind='json',index=index,request=body,response=row));responses.append(exact(row));save()
                sse=stream(server,requests_to_run[0]);assert sse['text']==responses[0]['choices'][0]['message']['content']and sse['finish_reason']==responses[0]['choices'][0]['finish_reason']
                report['cases'].append(dict(model=model['id'],backend=backend,variant=variant,kind='sse',request=requests_to_run[0],response=sse));save()
                zero=unary(server,common|dict(temperature=.7,top_p=.9,max_tokens=0));assert zero['usage']['completion_tokens']==0 and zero['choices'][0]['message']['content']==''
                responses.append(exact(zero))
                report['cases'].append(dict(model=model['id'],backend=backend,variant=variant,kind='zero',response=zero));save()
                if expected is None:expected=responses
                else:assert responses==expected,(model['id'],backend,'sampler changed tokens/usage/finish')
                last_metrics=metrics(server);delta={key:last_metrics.get(key,0)-value for key,value in first_metrics.items()}
                key={'qwen35-2b':'rbitnet_core_gpu_qwen_full_tokens_total','gpt-oss-20b':'rbitnet_core_gpu_gpt_full_tokens_total','glm47-flash':'rbitnet_core_gpu_mla_full_tokens_total'}.get(model['id'],'rbitnet_native_accelerated_calls_total')
                if gpu:assert delta.get(key,0)>0,(model['id'],key,delta.get(key))
                else:assert delta.get('rbitnet_native_accelerated_calls_total',0)==0
                if gpu:
                    for prompt in ['Write a detailed story about a robot exploring a forest. Begin immediately.','Describe a future museum and its unusual exhibits. Begin immediately.']:
                        body=common|dict(messages=[dict(role='user',content=prompt)],temperature=.7,top_p=.9,max_tokens=128)
                        samples=[]
                        for cycle in range(3):
                            before=metrics(server);start=time.perf_counter();row=unary(server,body);wall_ms=(time.perf_counter()-start)*1000;after=metrics(server)
                            count=after.get('rbitnet_completion_tokens_total',0)-before.get('rbitnet_completion_tokens_total',0);elapsed=after.get('rbitnet_inference_decode_ms_sum',0)-before.get('rbitnet_inference_decode_ms_sum',0)
                            # Keep raw deltas; no TPS inferred from an instantaneous gauge.
                            assert count==row['usage']['completion_tokens'],(model['id'],count,row['usage'])
                            sample=dict(cycle=cycle,warmup=cycle==0,wall_ms=wall_ms,response=row,metrics_before=before,metrics_after=after,counter_completion_tokens=count,counter_decode_ms=elapsed,wall_tokens_per_second=row['usage']['completion_tokens']*1000/wall_ms)
                            samples.append(sample)
                        report['timings'].append(dict(model=model['id'],backend=backend,variant=variant,prompt=prompt,request=body,samples=samples));save()
                print('NUCLEUS_SAMPLER_LIVE',model['id'],backend,'heap',variant,'exact',flush=True)
            finally:server.close()
assert len(report['cases'])==96 and len(report['timings'])==16
for candidate in [row for row in report['timings']if row['variant']=='1']:
    baseline=next(row for row in report['timings']if row['variant']=='0'and row['model']==candidate['model']and row['prompt']==candidate['prompt'])
    assert [exact(row['response'])for row in candidate['samples']]==[exact(row['response'])for row in baseline['samples']],('timed outputs changed',candidate['model'])
    candidate['change_wall_tokens_per_second_percent']=100*(statistics.median(row['wall_tokens_per_second']for row in candidate['samples']if not row['warmup'])/statistics.median(row['wall_tokens_per_second']for row in baseline['samples']if not row['warmup'])-1)
report['complete']=True;report['limits']=['Single client, one warmup and two measured samples per writing prompt; descriptive timings only.','JSON/SSE equality and token usage are bounded correctness evidence, not a semantic-quality or long-context evaluation.','GPU wall rates include prefill/HTTP overhead and do not establish Ollama/llama.cpp parity.'];save()
print('NUCLEUS_SAMPLER_NETWORK_DONE actual_models=4 backends=2 exact_records=96 timed_cells=16',flush=True)
