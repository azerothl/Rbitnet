"""Actual loopback requests distinguish observing EOS from exhausting a budget."""
from pathlib import Path
import argparse,hashlib,json,sys,subprocess
from unittest.mock import patch
import requests
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from benchmark_engines import Server
parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=args.output;out.mkdir(exist_ok=True)
config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
config.update(rbitnet=str(args.binary.resolve()),cuda_quant_library=str(args.library.resolve()),port=18138,cwd=str(root))
report=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),effective_options={},cases=[])
def save():(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for model in config['models']:
    if model['id']not in ['llama32-1b','qwen35-2b','gpt-oss-20b','glm47-flash']:continue
    config['rbitnet_env']=dict(RBITNET_MAX_SEQ='2048',RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_KV_FORMAT='f32',
        RBITNET_CUDA_PREFILL='1',RBITNET_CUDA_PREFILL_TF32X3='0',RBITNET_CUDA_SPLIT_KV='1',RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off',RBITNET_MOE_CACHE_POLICY='lru',RBITNET_MOE_CACHE_MB='0',RBITNET_MOE_EXECUTION='cache',
        RBITNET_CUDA_QWEN_FULL='1'if model['id']=='qwen35-2b'else'0',RBITNET_REQUIRE_QWEN_FULL='1'if model['id']=='qwen35-2b'else'0',
        RBITNET_CUDA_QWEN_PREFILL='1'if model['id']=='qwen35-2b'else'0',RBITNET_CUDA_GPT_FULL='1'if model['id']=='gpt-oss-20b'else'0',
        RBITNET_REQUIRE_GPT_FULL='1'if model['id']=='gpt-oss-20b'else'0',RBITNET_CUDA_GPT_PREFILL='1'if model['id']=='gpt-oss-20b'else'0',
        RBITNET_REQUIRE_GPT_PREFILL='1'if model['id']=='gpt-oss-20b'else'0',RBITNET_CUDA_GPT_PREFILL_TOKENS='16',
        RBITNET_CUDA_MLA_FULL='1'if model['id']=='glm47-flash'else'0',RBITNET_REQUIRE_MLA_FULL='1'if model['id']=='glm47-flash'else'0')
    server=Server(config,model,'rbitnet','gpu',out);server.log_path=out/(model['id']+'.log')
    original=subprocess.Popen
    def configured_child(*values,**options):
        if options.get('env'):
            options['env']=options['env'].copy()
            for key in ['RBITNET_CHAT_TEMPLATE','RBITNET_CHAT_FORMAT']:options['env'].pop(key,None)
            report['effective_options'][model['id']]={key:value for key,value in options['env'].items() if key.startswith('RBITNET_')}
        return original(*values,**options)
    def unary(body):
        response=requests.post(server.base+'/v1/chat/completions',json=body,timeout=600);response.raise_for_status();return response.json()
    def streaming(body):
        with requests.post(server.base+'/v1/chat/completions',json=body|dict(stream=True),stream=True,timeout=600)as response:
            response.raise_for_status();rows=[];dones=0
            for raw in response.iter_lines():
                if not raw.startswith(b'data: '):continue
                raw=raw[6:]
                if raw==b'[DONE]':dones+=1;continue
                row=json.loads(raw);assert 'error'not in row,row;rows.append(row)
        assert dones==1,dones
        reasons=[r['choices'][0]['finish_reason']for r in rows if r['choices'][0]['finish_reason']is not None]
        assert len(reasons)==1,reasons
        text=''.join(r['choices'][0].get('delta',{}).get('content','')for r in rows)
        return dict(text=text,finish_reason=reasons[0],events=rows)
    try:
        with patch('subprocess.Popen',configured_child):server.start()
        full=None
        for question in ['Quelle est la capitale de la France ? Réponds uniquement par le nom de la ville.','Réponds uniquement oui : est-ce que 2 + 2 = 4 ?']:
            body=dict(model=model['id'],messages=[dict(role='system',content='Tu es un assistant précis. Réponds brièvement et commence directement ta réponse.'),dict(role='user',content=question)],max_tokens=128,temperature=0,seed=42)
            candidate=unary(body);report['cases'].append(dict(model=model['id'],kind='eos-search',request=body,response=candidate));save()
            if candidate['choices'][0]['finish_reason']=='stop'and candidate['choices'][0]['message']['content'].strip():full=candidate;break
        assert full is not None,('no actual nonempty EOS completion observed',model['id'])
        count=full['usage']['completion_tokens'];assert 0<count<128,count
        for limit in sorted(set([0,1,count,count+1,128])):
            request=body|dict(max_tokens=limit);response=unary(request);stream=streaming(request)
            expected='stop'if limit>count else'length'
            report['cases'].append(dict(model=model['id'],kind='eos-boundary',request=request,response=response,stream=stream,expected_finish_reason=expected));save()
            assert response['choices'][0]['finish_reason']==stream['finish_reason']==expected,(model['id'],limit,response,stream)
            assert response['usage']['completion_tokens']==min(count,limit),(model['id'],limit,response)
            assert response['choices'][0]['message']['content']==stream['text'],(model['id'],limit,response,stream)
        visible=full['choices'][0]['message']['content'];stop=visible[:min(2,len(visible))]
        request=body|dict(stop=stop);response=unary(request);stream=streaming(request)
        report['cases'].append(dict(model=model['id'],kind='client-stop',request=request,response=response,stream=stream));save()
        assert response['choices'][0]['finish_reason']==stream['finish_reason']=='stop'
        assert response['choices'][0]['message']['content']==stream['text']==''
        print('FINISH_REASON_ACTUAL',model['id'],'eos_completion_tokens',count,flush=True)
    finally:
        server.close()
report['complete']=True;save();print('FINISH_REASON_NETWORK_DONE actual_models=4',flush=True)
