import hashlib,json,pathlib
root=pathlib.Path('target/gpt-full');(root/'comparison').mkdir(parents=True,exist_ok=True)
m=json.loads(pathlib.Path('docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
m.pop('ollama_pid',None)
m['context']=2048
m['cuda_quant_library']=str((root/'cuda/rbitnet_cuda_quant64.dll').resolve())
m['ollama_url']='http://127.0.0.1:11439';m['port']=18111
m['rbitnet_env']={'RBITNET_CUDA_GPT_FULL':'1','RBITNET_REQUIRE_GPT_FULL':'1','RBITNET_MOE_CACHE_MB':'0','RBITNET_CUDA_SPLIT_KV':'1','RBITNET_PREFIX_KV':'0','RBITNET_CUDA_QWEN_FULL':'0','RBITNET_REQUIRE_QWEN_FULL':'0','RBITNET_CUDA_PREFILL':'0','RBITNET_MAX_SEQ':'2048'}
m.setdefault('environment',{}).update(rbitnet_commit='72b3b9c + local GPT-OSS resident graph',rbitnet_branch='codex/gpt-oss-resident',rbitnet_context_capacity=2048,cuda_execution='Complete fixed-bank GPT-OSS token pipeline, ordered RMS/router, resident KV with sinks and alternating windows; split-KV; prefix off',ollama_kv_dtype='default f16',llama_rbitnet_kv_dtype='f32')
for key,path in [('rbitnet_exe_sha256',m['rbitnet']),('cuda_quant_library_sha256',m['cuda_quant_library'])]:m['environment'][key]=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
(root/'comparison/manifest-prelaunch.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Same-revision ablation uses the existing default 8192-token resident capacity.
m['rbitnet_env']['RBITNET_MAX_SEQ']='8192';m['environment']['rbitnet_context_capacity']=8192
(root/'ablation-manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
