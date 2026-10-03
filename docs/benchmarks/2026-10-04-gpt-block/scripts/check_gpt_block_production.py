"""Validate the adopted GPT block/fusion sources after the quiet experiments."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'kv-codec-chain.log'
deadline=time.monotonic()+28800
while not gate.exists()or 'Actual GPU F32/F16/Q8 primitive oracles and rejection/lifetime checks passed;'not in gate.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'prior validation did not finish; refusing competing build or inference'
 time.sleep(15)
proof=root/'target/gpt-block/production-proof';proof.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16')
def run(name,command,extra=None,marker=None):
 print('Adopted GPT block/fusion validation:',name,flush=True)
 with(proof/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
 s=(proof/(name+'.log')).read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,s[-5000:])
 if marker:assert marker in s and 'running 0 tests'not in s,(name,s[-2000:])
run('native-build',['pwsh','-NoProfile','-File','scripts/build_cuda_quant.ps1','-OutDir',str(root/'target/gpt-block/cuda')])
run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
run('clippy',['cargo','clippy','--workspace','--all-targets'])
run('release',['cargo','build','--release','-p','rbitnet-cli'])
gpu={'RBITNET_CUDA_QUANT_LIB':str(root/'target/gpt-block/cuda/rbitnet_cuda_quant64.dll'),'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256',
 'RBITNET_CUDA_QUANT_SMOKE':'1','RBITNET_CUDA_GROUPED_MOE_TEST':'1','RBITNET_CUDA_PREFILL_TEST':'1'}
filters=[('ordered','native::ordered_tests'),('grouped','native::moe::grouped_tests'),('fused','native::moe::fused_tests'),('block','gpt_block_all_positions_modes_tails_f64_and_prefix_continuation')]
for name,flt in filters:
 run(name,['cargo','test','--release','-p','bitnet-core','--lib',flt,'--','--nocapture','--test-threads=1'],gpu,'1 passed; 0 failed')
actual=gpu|{'RBITNET_GPT_BLOCK_RUNTIME_TEST':'1','RBITNET_GPT_BLOCK_GGUF':'D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf','RBITNET_GPT_BLOCK_TOKENIZER':str(root/'target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json')}
run('actual-gpt',['cargo','test','--release','-p','bitnet-core','--lib','actual_gpt_block_and_fusion_logits_seeded_outputs_prefixes_and_cancellation','--','--nocapture','--test-threads=1'],actual,'Real GPT block/fusion worst KL=')
shutil.copy2(root/'target/release/rbitnet.exe',proof/'rbitnet.exe')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=json.loads((base/'gpt-block-production-paths.json').read_text(encoding='utf-8'))
m={'source_sha256':{s:sha(root/s)for s in paths},'binary_sha256':sha(proof/'rbitnet.exe'),'library_sha256':sha(Path(gpu['RBITNET_CUDA_QUANT_LIB']))}
(proof/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print('Adopted GPT block/fusion native, workspace, Clippy, release, numerical and real GGUF validation passed.',flush=True)
