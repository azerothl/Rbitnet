"""Serialized production checks for mapped weights and measured MoE policies."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path.cwd();base=root/'target/moe-placement';base.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',RAYON_NUM_THREADS='16',CARGO_INCREMENTAL='0')
library=root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(library)=='b8fb68b35761e2fc13a373846c21e9ffd3e5e62eb83aa060ecedc86cbf73ed43'
def run(name,command,extra=None):
    print(name,'starting',flush=True)
    options=env.copy();options.update(extra or {})
    with(base/f'{name}.log').open('w',encoding='utf-8')as log:
        result=subprocess.run(command,cwd=root,env=options,stdout=log,stderr=subprocess.STDOUT)
    print(name,'exit',result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
    if name in ('native','mapped','ownership')or name.startswith('real-'):
        assert 'running 0 tests'not in(base/f'{name}.log').read_text(encoding='utf-8'),f'{name} did not exercise tests'
run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
run('clippy',['cargo','clippy','--workspace','--all-targets'])
run('cli',['cargo','build','-p','rbitnet-cli','--release'])
gpu={'RBITNET_CUDA_QUANT_LIB':str(library),'RBITNET_CUDA_QUANT_SMOKE':'1',
    'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256'}
for name,filter in [('native','native::'),('mapped','backend::mapped_quant_tests'),('ownership','backend::transfer_tests')]:
    run(name,['cargo','test','-p','bitnet-core','--release','--lib',filter,'--','--nocapture','--test-threads=1'],gpu)
models={
 'gptoss':('D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf','target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json'),
 'mla':('D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf','target/engine-benchmark/tokenizers/GLM-4.7-Flash/tokenizer.json')}
for family,cache,cap in [('gptoss',0,12288),('gptoss',8192,12288),('gptoss',16,6144),('mla',8192,12288),('mla',0,6144)]:
    gguf,tok=models[family]
    case=gpu.copy();case.update(RBITNET_CUDA_QUANT_SMOKE='0',RBITNET_MOE_POLICY_TEST='1',
        RBITNET_MOE_POLICY_GGUF=str(Path(gguf).resolve()),RBITNET_MOE_POLICY_TOKENIZER=str((root/tok).resolve()),
        RBITNET_MOE_POLICY_FAMILY=family,RBITNET_MOE_POLICY_CACHE=str(cache),RBITNET_CUDA_DEVICE_BUDGET_MB=str(cap),
        RBITNET_HYBRID_MAX_VRAM_MB=str(cap))
    run(f'real-{family}-cache{cache}-cap{cap}',['cargo','test','-p','bitnet-core','--release','--lib',
        'native::graph::moe_policy_runtime_tests::real_moe_cpu_and_adaptive','--','--nocapture','--test-threads=1'],case)
source=[p for p in (root/'crates/bitnet-core/src').rglob('*.rs')]
source += [root/'crates/bitnet-server/src/metrics.rs']
(base/'validation-manifest.json').write_text(json.dumps({'dll_sha256':sha(library),
    'cli_sha256':sha(root/'target/release/rbitnet.exe'),'source_sha256':{p.relative_to(root).as_posix():sha(p)for p in source},
    'cases':[{'family':f,'cache_mib':c,'cap_mib':b}for f,c,b in [('gptoss',0,12288),('gptoss',8192,12288),('gptoss',16,6144),('mla',8192,12288),('mla',0,6144)]]},indent=2)+'\n',encoding='utf-8',newline='\n')
print('Mapped/cost production checks passed; live and same-build performance remain.',flush=True)
