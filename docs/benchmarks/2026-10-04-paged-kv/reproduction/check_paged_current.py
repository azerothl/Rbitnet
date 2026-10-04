"""Compile isolated native pages and execute actual Llama ownership/COW oracles."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';src=base/'paged-kv';out=base/'check-paged-current';native=base/'paged-native-current'
gate=base/'async-current-chain.log';deadline=time.monotonic()+14400
while 'Optional async current sources, actual GPU lifetime and two GGUF numerical suites passed;'not in gate.read_text(encoding='utf-8-sig'):
    assert time.monotonic()<deadline,'async validation not complete; refusing competing native build'
    time.sleep(15)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16')
for name in ['prepare_context_capacity.py','prepare_paged_kernels.py','prepare_paged_resident.py','prepare_paged_rust.py']:
    subprocess.run([sys.executable,str(base/name)],cwd=root,env=env,check=True)
native.mkdir(exist_ok=True)
for p in (root/'native/cuda_quant/src').iterdir():
    if p.is_file():shutil.copy2(p,native/p.name)
for p in (root/'native/cuda_quant/include').iterdir():
    if p.is_file():shutil.copy2(p,native/p.name)
for name in ['paged_pool.cuh','paged_kernels.cuh','llama_resident.cuh','rbitnet_cuda_quant.h']:shutil.copy2(src/name,native/name)
s=(root/'scripts/build_cuda_quant.ps1').read_text(encoding='utf-8-sig')
s=s.replace('$Src = Join-Path $Root "native\\cuda_quant\\src\\quant_matvec.cu"','$Src = Join-Path $PSScriptRoot "quant_matvec.cu"').replace('$Inc = Join-Path $Root "native\\cuda_quant\\include"','$Inc = $PSScriptRoot')
(native/'build_cuda.ps1').write_text(s,encoding='utf-8',newline='\n')
proof=base/'paged-current-proof';proof.mkdir(exist_ok=True)
def run(name,command,cwd,actual_env=None,marker=None):
    print('Isolated native paging:',name,flush=True);path=proof/(name+'.log')
    with path.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=cwd,env=actual_env or env,stdout=log,stderr=subprocess.STDOUT)
    text=path.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-5000:])
    if marker:assert marker in text and 'running 0 tests'not in text,(name,text[-3000:])
run('native-build',['pwsh','-NoProfile','-File',str(native/'build_cuda.ps1'),'-OutDir',str(native/'cuda')],root)
out.mkdir(exist_ok=True)
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(root/name,out/name)
shutil.copytree(root/'crates',out/'crates',dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
shutil.copytree(root/'recipes',out/'recipes',dirs_exist_ok=True)
if(root/'.cargo').exists():shutil.copytree(root/'.cargo',out/'.cargo',dirs_exist_ok=True)
for p in (base/'context-integration').rglob('*'):
    if p.is_file():q=out/p.relative_to(base/'context-integration');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
core=out/'crates/bitnet-core/src'
for name in ['runtime.rs','resident.rs']:shutil.copy2(src/name,core/'llama'/name)
dest=core/'llama/resident/paged_tests.rs';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/'paged_tests.rs',dest)
env['CARGO_TARGET_DIR']=str(base/'check-build')
run('check',['cargo','check','--workspace','--all-targets'],out)
run('clippy',['cargo','clippy','--workspace','--all-targets'],out)
env['CARGO_TARGET_DIR']=str(base/'check-release')
gpu=env|{'RBITNET_CUDA_QUANT_LIB':str(native/'cuda/rbitnet_cuda_quant64.dll'),
 'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_PAGES_TEST':'1',
 'RBITNET_TEST_GGUF':str(root/'models/exported-llama/model.gguf'),'RBITNET_TOKENIZER':str(root/'models/exported-llama/tokenizer.json')}
for split in ['0','1']:
    run('actual-pages-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','resident::paged_tests','--','--nocapture','--test-threads=1'],out,gpu|{'RBITNET_CUDA_SPLIT_KV':split,'RBITNET_CUDA_PREFILL_TF32X3':'0'},'PAGED_OWNER dense foreign/')
manifest={'native_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in native.iterdir()if p.is_file()},
 'library_sha256':hashlib.sha256(Path(gpu['RBITNET_CUDA_QUANT_LIB']).read_bytes()).hexdigest(),
 'source_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in (out/'crates').rglob('*.rs')}}
(proof/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Isolated actual CUDA Llama F32 paging/COW/owner/lifetime suites passed; serving integration and performance still pending.',flush=True)
