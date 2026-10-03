"""Adopt pages only after context publication; serialize all GPU work."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';dest=root/'target/paged-kv';proof=dest/'production-proof'
deadline=time.monotonic()+28800
for path,marker in [(base/'followups-chain.log','Passed serial follow-up: async-benchmark-gpt-oss-20b-cache8192'),(root/'target/context-tokenizer/published.json','"published": true')]:
 while not path.exists()or marker not in path.read_text(encoding='utf-8-sig'):
  assert time.monotonic()<deadline,'publication/quiet window did not complete; no adoption or inference started'
  time.sleep(15)
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='codex/context-tokenizer'
assert not subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).strip(),'tracked work must be delivered before switching lots'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip()
subprocess.run(['git','checkout','-b','codex/paged-kv'],cwd=root,check=True)
src=base/'paged-kv';paths=[]
for name in ['paged_pool.cuh','paged_kernels.cuh','llama_resident.cuh']:
 p=root/'native/cuda_quant/src'/name;shutil.copy2(src/name,p);paths.append(p.relative_to(root).as_posix())
for name in ['paged_pool.cuh','paged_kernels.cuh']:
 p=root/'native/cuda_quant/src'/name;s=p.read_text(encoding='utf-8')
 s=s.replace('// Draft native physical F32 KV pages. No GPU/quality validation yet.','// Native physical F32 KV pages, shared by matching context owners.')
 s=s.replace('// Draft: F32 page indirection; original arithmetic/reduction order.','// F32 page indirection; preserve dense arithmetic and reduction order.')
 p.write_text(s,encoding='utf-8',newline='\n')
header=root/'native/cuda_quant/include/rbitnet_cuda_quant.h';s=header.read_text(encoding='utf-8');p=(src/'rbitnet_cuda_quant.h').read_text(encoding='utf-8')
start=p.index('/* Optional F32 Llama pages.');end=p.index('#ifdef __cplusplus',start);addition=p[start:end]
assert 'rbitnet_cuda_llama_create_paged'not in s;at=s.rindex('#ifdef __cplusplus');s=s[:at]+addition+s[at:];header.write_text(s,encoding='utf-8',newline='\n');paths.append(header.relative_to(root).as_posix())
resident=root/'crates/bitnet-core/src/llama/resident.rs';shutil.copy2(src/'resident.rs',resident);paths.append(resident.relative_to(root).as_posix())
tests=root/'crates/bitnet-core/src/llama/resident/paged_tests.rs';tests.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/'paged_tests.rs',tests);paths.append(tests.relative_to(root).as_posix())
runtime=root/'crates/bitnet-core/src/llama/runtime.rs';s=runtime.read_text(encoding='utf-8')
old='        let resident = if backend_kind == BackendKind::Cuda && kv.as_paged().is_none() {'
assert s.count(old)==1
s=s.replace(old,'''        let native_pages = super::resident::configured_page_limit()?;
        if native_pages.is_some() && (backend_kind != BackendKind::Cuda || kv.as_paged().is_some()) {
            return Err(BitNetError::Inference("native CUDA KV pages require CUDA dense Llama residency and no host paged KV".into()));
        }
'''+old)
old='        let backend = make_backend(backend_kind);';assert s.count(old)==1
s=s.replace(old,'''        if native_pages.is_some() && resident.is_none() {
            return Err(BitNetError::Inference("native CUDA KV paging unavailable or pool/context allocation refused".into()));
        }
'''+old)
runtime.write_text(s,encoding='utf-8',newline='\n');paths.append(runtime.relative_to(root).as_posix())
for p in [resident,tests,runtime]:subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true',str(p)],cwd=root,check=True)
dest.mkdir(parents=True,exist_ok=True);proof.mkdir(parents=True,exist_ok=True)
(dest/'paths.json').write_text(json.dumps(paths,indent=2)+'\n',encoding='utf-8')
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16')
def run(name,command,extra=None,marker=None):
 print('Adopted F32 paging:',name,flush=True);path=proof/(name+'.log')
 with path.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
 s=path.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,s[-6000:])
 if marker:assert marker in s and 'running 0 tests'not in s,(name,s[-3000:])
run('native-build',['pwsh','-NoProfile','-File',str(root/'scripts/build_cuda_quant.ps1'),'-OutDir',str(dest/'cuda')])
run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
run('clippy',['cargo','clippy','--workspace','--all-targets'])
gpu={'RBITNET_CUDA_QUANT_LIB':str(dest/'cuda/rbitnet_cuda_quant64.dll'),'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_PAGES_TEST':'1',
 'RBITNET_TEST_GGUF':str(root/'models/exported-llama/model.gguf'),'RBITNET_TOKENIZER':str(root/'models/exported-llama/tokenizer.json'),'RBITNET_CUDA_PREFILL_TF32X3':'0'}
for split in ['0','1']:
 run('actual-pages-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','resident::paged_tests','--','--nocapture','--test-threads=1'],gpu|{'RBITNET_CUDA_SPLIT_KV':split},'PAGED_OWNER dense foreign/')
run('release',['cargo','build','--release','-p','rbitnet-cli'])
shutil.copy2(root/'target/release/rbitnet.exe',proof/'rbitnet.exe');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m={'source_sha256':{p:sha(root/p)for p in paths},'binary_sha256':sha(proof/'rbitnet.exe'),'library_sha256':sha(Path(gpu['RBITNET_CUDA_QUANT_LIB'])),
 'native_source_sha256':{p.relative_to(root).as_posix():sha(p)for folder in ['native/cuda_quant/src','native/cuda_quant/include']for p in(root/folder).iterdir()if p.is_file()}}
(proof/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
run('quiet-ablation',[sys.executable,str(base/'followup-harness/paged_benchmark.py'),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--backend','gpu','--cycles','3','--notes','24','--max-tokens','128','--device-mib','12288','--port','18138','--binary',str(proof/'rbitnet.exe'),'--library',gpu['RBITNET_CUDA_QUANT_LIB'],'--model','llama32-1b','--paged','--output-dir',str(dest/'ablation')])
r=json.loads((dest/'ablation/results.json').read_text(encoding='utf-8'));assert len(r['rows'])==36 and all(x['matches_baseline']for x in r['rows'])and len(r['sse'])==12
run('network',[sys.executable,str(base/'followup-harness/paged_live.py'),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(proof/'rbitnet.exe'),'--library',gpu['RBITNET_CUDA_QUANT_LIB'],'--paged','--split-kv','--output-dir',str(dest/'live'),'--port','18138','--device-mib','12288'])
print('Adopted native Llama F32 pages, actual GPU ownership, quiet ablations and network suites passed.',flush=True)
