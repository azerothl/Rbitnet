"""Fresh serial build and actual GPU byte validation; diagnostic only, no speed claim."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
import psutil
from project_build_cache import ensure_project_build

root=Path.cwd(); base=root/'target/performance-cache'
w=Path('C:/Users/azero/.codex/worktrees/exact-nucleus-sampling/Rbitnet')
out=base/'moe-device-payload-proof'
assert not out.exists(), 'Preserve observations'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=w)
assert not any(p.name().lower() in ('rbitnet.exe','bitnet-core-async.exe') for p in psutil.process_iter()), 'Another GPU owner exists'
out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
binding={p.relative_to(w).as_posix():sha(p) for p in (w/'crates').rglob('*.rs')}
binding.update({p.relative_to(w).as_posix():sha(p) for folder in ['src','include'] for p in (w/'native/cuda_quant'/folder).iterdir() if p.is_file()})
binding.update({p.relative_to(w).as_posix():sha(p) for p in [w/'Cargo.toml',w/'Cargo.lock',*(w/'crates').rglob('Cargo.toml')]})
library=base/'performance-stack-proof/cuda/rbitnet_cuda_quant64.dll'
native_proof=json.loads((base/'performance-stack-proof/manifest.json').read_text(encoding='utf-8'))
assert sha(library)==native_proof['library_sha256']
native_paths=sorted(n for n in binding if n.startswith('native/'))
assert native_paths==sorted(n for n in native_proof['source_sha256'] if n.startswith('native/'))
for name in native_paths:
    previous=subprocess.check_output(['git','show',native_proof['head']+':'+name],cwd=w)
    assert previous.replace(b'\r\n',b'\n')==(w/name).read_bytes().replace(b'\r\n',b'\n'),name
manifest=dict(head=head,workspace=str(w),source_sha256=binding,checker_sha256=sha(Path(__file__)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running',commands=[],library_sha256=sha(library),native_reuse=dict(compiled_head=native_proof['head'],paths=native_paths,new_build=False),limits=['Test-only synchronous GPU readback; production inference unchanged.','Passing isolated byte validation cannot prove a fix of the historical quiet HTTP divergence.','No throughput benchmark or default-policy change.'])
def save():
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
def unchanged():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()==head
    for name,digest in binding.items(): assert sha(w/name)==digest,name
    assert sha(library)==manifest['library_sha256']
env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_TARGET_DIR=str(base/'check-build'))
save()
def run(name,command,extra=None,marker=None):
    if command[0]=='cargo': ensure_project_build(env,w,out)
    unchanged(); print('Device expert validation:',name,flush=True)
    path=out/(name+'.log')
    with path.open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=w,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,timeout=900)
    output=path.read_text(encoding='utf-8',errors='replace')
    manifest['commands'].append(dict(name=name,command=command,extra_environment=extra or {},returncode=result.returncode,log_sha256=sha(path)))
    save()
    assert result.returncode==0,(name,output[-7000:])
    if marker: assert marker in output and 'running 0 tests' not in output,(name,output[-7000:])
    unchanged()
try:
    run('check',['cargo','check','--workspace','--all-targets'])
    run('clippy',['cargo','clippy','--workspace','--all-targets'])
    run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    fixture='optional_expert_cache_all_policies_preserve_device_bytes_across_refills_and_passes'
    command=['cargo','test','-p','bitnet-core','--release','--lib',fixture,'--','--nocapture','--test-threads=1']
    config=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
    manifest['models']=[]
    for model_id in ['gpt-oss-20b','glm47-flash']:
        model=next(m for m in config['models'] if m['id']==model_id)
        gguf=Path(model['gguf']); stat=gguf.stat()
        manifest['models'].append(dict(id=model_id,path=str(gguf),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,source=model.get('source'),full_file_hash=False))
        run(model_id,command,dict(RBITNET_MOE_CACHE_TEST='1',RBITNET_TEST_GGUF=str(gguf),RBITNET_CUDA_QUANT_LIB=str(library)),marker='EXPERT_DEVICE_BYTES_DONE groups=432 projections=1296')
        assert gguf.stat().st_size==stat.st_size and gguf.stat().st_mtime_ns==stat.st_mtime_ns
    candidates=list((base/'check-release/release/deps').glob('bitnet_core-*.exe'))
    assert len(candidates)==1,candidates
    frozen=out/'bitnet-core-device-bytes.exe';shutil.copy2(candidates[0],frozen)
    manifest.update(test_binary_sha256=sha(frozen),status='passed',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save();print('MOE_DEVICE_PAYLOAD_DONE models=2 groups=864 projections=2592',flush=True)
except BaseException as error:
    manifest.update(status='failed',error=str(error));save();raise
