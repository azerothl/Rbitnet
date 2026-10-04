"""Thirteenth serial owner: prove original junction bug, then validate guarded cache."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=a.workspace.resolve();out=base/'context-namespace-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=workspace,text=True).strip()
binding={p.relative_to(workspace).as_posix():sha(p) for p in(workspace/'crates').rglob('*.rs')}
binding.update({p.relative_to(workspace).as_posix():sha(p) for p in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
binding.update({p.relative_to(workspace).as_posix():sha(p) for folder in ['src','include'] for p in (workspace/'native/cuda_quant'/folder).iterdir() if p.is_file()})
harnesses={name:sha(base/name) for name in ['context_tiers_live.py','context_tiers_proxy_live.py','project_build_cache.py','diagnostic_logs.py']}
journal=base/'context-namespace-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text(encoding='utf-8'))
    try:
        owner=psutil.Process(prior['pid']);assert owner.pid==os.getpid() or Path(__file__).name not in ' '.join(owner.cmdline())
    except psutil.NoSuchProcess:pass
status={'pid':os.getpid(),'status':'waiting','complete':False,'workspace':str(workspace),'head':head,'checker_sha256':sha(Path(__file__)),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def save():write_json(journal,status)
def unchanged():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()==head
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
    for name,digest in harnesses.items():assert sha(base/name)==digest,name
save()
try:
    deadline=time.monotonic()+43200
    while True:
        state=json.loads((base/'stack-parity-experiment.json').read_text(encoding='utf-8'))
        assert state['status']!='failed','preceding parity owner failed; namespace hardware work refused'
        if state['complete']:
            assert state['status']=='passed';break
        owner=psutil.Process(state['pid']);assert owner.is_running() and 'check_stack_parity.py' in ' '.join(owner.cmdline())
        assert time.monotonic()<deadline;time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    baseline=base/'check-context-namespace-baseline';assert not baseline.exists(),'preserve existing attempt before preparing baseline'
    baseline.mkdir()
    for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(workspace/name,baseline/name)
    for name in ['crates','.cargo','recipes','tests']:
        if(workspace/name).exists():shutil.copytree(workspace/name,baseline/name,ignore=shutil.ignore_patterns('target','__pycache__'))
    name='crates/bitnet-core/src/context_tiers.rs';boundary='#[cfg(test)]\nmod tests {'
    old=subprocess.check_output(['git','show','HEAD^:'+name],cwd=workspace).decode('utf-8')
    new=(workspace/name).read_text(encoding='utf-8');assert old.count(boundary)==new.count(boundary)==1
    (baseline/name).write_text(old.split(boundary,1)[0]+boundary+new.split(boundary,1)[1],encoding='utf-8',newline='\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(label,command,cwd=workspace,marker=None,expected_failure=False):
        unchanged();status['current_step']=label;save();print('Context namespace guards:',label,flush=True)
        if command[0]=='cargo':ensure_project_build(env,cwd,out)
        with(out/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        content=diagnostic_text((out/(label+'.log')))
        commands.append({'label':label,'command':command,'cwd':str(cwd),'returncode':result.returncode,'expected_failure':expected_failure,'log_sha256':sha(out/(label+'.log'))})
        if expected_failure:
            assert result.returncode==101 and 'running 1 test' in content and 'redirected managed format accepted' in content and 'FAILED' in content,(label,content[-6000:])
        else:assert result.returncode==0,(label,content[-6000:])
        if marker:assert marker in content and 'running 0 tests' not in content,(label,marker)
    run('original-junction-negative-control',['cargo','test','-p','bitnet-core','--lib','redirected_format_directory_is_refused_before_namespace_creation','--','--nocapture','--test-threads=1'],cwd=baseline,expected_failure=True)
    run('check',['cargo','check','--workspace','--all-targets'])
    run('clippy',['cargo','clippy','--workspace','--all-targets'])
    run('guarded-context-unit-tests',['cargo','test','-p','bitnet-core','--lib','context_tiers::tests','--','--nocapture','--test-threads=1'],marker='11 passed; 0 failed')
    run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    proof=json.loads((base/'performance-stack-proof/manifest.json').read_text(encoding='utf-8'))
    native=workspace/'native/cuda_quant';source=Path(proof['workspace'])/'native/cuda_quant'
    for folder in ['src','include']:
        for item in (native/folder).iterdir():
            if item.is_file():
                origin=source/folder/item.name
                assert sha(origin)==proof['source_sha256'][origin.relative_to(Path(proof['workspace'])).as_posix()],origin
                assert item.read_bytes().replace(b'\r\n',b'\n')==origin.read_bytes().replace(b'\r\n',b'\n'),item
    library=base/'performance-stack-proof/cuda/rbitnet_cuda_quant64.dll';assert sha(library)==proof['library_sha256']
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    run('release',['cargo','build','--release','-p','rbitnet-cli','-p','rbitnet-proxy','-p','rbitnet-runner'])
    binaries={}
    for name in ['rbitnet','rbitnet-proxy','rbitnet-runner']:
        target=out/(name+'.exe');shutil.copy2(base/'check-release/release'/target.name,target);binaries[name]=target
    run('actual-context-http',[sys.executable,'-B',str(base/'context_tiers_live.py'),'--binary',str(binaries['rbitnet']),'--library',str(library),'--output',str(out/'live')],cwd=root,marker='CONTEXT_TIERS_HTTP_DONE models=2 process_restart=true ram_eviction=true corruption_replay=true token_loop_io=false')
    run('actual-context-proxy',[sys.executable,'-B',str(base/'context_tiers_proxy_live.py'),'--proxy',str(binaries['rbitnet-proxy']),'--runner',str(binaries['rbitnet-runner']),'--library',str(library),'--standalone',str(out/'live/results.json'),'--output',str(out/'proxy')],cwd=root,marker='CONTEXT_TIERS_PROXY_DONE models=2 sticky_sessions=true idle_reload=true proxy_process_restart=true exact=true')
    unchanged();write_json(out/'manifest.json',{'head':head,'source_sha256':binding,'checker_sha256':sha(Path(__file__)),'harness_sha256':harnesses,'commands':commands,'baseline_source_sha256':sha(baseline/'crates/bitnet-core/src/context_tiers.rs'),
        'binary_sha256':{name:sha(path) for name,path in binaries.items()},'reused_library_sha256':sha(library),'limits':['Configured parent directory remains user-selected.','Windows junction/rename protections observed; non-Windows coverage requires a separate machine.','No global old-namespace quota or SSD streamed model weights.']})
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('CONTEXT_NAMESPACE_PROOF_DONE original failure reproduced, Windows directory refusals and actual context/proxy serving passed',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
