"""Fourth serial owner: compile guards and exercise real CPU/CUDA API refusals."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve()
out=base/'structured-guards-proof';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
journal=base/'structured-guards-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text(encoding='utf-8'))
    try:
        owner=psutil.Process(prior['pid'])
        assert not(owner.pid!=os.getpid()and owner.is_running()and 'check_structured_guards.py'in ' '.join(owner.cmdline())),'another structured owner is active'
    except psutil.NoSuchProcess:pass
rust={path.relative_to(workspace).as_posix():sha(path)for path in(workspace/'crates').rglob('*.rs')}
native={path.relative_to(workspace).as_posix():sha(path)for folder in ['src','include']for path in(workspace/'native/cuda_quant'/folder).iterdir()if path.is_file()}
cargo={path.relative_to(workspace).as_posix():sha(path)for path in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]}
harness=base/'structured_guards_live.py';harness_digest=sha(harness)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
status=dict(pid=os.getpid(),status='waiting',complete=False,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checker_sha256=sha(Path(__file__)),workspace=str(workspace),base_head=head,harness_sha256=harness_digest)
def save():write_json(journal,status)
def unchanged():
    for name,digest in(rust|native|cargo).items():assert sha(workspace/name)==digest,name
    assert sha(harness)==harness_digest,'live harness changed'
save()
try:
    deadline=time.monotonic()+43200
    while True:
        states=[json.loads((base/name).read_text(encoding='utf-8'))for name in ['serial-experiments.json','post-serial-experiments.json','qwen-serving-experiment.json']]
        for state in states:
            assert state.get('status')!='failed'and not any(row['status']=='failed'for row in state.get('stages',[])),'a preceding owner failed; structured hardware work refused'
        if all(state['complete']for state in states):
            assert len(states[0]['stages'])==12 and len(states[1]['stages'])==4
            assert all(row['status']=='passed'for state in states[:2]for row in state['stages'])and states[2]['status']=='passed'
            break
        for state,marker in zip(states,['run_serial_experiments.py','run_post_serial_experiments.py','check_qwen_serving.py']):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running()and marker in ' '.join(owner.cmdline()),'preceding owner stopped before success'
        assert time.monotonic()<deadline,'preceding serial suites exceeded wait budget'
        time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,marker=None):
        if command[0]=='cargo': ensure_project_build(env, workspace, out)
        status['current_step']=name;save();print('Structured output guards:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        content=diagnostic_text((out/(name+'.log')))
        commands.append(dict(name=name,command=command,cwd=str(cwd),returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,content[-6000:])
        if marker:assert marker in content and 'running 0 tests'not in content,(name,content[-3500:])
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets'])
    run('clippy',['cargo','clippy','--workspace','--all-targets'])
    run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')])
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    run('release',['cargo','build','--release','-p','rbitnet-cli'])
    binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary);library=out/'cuda/rbitnet_cuda_quant64.dll'
    run('actual-api',[sys.executable,'-B',str(harness),'--binary',str(binary),'--library',str(library),'--output',str(out/'live')],cwd=root,marker='STRUCTURED_GUARDS_LIVE_DONE models=4 backends=2 refusals=208 controls=16')
    live=json.loads((out/'live/results.json').read_text(encoding='utf-8'));assert live['complete']and len(live['refusals'])==208 and len(live['controls'])==16
    manifest=dict(workspace=str(workspace),base_head=head,checker_sha256=sha(Path(__file__)),harness_sha256=harness_digest,binary_sha256=sha(binary),library_sha256=sha(library),rust_source_sha256=rust,native_source_sha256=native,cargo_source_sha256=cargo,commands=commands,preceding_owners=states,limits=['Fail-closed capability guards only: no tokenizer-aware grammar implementation.','Four actual models CPU/CUDA, 208 JSON refusals before SSE and unchanged forward counters; 16 before/after control records each containing JSON and SSE.','No inference throughput claim from this capability repair; issue #24 remains open.'])
    write_json(out/'manifest.json',manifest)
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    print('STRUCTURED_GUARDS_PROOF_DONE actual four-model CPU/CUDA fail-closed requests and resumed ordinary generation passed; grammar remains unimplemented',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
