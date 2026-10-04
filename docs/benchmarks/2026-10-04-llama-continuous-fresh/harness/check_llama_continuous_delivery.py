"""Fresh public-checkout proof after every preceding hardware owner."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve();out=base/'llama-continuous-delivery-proof';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
private=json.loads((base/'llama-continuous-proof/manifest.json').read_text())
published=json.loads((workspace/'docs/benchmarks/2026-10-04-continuous-llama/receipt.json').read_text())
for name,digest in published['published_git_blob_sha256'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=workspace)).hexdigest()==digest,name
binding={path.relative_to(workspace).as_posix():sha(path)for path in(workspace/'crates').rglob('*.rs')}
binding.update({path.relative_to(workspace).as_posix():sha(path)for folder in ['src','include']for path in(workspace/'native/cuda_quant'/folder).iterdir()if path.is_file()})
binding.update({path.relative_to(workspace).as_posix():sha(path)for path in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
for name,digest in published['preparation']['proven_rust_source_sha256'].items():assert sha(workspace/name)==digest,name
journal=base/'llama-continuous-delivery-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text())
    try:
        owner=psutil.Process(prior['pid']);assert not(owner.pid!=os.getpid()and owner.is_running()and Path(__file__).name in ' '.join(owner.cmdline()))
    except psutil.NoSuchProcess:pass
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
harness=workspace/'docs/benchmarks/2026-10-04-continuous-llama/harness/llama_continuous_live.py';harness_sha=sha(harness)
status=dict(pid=os.getpid(),status='waiting',complete=False,workspace=str(workspace),head=head,checker_sha256=sha(Path(__file__)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save():write_json(journal,status)
def unchanged():
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
    assert sha(harness)==harness_sha
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()==head
save()
try:
    predecessors=[('serial-experiments.json','run_serial_experiments.py'),('post-serial-experiments.json','run_post_serial_experiments.py'),
        ('qwen-serving-experiment.json','check_qwen_serving.py'),('structured-guards-experiment.json','check_structured_guards.py'),
        ('gpt-norm-delivery-experiment.json','check_gpt_norm_delivery.py'),('context-tiers-delivery-experiment.json','check_context_tiers_delivery.py')]
    deadline=time.monotonic()+43200
    while True:
        states=[json.loads((base/name).read_text())for name,_ in predecessors]
        for state in states:assert state.get('status')!='failed'and not any(row['status']=='failed'for row in state.get('stages',[])),'preceding owner failed; public continuous hardware work refused'
        if all(state['complete']for state in states):
            assert len(states[0]['stages'])==12 and len(states[1]['stages'])==4
            assert all(row['status']=='passed'for state in states[:2]for row in state['stages'])and all(state['status']=='passed'for state in states[2:]);break
        for state,(_,marker)in zip(states,predecessors):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running()and marker in ' '.join(owner.cmdline()),'preceding owner stopped before success'
        assert time.monotonic()<deadline;time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CARGO_INCREMENTAL='0',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        if command[0]=='cargo': ensure_project_build(env, workspace, out)
        unchanged();status['current_step']=name;save();print('Public continuous Llama:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        text=diagnostic_text((out/(name+'.log')));commands.append(dict(name=name,command=command,returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,text[-6000:])
        if marker:assert marker in text and 'running 0 tests'not in text,(name,marker)
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets']);run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')]);library=out/'cuda/rbitnet_cuda_quant64.dll'
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_LLAMA_CONTINUOUS_TEST='1',RBITNET_TEST_GGUF=str(root/'models/exported-llama/model.gguf'),RBITNET_TOKENIZER=str(root/'models/exported-llama/tokenizer.json'))
    run('actual-driver-and-threads',['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_llama_continuous','--','--nocapture','--test-threads=1'],extra=gpu,marker='LLAMA_CONTINUOUS_THREAD_DONE layouts=2')
    text=(out/'actual-driver-and-threads.log').read_text();assert 'LLAMA_CONTINUOUS_DONE cases=24'in text and '2 passed; 0 failed'in text
    run('release',['cargo','build','--release','-p','rbitnet-cli']);binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
    run('actual-http',[sys.executable,'-B',str(harness),'--binary',str(binary),'--library',str(library),
        '--reference-binary',str(base/'finish-reason-proof/rbitnet.exe'),'--reference-library',str(base/'finish-reason-baseline-native.dll'),'--output',str(out/'live')],
        cwd=root,marker='LLAMA_CONTINUOUS_HTTP_DONE layouts=2 capacities=3 serial_cases=54 waves=21 owned_disconnects=6 explicit_stops=6')
    run('default-four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'default-finish')],cwd=root,marker='FINISH_REASON_NETWORK_DONE actual_models=4')
    unchanged();write_json(out/'manifest.json',dict(workspace=str(workspace),head=head,source_sha256=binding,harness_sha256=harness_sha,checker_sha256=sha(Path(__file__)),binary_sha256=sha(binary),library_sha256=sha(library),commands=commands,preceding_owners=states,limits=private['limits']))
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    print('LLAMA_CONTINUOUS_PUBLIC_PROOF_DONE fresh public Native/CLI, exact owners/HTTP and default four-model finish validated; publication refresh pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
