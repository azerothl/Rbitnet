"""Fresh mutable-ABI guard proof, serialized after all seven preceding owners."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve();out=base/'kv-tf32-guard-proof';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
binding={path.relative_to(workspace).as_posix():sha(path)for path in(workspace/'crates').rglob('*.rs')}
binding.update({path.relative_to(workspace).as_posix():sha(path)for folder in ['src','include']for path in(workspace/'native/cuda_quant'/folder).iterdir()if path.is_file()})
binding.update({path.relative_to(workspace).as_posix():sha(path)for path in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
journal=base/'kv-tf32-guard-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text())
    try:
        owner=psutil.Process(prior['pid']);assert not(owner.pid!=os.getpid()and owner.is_running()and Path(__file__).name in ' '.join(owner.cmdline()))
    except psutil.NoSuchProcess:pass
status=dict(pid=os.getpid(),status='waiting',complete=False,workspace=str(workspace),checker_sha256=sha(Path(__file__)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save():write_json(journal,status)
def unchanged():
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
save()
try:
    predecessors=[('serial-experiments.json','run_serial_experiments.py'),('post-serial-experiments.json','run_post_serial_experiments.py'),
        ('qwen-serving-experiment.json','check_qwen_serving.py'),('structured-guards-experiment.json','check_structured_guards.py'),
        ('gpt-norm-delivery-experiment.json','check_gpt_norm_delivery.py'),('context-tiers-delivery-experiment.json','check_context_tiers_delivery.py'),
        ('llama-continuous-delivery-experiment.json','check_llama_continuous_delivery.py')]
    deadline=time.monotonic()+43200
    while True:
        states=[json.loads((base/name).read_text())for name,_ in predecessors]
        for state in states:assert state.get('status')!='failed'and not any(row['status']=='failed'for row in state.get('stages',[])),'preceding owner failed; mutable KV guard hardware work refused'
        if all(state['complete']for state in states):
            assert len(states[0]['stages'])==12 and len(states[1]['stages'])==4
            assert all(row['status']=='passed'for state in states[:2]for row in state['stages'])and all(state['status']=='passed'for state in states[2:]);break
        for state,(_,marker)in zip(states,predecessors):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running()and marker in ' '.join(owner.cmdline()),'preceding owner stopped'
        assert time.monotonic()<deadline;time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')}
    env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CARGO_INCREMENTAL='0',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        if command[0]=='cargo': ensure_project_build(env, workspace, out)
        unchanged();status['current_step']=name;save();print('Mutable KV TF32 guard:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        text=diagnostic_text((out/(name+'.log')));commands.append(dict(name=name,command=command,returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,text[-6000:])
        if marker:assert marker in text and 'running 0 tests'not in text,(name,marker)
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets']);run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')]);library=out/'cuda/rbitnet_cuda_quant64.dll'
    run('attention-f64',[sys.executable,'-B',str(base/'check_encoded_attention_oracle.py'),str(library),str(out)],cwd=root,marker='64 cases passed.')
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',
             RBITNET_ENCODED_GUARD_TEST='1',RBITNET_KV_CANONICAL_TEST='1',RBITNET_TEST_GGUF=str(root/'models/exported-llama/model.gguf'),RBITNET_TOKENIZER=str(root/'models/exported-llama/tokenizer.json'))
    for split in ['0','1']:
        run('mutable-refusal-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_encoded_tf32_mutable_configuration_refused','--','--nocapture','--test-threads=1'],
            extra=gpu|dict(RBITNET_CUDA_SPLIT_KV=split),marker='ENCODED_GUARD_DONE cases=8')
        run('canonical-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','resident::canonical_tests','--','--nocapture','--test-threads=1'],
            extra=gpu|dict(RBITNET_CUDA_SPLIT_KV=split),marker='KV_CANONICAL_DONE')
    unchanged();write_json(out/'manifest.json',dict(workspace=str(workspace),source_sha256=binding,checker_sha256=sha(Path(__file__)),library_sha256=sha(library),commands=commands,preceding_owners=states,
        limits=['Guard-only change; encoded storage and attention kernels are unchanged.','Full KV format quality/serving evidence remains the earlier fresh delivery proof; this validates mutable refusal and exact post-refusal state.','No encoded-store fusion is adopted; PR #115 remains draft until this proof is folded into its published implementation.']))
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    print('KV_TF32_GUARD_PROOF_DONE mutable ABI refusal, shared pools, F32 control, canonical state and F64 validated; publication pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
