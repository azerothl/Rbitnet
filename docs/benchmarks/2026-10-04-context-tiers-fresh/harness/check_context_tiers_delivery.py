"""Sixth hardware owner: actual persistent prompt state and real two-model proxy."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve()
out=base/'context-tiers-delivery-proof';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
binding={path.relative_to(workspace).as_posix():sha(path)for path in(workspace/'crates').rglob('*.rs')}
binding.update({path.relative_to(workspace).as_posix():sha(path)for folder in ['src','include']for path in(workspace/'native/cuda_quant'/folder).iterdir()if path.is_file()})
binding.update({path.relative_to(workspace).as_posix():sha(path)for path in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml'),workspace/'docs/CONTEXT_TIERS.md']})
harnesses={name:sha(base/name)for name in ['context_tiers_live.py','context_tiers_proxy_live.py']}
journal=base/'context-tiers-delivery-experiment.json'
if journal.exists():
    previous=json.loads(journal.read_text())
    try:
        owner=psutil.Process(previous['pid']);assert not(owner.pid!=os.getpid()and owner.is_running()and Path(__file__).name in ' '.join(owner.cmdline())),'another context owner is active'
    except psutil.NoSuchProcess:pass
status=dict(pid=os.getpid(),status='waiting',complete=False,workspace=str(workspace),checker_sha256=sha(Path(__file__)),
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save():write_json(journal,status)
def unchanged():
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
    for name,digest in harnesses.items():assert sha(base/name)==digest,name
save()
try:
    deadline=time.monotonic()+43200
    predecessors=[('serial-experiments.json','run_serial_experiments.py'),('post-serial-experiments.json','run_post_serial_experiments.py'),
                  ('qwen-serving-experiment.json','check_qwen_serving.py'),('structured-guards-experiment.json','check_structured_guards.py'),
                  ('gpt-norm-delivery-experiment.json','check_gpt_norm_delivery.py')]
    while True:
        states=[json.loads((base/name).read_text(encoding='utf-8'))for name,_ in predecessors]
        for state in states:assert state.get('status')!='failed'and not any(row['status']=='failed'for row in state.get('stages',[])),'preceding owner failed; context hardware work refused'
        if all(state['complete']for state in states):
            assert len(states[0]['stages'])==12 and len(states[1]['stages'])==4
            assert all(row['status']=='passed'for state in states[:2]for row in state['stages'])and all(state['status']=='passed'for state in states[2:]);break
        for state,(_,marker)in zip(states,predecessors):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running()and marker in ' '.join(owner.cmdline()),'required preceding owner stopped'
        assert time.monotonic()<deadline,'preceding suites exceeded wait budget'
        time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={name:value for name,value in os.environ.items()if not name.startswith('RBITNET_')}
    env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CARGO_INCREMENTAL='0',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        if command[0]=='cargo': ensure_project_build(env, workspace, out)
        unchanged();status['current_step']=name;save();print('Context tiers delivery:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        text=diagnostic_text((out/(name+'.log')))
        commands.append(dict(name=name,command=command,cwd=str(cwd),returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,text[-6000:])
        if marker:assert marker in text and 'running 0 tests'not in text,(name,marker,text[-3000:])
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets'])
    run('clippy',['cargo','clippy','--workspace','--all-targets'])
    run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')])
    library=out/'cuda/rbitnet_cuda_quant64.dll';env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',
             RBITNET_PORTABLE_TEST='1',RBITNET_CUDA_QUANT_SMOKE='1',
             RBITNET_TEST_GGUF=str(root/'models/exported-llama/model.gguf'),RBITNET_TOKENIZER=str(root/'models/exported-llama/tokenizer.json'),
             RBITNET_QWEN_TEST_GGUF='D:/Rbitnet-benchmark-models/qwen35-2b/Qwen3.5-2B-Q8_0.gguf',
             RBITNET_QWEN_TEST_TOKENIZER=str(root/'target/engine-benchmark/tokenizers/Qwen3.5-2B/tokenizer.json'))
    for family in ['llama','qwen']:
        for split in ['0','1']:
            run('actual-transport-'+family+'-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','optional_native_portable_'+family,'--','--nocapture','--test-threads=1'],
                extra=gpu|dict(RBITNET_CUDA_SPLIT_KV=split),marker='PORTABLE_'+family.upper()+' actual')
    for name,fixture in [('qwen-attention-f64','opt_in_full_attention_matches_f64_for_eight_formats_gating_rope_gqa_and_restore'),
                         ('qwen-recurrent-f64','opt_in_dense_recurrent_layer_matches_f64_and_restarts_sequence')]:
        run(name,['cargo','test','-p','bitnet-core','--release','--lib',fixture,'--','--nocapture','--test-threads=1'],extra=gpu,marker='1 passed; 0 failed')
    run('release',['cargo','build','--release','-p','rbitnet-cli','-p','rbitnet-proxy','-p','rbitnet-runner'])
    binaries={}
    for name in ['rbitnet','rbitnet-proxy','rbitnet-runner']:
        target=out/(name+'.exe');shutil.copy2(base/'check-release/release'/target.name,target);binaries[name]=target
    run('actual-serving',[sys.executable,'-B',str(base/'context_tiers_live.py'),'--binary',str(binaries['rbitnet']),'--library',str(library),'--output',str(out/'live')],
        cwd=root,marker='CONTEXT_TIERS_HTTP_DONE models=2 process_restart=true ram_eviction=true corruption_replay=true token_loop_io=false')
    run('actual-proxy',[sys.executable,'-B',str(base/'context_tiers_proxy_live.py'),'--proxy',str(binaries['rbitnet-proxy']),'--runner',str(binaries['rbitnet-runner']),
        '--library',str(library),'--standalone',str(out/'live/results.json'),'--output',str(out/'proxy')],cwd=root,
        marker='CONTEXT_TIERS_PROXY_DONE models=2 sticky_sessions=true idle_reload=true proxy_process_restart=true exact=true')
    run('default-four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binaries['rbitnet']),'--library',str(library),'--output',str(out/'default-finish')],cwd=root,marker='FINISH_REASON_NETWORK_DONE actual_models=4')
    unchanged()
    manifest=dict(workspace=str(workspace),source_sha256=binding,harness_sha256=harnesses,checker_sha256=sha(Path(__file__)),commands=commands,
                  binary_sha256={name:sha(path)for name,path in binaries.items()},library_sha256=sha(library),preceding_owners=states,
                  limits=['F32 Llama/Qwen CUDA only; no encoded KV, CPU state, GPT or GLM checkpoint transport.',
                          'RAM and disk quotas are per compatibility namespace/model runtime; old namespaces are not globally reclaimed.',
                          'No physical disk-full, crash timing sweep, streamed model weights or reference-engine performance comparison.',
                          'Metrics probes are correctness checks; no throughput claim from these serving samples.'])
    write_json(out/'manifest.json',manifest)
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    print('CONTEXT_TIERS_DELIVERY_PROOF_DONE actual model outputs, RAM/disk/restart and two-model runners validated; performance/publication pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
