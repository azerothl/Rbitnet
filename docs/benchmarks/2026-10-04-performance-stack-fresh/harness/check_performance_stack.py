"""Eleventh exclusive owner: validate the combined checkout, including cross-feature guards."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve();out=base/'performance-stack-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=workspace,text=True).strip()
binding={p.relative_to(workspace).as_posix():sha(p) for p in (workspace/'crates').rglob('*.rs')}
binding.update({p.relative_to(workspace).as_posix():sha(p) for folder in ['src','include'] for p in (workspace/'native/cuda_quant'/folder).iterdir() if p.is_file()})
binding.update({p.relative_to(workspace).as_posix():sha(p) for p in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
harness_names=['finish_reason_live.py','cpu_direct_rows_live.py','context_tiers_live.py','context_tiers_proxy_live.py','llama_continuous_live.py','structured_guards_live.py','stack_guard_live.py','project_build_cache.py','diagnostic_logs.py']
assert all((base/name).is_file() for name in harness_names)
harnesses={name:sha(base/name) for name in harness_names}
journal=base/'performance-stack-experiment.json'
if journal.exists():
    old=json.loads(journal.read_text(encoding='utf-8'))
    try:
        owner=psutil.Process(old['pid']);assert owner.pid==os.getpid() or Path(__file__).name not in ' '.join(owner.cmdline())
    except psutil.NoSuchProcess:pass
status={'pid':os.getpid(),'status':'waiting','complete':False,'workspace':str(workspace),'head':head,'checker_sha256':sha(Path(__file__)),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def save():write_json(journal,status)
def unchanged():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()==head
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
    for name,digest in harnesses.items():assert sha(base/name)==digest,name
save()
try:
    predecessors=[('serial-experiments.json','run_serial_experiments.py'),('post-serial-experiments.json','run_post_serial_experiments.py'),
        ('qwen-serving-experiment.json','check_qwen_serving.py'),('structured-guards-experiment.json','check_structured_guards.py'),
        ('gpt-norm-delivery-experiment.json','check_gpt_norm_delivery.py'),('context-tiers-delivery-experiment.json','check_context_tiers_delivery.py'),
        ('llama-continuous-delivery-experiment.json','check_llama_continuous_delivery.py'),('kv-tf32-guard-experiment.json','check_kv_tf32_guard.py'),
        ('top-p-heap-experiment.json','check_top_p_heap.py'),('expert-arena-delivery-experiment.json','check_expert_arena_delivery.py')]
    deadline=time.monotonic()+43200
    while True:
        states=[json.loads((base/name).read_text(encoding='utf-8')) for name,_ in predecessors]
        assert all(s.get('status')!='failed' and not any(r['status']=='failed' for r in s.get('stages',[])) for s in states),'preceding proof failed; combined hardware work refused'
        if all(s['complete'] for s in states):
            assert all(r['status']=='passed' for s in states[:2] for r in s['stages']) and all(s['status']=='passed' for s in states[2:]);break
        for state,(_,marker) in zip(states,predecessors):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running() and marker in ' '.join(owner.cmdline()),marker
        assert time.monotonic()<deadline
        time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')}
    env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CARGO_INCREMENTAL='0',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        unchanged();status['current_step']=name;save();print('Combined performance stack:',name,flush=True)
        if command[0]=='cargo':ensure_project_build(env,workspace,out)
        with(out/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        content=diagnostic_text((out/(name+'.log')))
        commands.append({'name':name,'command':command,'cwd':str(cwd),'extra':extra or{},'returncode':result.returncode,'log_sha256':sha(out/(name+'.log'))})
        assert result.returncode==0,(name,result.returncode,content[-6000:])
        if marker:assert marker in content and 'running 0 tests' not in content,(name,marker)
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets'])
    run('clippy',['cargo','clippy','--workspace','--all-targets'])
    run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')])
    library=out/'cuda/rbitnet_cuda_quant64.dll';env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu={'RBITNET_CUDA_QUANT_LIB':str(library),'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256',
         'RBITNET_STACK_GUARD_TEST':'1','RBITNET_TEST_GGUF':str(root/'models/exported-llama/model.gguf'),
         'RBITNET_TOKENIZER':str(root/'models/exported-llama/tokenizer.json'),
         'RBITNET_QWEN_TEST_GGUF':'D:/Rbitnet-benchmark-models/qwen35-2b/Qwen3.5-2B-Q8_0.gguf',
         'RBITNET_QWEN_TEST_TOKENIZER':str(root/'target/engine-benchmark/tokenizers/Qwen3.5-2B/tokenizer.json')}
    def fixture(name,filter,extra,marker):
        run(name,['cargo','test','-p','bitnet-core','--release','--lib',filter,'--','--nocapture','--test-threads=1'],extra=extra,marker=marker)
    fixture('combined-encoded-native-refusals','optional_actual_combined_encoded_native_consumers_refused_without_mutation',gpu,'STACK_ENCODED_GUARDS_DONE cases=8')
    for split in ['0','1']:
        fixture('combined-qwen-import-split'+split,'optional_actual_combined_qwen_import_invalidates_speculative_nonce',gpu|{'RBITNET_CUDA_SPLIT_KV':split},'STACK_QWEN_IMPORT_GUARDS_DONE cases=4')
    fixture('actual-continuous','optional_actual_llama_continuous',gpu|{'RBITNET_LLAMA_CONTINUOUS_TEST':'1'},'LLAMA_CONTINUOUS_THREAD_DONE layouts=2')
    for family in ['llama','qwen']:
        for split in ['0','1']:
            fixture('portable-'+family+'-split'+split,'optional_native_portable_'+family,gpu|{'RBITNET_PORTABLE_TEST':'1','RBITNET_CUDA_SPLIT_KV':split},'PORTABLE_'+family.upper()+' actual')
    fixture('qwen-draft','optional_actual_qwen_draft_seed_penalties_stream_cancel_and_rejection_exact',gpu|{'RBITNET_QWEN_DRAFT_TEST':'1','RBITNET_CUDA_QWEN_FULL':'1','RBITNET_REQUIRE_QWEN_FULL':'1',
        'RBITNET_QWEN_DRAFT_GGUF':'D:/Rbitnet-benchmark-models/qwen35-08b/Qwen3.5-0.8B-Q8_0.gguf','RBITNET_CUDA_QWEN_PREFILL':'1','RBITNET_CUDA_PREFILL_TF32X3':'0'},'QWEN_SPEC_DECODE_DONE')
    models=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))['models']
    for label,threads,options in [('one',1,{}),('eight',8,{}),('sixteen',16,{}),('avx2',8,{'RBITNET_CPU_AVX512':'0'}),('scalar',8,{'RBITNET_CPU_SIMD_QUANT':'0'})]:
        cpu=options|{'RAYON_NUM_THREADS':str(threads),'RBITNET_CPU_DIRECT_ROWS':'1'}
        fixture('direct-bits-'+label,'cpu_direct_rows_match_original_bits_across_quant_formats',cpu,'CPU_DIRECT_ROWS_BITS_DONE formats=8 tail_shapes=6 cases=48')
        for model in models:
            fixture('direct-actual-'+model['id']+'-'+label,'optional_actual_cpu_direct_rows_match_original_gguf_matrices',
                cpu|{'RBITNET_CPU_DIRECT_ROWS_REAL_TEST':'1','RBITNET_TEST_GGUF':model.get('gguf') or model['rbitnet_model']},'CPU_DIRECT_ROWS_ACTUAL_DONE matrices=12 cases=60 exact_bits=true')
    run('release',['cargo','build','--release','-p','rbitnet-cli','-p','rbitnet-proxy','-p','rbitnet-runner'])
    binaries={}
    for name in ['rbitnet','rbitnet-proxy','rbitnet-runner']:
        target=out/(name+'.exe');shutil.copy2(base/'check-release/release'/target.name,target);binaries[name]=target
    binary=binaries['rbitnet']
    run('default-four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'default-finish')],cwd=root,marker='FINISH_REASON_NETWORK_DONE actual_models=4')
    run('combined-refusal-http',[sys.executable,'-B',str(base/'stack_guard_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'combined-refusal')],cwd=root,marker='STACK_GUARD_HTTP_DONE active_paths=2 refusals=8 state_preserved=true')
    run('cpu-direct-http',[sys.executable,'-B',str(base/'cpu_direct_rows_live.py'),'--binary',str(binary),'--output',str(out/'cpu-direct')],cwd=root,marker='CPU_DIRECT_HTTP_DONE actual_models=4 rows=72 sse=16 stops=8')
    run('context-http',[sys.executable,'-B',str(base/'context_tiers_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'context-live')],cwd=root,marker='CONTEXT_TIERS_HTTP_DONE models=2 process_restart=true ram_eviction=true corruption_replay=true token_loop_io=false')
    run('context-proxy',[sys.executable,'-B',str(base/'context_tiers_proxy_live.py'),'--proxy',str(binaries['rbitnet-proxy']),'--runner',str(binaries['rbitnet-runner']),
        '--library',str(library),'--standalone',str(out/'context-live/results.json'),'--output',str(out/'context-proxy')],cwd=root,marker='CONTEXT_TIERS_PROXY_DONE models=2 sticky_sessions=true idle_reload=true proxy_process_restart=true exact=true')
    unchanged()
    write_json(out/'manifest.json',{'head':head,'workspace':str(workspace),'source_sha256':binding,'harness_sha256':harnesses,'checker_sha256':sha(Path(__file__)),
        'binary_sha256':{name:sha(path) for name,path in binaries.items()},'library_sha256':sha(library),'commands':commands,'preceding_owners':states,
        'limits':['Combined opt-in features, defaults preserved; unsupported combinations refuse before state mutation.','No true subword grammar, learned expert predictor, KIVI or MoE continuous implementation claim.','CPU rates require a fresh comparison analysis; cross-engine parity remains pending.']})
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('PERFORMANCE_STACK_PROOF_DONE fresh combined source, cross-feature owners and real serving passed; parity pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
