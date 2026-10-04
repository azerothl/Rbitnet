"""Fresh selected GPT normalization proof, serialized after all earlier owners."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve();out=base/'gpt-norm-delivery-proof';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
prepared=json.loads((base/'gpt-norm-delivery-preparation.json').read_text(encoding='utf-8'));assert Path(prepared['workspace']).resolve()==workspace
for name,digest in prepared['implementation_sha256'].items():assert sha(workspace/name)==digest,name
binding={path.relative_to(workspace).as_posix():sha(path)for path in(workspace/'crates').rglob('*.rs')}
binding.update({path.relative_to(workspace).as_posix():sha(path)for folder in ['src','include']for path in(workspace/'native/cuda_quant'/folder).iterdir()if path.is_file()})
binding.update({path.relative_to(workspace).as_posix():sha(path)for path in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
journal=base/'gpt-norm-delivery-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text(encoding='utf-8'))
    try:
        owner=psutil.Process(prior['pid']);assert not(owner.pid!=os.getpid()and owner.is_running()and 'check_gpt_norm_delivery.py'in ' '.join(owner.cmdline())),'another selected norm owner is active'
    except psutil.NoSuchProcess:pass
status=dict(pid=os.getpid(),status='waiting',complete=False,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checker_sha256=sha(Path(__file__)))
def save():write_json(journal,status)
def unchanged():
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
save()
try:
    deadline=time.monotonic()+43200
    while True:
        specs=[('serial-experiments.json','run_serial_experiments.py'),('post-serial-experiments.json','run_post_serial_experiments.py'),('qwen-serving-experiment.json','check_qwen_serving.py'),('structured-guards-experiment.json','check_structured_guards.py')]
        states=[json.loads((base/name).read_text(encoding='utf-8'))for name,_ in specs]
        for state in states:assert state.get('status')!='failed'and not any(row['status']=='failed'for row in state.get('stages',[])),'preceding owner failed; selected GPT work refused'
        if all(state['complete']for state in states):
            assert len(states[0]['stages'])==12 and len(states[1]['stages'])==4
            assert all(row['status']=='passed'for state in states[:2]for row in state['stages'])and all(state['status']=='passed'for state in states[2:]);break
        for state,(_,marker)in zip(states,specs):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running()and marker in ' '.join(owner.cmdline()),'preceding owner stopped'
        assert time.monotonic()<deadline,'preceding suites exceeded wait budget'
        time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        if command[0]=='cargo': ensure_project_build(env, workspace, out)
        print('Selected GPT normalization:',name,flush=True);status['current_step']=name;save()
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        content=diagnostic_text((out/(name+'.log')));commands.append(dict(name=name,command=command,cwd=str(cwd),returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,content[-6000:])
        if marker:assert marker in content and 'running 0 tests'not in content,(name,content[-3000:])
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets']);run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')]);library=out/'cuda/rbitnet_cuda_quant64.dll'
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_QUANT_SMOKE='1',RBITNET_GPT_FULL_TEST='1',RBITNET_GPT_BLOCK_RUNTIME_TEST='1',RBITNET_GPT_ORDERED_FASTPATH_TEST='1',RBITNET_GPT_BLOCK_GGUF='D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',RBITNET_GPT_BLOCK_TOKENIZER=str(root/'target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json'),RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off')
    run('independent-norm',['cargo','test','-p','bitnet-core','--release','--lib','optional_gpt_staged_norm_preserves_original_bits_independent_f64_and_true_scratch','--','--nocapture','--test-threads=1'],extra=gpu,marker='GPT_NORM_ORACLE_DONE norms=108')
    for label,name in [('f64-fixed','full_graph_matches_f64_biases_sinks_windows_rope_gqa_reset_and_output_modes'),('f64-segmented','segmented_gpt_f64_dynamic_and_cpu_graphs_prefix_generation_and_cancel'),('block-modes','gpt_block_all_positions_modes_tails_f64_and_prefix_continuation')]:
        run(label,['cargo','test','-p','bitnet-core','--release','--lib',name,'--','--nocapture','--test-threads=1'],extra=gpu,marker='1 passed; 0 failed')
    vectors={}
    libraries={'baseline':base/'finish-reason-baseline-native.dll','norm':library}
    for variant,dll in libraries.items():
        run('actual-'+variant,['cargo','test','-p','bitnet-core','--release','--lib','actual_gpt_ordered_fastpaths_cross_library_logits_seeded_outputs_prefixes_and_cancellation','--','--nocapture','--test-threads=1'],extra=gpu|dict(RBITNET_CUDA_QUANT_LIB=str(dll),RBITNET_GPT_FASTPATH_OUTPUT_DIR=str(out/'vectors'/variant)),marker='GPT_ORDERED_FASTPATH_ACTUAL_DONE')
        vectors[variant]={path.name:sha(path)for path in(out/'vectors'/variant).iterdir()if path.is_file()};assert len(vectors[variant])==145 and vectors[variant]==vectors['baseline']
    run('release',['cargo','build','--release','-p','rbitnet-cli']);binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
    common=['--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(binary),'--model','gpt-oss-20b','--backend','gpu','--gpt-full','--device-mib','12288','--port','18138','--cycles','3','--notes','24','--max-tokens','128']
    choices={}
    for layout in ['fixed','segmented']:
        for variant,dll in libraries.items():
            options=['--gpt-prefill','--modes','serial,block16,block16-prefix']if layout=='fixed'else['--gpt-segmented','--moe-cache','8192','--modes','full,full-split,full-split-prefix']
            folder=out/('quiet-'+layout)/variant
            run('quiet-'+layout+'-'+variant,[sys.executable,'-B','scripts/benchmark_cache_stack.py',*common,'--library',str(dll),*options,'--output-dir',str(folder)],cwd=root)
            capture=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert len(capture['rows'])==27 and len(capture['sse'])==9 and len(capture['stops'])==3 and all(row['matches_baseline']for row in capture['rows'])
            choices[layout+'-'+variant]=[(row['mode'],row['cycle'],row['prompt'],row['response']['choices'],row['response']['usage'])for row in capture['rows']];assert choices[layout+'-'+variant]==choices[layout+'-baseline']
    run('network',['C:/Users/azero/anaconda3/python.exe','-B','scripts/validate_cache_streaming.py','--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(binary),'--library',str(library),'--gpt-full','--gpt-prefill','16','--split-kv','--device-mib','12288','--port','18138','--output-dir',str(out/'live')],cwd=root)
    run('four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'finish')],cwd=root)
    write_json(out/'manifest.json',dict(preparation=prepared,source_sha256=binding,commands=commands,binary_sha256=sha(binary),library_sha256={name:sha(dll)for name,dll in libraries.items()},complete_actual_vector_sha256=vectors,preceding_owners=states,limits=['Normalization only; router original.','108 independent cases, complete fixed-bank model vectors, F64 segmented fixture and actual segmented HTTP outputs; no segmented corpus logits archive yet.','Quiet two-repeat medians; no general parity claim.','Fresh exact source validation; evidence publication still pending.']))
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('GPT_NORM_DELIVERY_PROOF_DONE selected normalization fresh source, exact logits, segmented quiet/network and four-model finish passed; publication pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
