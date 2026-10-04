"""Fresh arena integration validation after the nine earlier hardware owners."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve();out=base/'expert-arena-delivery-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prepared=json.loads((base/'expert-arena-delivery-preparation.json').read_text())
assert Path(prepared['workspace']).resolve()==workspace
for name,digest in prepared['implementation_sha256'].items():assert sha(workspace/name)==digest,name
binding={p.relative_to(workspace).as_posix():sha(p)for p in(workspace/'crates').rglob('*.rs')}
binding.update({p.relative_to(workspace).as_posix():sha(p)for folder in ['src','include']for p in(workspace/'native/cuda_quant'/folder).iterdir()if p.is_file()})
binding.update({p.relative_to(workspace).as_posix():sha(p)for p in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
harness=base/'expert_arena_finish_live.py';harness_sha=sha(harness);assert harness_sha==prepared['finish_harness_sha256']
journal=base/'expert-arena-delivery-experiment.json'
if journal.exists():
    old=json.loads(journal.read_text())
    try:
        owner=psutil.Process(old['pid']);assert not(owner.is_running()and Path(__file__).name in ' '.join(owner.cmdline()))
    except psutil.NoSuchProcess:pass
status=dict(pid=os.getpid(),status='waiting',complete=False,workspace=str(workspace),checker_sha256=sha(Path(__file__)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save():write_json(journal,status)
def unchanged():
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
    assert sha(harness)==harness_sha
save()
try:
    predecessors=[('serial-experiments.json','run_serial_experiments.py'),('post-serial-experiments.json','run_post_serial_experiments.py'),
        ('qwen-serving-experiment.json','check_qwen_serving.py'),('structured-guards-experiment.json','check_structured_guards.py'),
        ('gpt-norm-delivery-experiment.json','check_gpt_norm_delivery.py'),('context-tiers-delivery-experiment.json','check_context_tiers_delivery.py'),
        ('llama-continuous-delivery-experiment.json','check_llama_continuous_delivery.py'),('kv-tf32-guard-experiment.json','check_kv_tf32_guard.py'),('top-p-heap-experiment.json','check_top_p_heap.py')]
    deadline=time.monotonic()+43200
    while True:
        states=[json.loads((base/name).read_text())for name,_ in predecessors]
        for state in states:assert state.get('status')!='failed'and not any(row['status']=='failed'for row in state.get('stages',[])),'preceding hardware owner failed; arena integration refused'
        if all(state['complete']for state in states):
            assert all(row['status']=='passed'for state in states[:2]for row in state['stages'])and all(state['status']=='passed'for state in states[2:]);break
        for state,(_,marker)in zip(states,predecessors):
            if not state['complete']:
                owner=psutil.Process(state['pid']);assert owner.is_running()and marker in ' '.join(owner.cmdline()),'preceding owner stopped'
        assert time.monotonic()<deadline;time.sleep(15)
    unchanged();private=json.loads((base/'expert-arena-proof/manifest.json').read_text())
    for name,digest in prepared['implementation_sha256'].items():assert private['prepared']['source_sha256'][name]==digest,name
    status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        unchanged();status['current_step']=name;save();print('Public expert arena:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        content=(out/(name+'.log')).read_text();commands.append(dict(name=name,command=command,returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,content[-6000:])
        if marker:assert marker in content and 'running 0 tests'not in content,(name,marker)
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets']);run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')]);library=out/'cuda/rbitnet_cuda_quant64.dll'
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_QUANT_SMOKE='1',RBITNET_EXPERT_ARENA_TEST='1',RBITNET_MOE_ARENA='1',RBITNET_MOE_EXECUTION='cache')
    run('f64-routed',['cargo','test','-p','bitnet-core','--release','--lib','opt_in_routed_graph_matches_f64_experts_biases_and_selection_changes','--','--nocapture','--test-threads=1'],extra=gpu,marker='1 passed; 0 failed')
    run('actual-owner',['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_expert_arena_views','--','--nocapture','--test-threads=1'],extra=gpu,marker='EXPERT_ARENA_OWNER_DONE physical_allocations=1 groups=4 views=12')
    models=[('gpt-oss-20b','gptoss','gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf','gpt-oss-20b'),('glm47-flash','mla','GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf','GLM-4.7-Flash')]
    for name,family,relative,tokenizer in models:
        gguf=str(Path('D:/Rbitnet-benchmark-models')/relative)
        run('actual-cache-'+name,['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_expert_arena_cache','--','--nocapture','--test-threads=1'],extra=gpu|dict(RBITNET_TEST_GGUF=gguf),marker='physical_allocations=1 refill_allocations=0 poison_and_last_lease=true')
        for budget in ['512','8192']:
            actual=gpu|dict(RBITNET_MOE_POLICY_TEST='1',RBITNET_MOE_POLICY_FAMILY=family,RBITNET_MOE_POLICY_GGUF=gguf,RBITNET_MOE_POLICY_CACHE=budget,RBITNET_MOE_POLICY_TOKENIZER=str(root/'target/engine-benchmark/tokenizers'/tokenizer/'tokenizer.json'),RBITNET_MOE_REQUIRE_PREFETCH='1'if budget=='512'else'0')
            run('actual-runtime-'+name+'-'+budget,['cargo','test','-p','bitnet-core','--release','--lib','real_moe_async_cache_preserves_logits_generations_prefix_and_model_lifetime','--','--nocapture','--test-threads=1'],extra=actual,marker='MoE async real worst KL=')
    run('release',['cargo','build','--release','-p','rbitnet-cli']);binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
    run('arena-finish-network',[sys.executable,'-B',str(harness),'--binary',str(binary),'--library',str(library),'--output',str(out/'finish-arena')],cwd=root,marker='ARENA_FINISH_NETWORK_DONE actual_models=2')
    run('default-four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'finish-default')],cwd=root,marker='FINISH_REASON_NETWORK_DONE actual_models=4')
    write_json(out/'manifest.json',dict(preparation=prepared,source_sha256=binding,commands=commands,binary_sha256=sha(binary),library_sha256=sha(library),private_proof_manifest_sha256=sha(base/'expert-arena-proof/manifest.json'),harness_sha256=harness_sha,preceding_owners=states,
        limits=['F32 finish-reason base plus identical seven arena source paths; no changed Native kernel content.','Private quiet timing remains the same-revision opt-in ablation; fresh source validation is numerical/ownership/serving, not a new timing comparison.','Global sampled GPU peaks are not process VRAM or a complete physical-memory budget guarantee.','Fixed maximum-format slots; no dynamic compaction or trained predictor.','Arena default remains disabled; publication and capacity/performance decision are pending.']))
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('EXPERT_ARENA_DELIVERY_PROOF_DONE fresh public source, Native F64, owners, full models and actual serving passed; publication pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
