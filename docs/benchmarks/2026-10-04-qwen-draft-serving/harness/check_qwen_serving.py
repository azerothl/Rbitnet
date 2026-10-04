"""One waiting successor for real Qwen draft serving, after both existing owners."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve()
out=base/'qwen-serving-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
status=dict(pid=os.getpid(),status='waiting',complete=False,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checker_sha256=sha(Path(__file__)))
previous=base/'qwen-serving-experiment.json'
if previous.exists():
    prior=json.loads(previous.read_text(encoding='utf-8'))
    try:
        process=psutil.Process(prior['pid'])
        assert not(process.is_running()and 'check_qwen_serving.py'in ' '.join(process.cmdline())),'another serving owner is active'
    except psutil.NoSuchProcess:pass
def save():write_json(base/'qwen-serving-experiment.json',status)
save()
try:
    deadline=time.monotonic()+43200
    while True:
        primary=json.loads((base/'serial-experiments.json').read_text(encoding='utf-8'))
        post=json.loads((base/'post-serial-experiments.json').read_text(encoding='utf-8'))
        assert not any(row['status']=='failed'for row in primary['stages']),'primary failed; serving hardware work refused'
        assert post['status']!='failed','post-serial failed; serving hardware work refused'
        if primary['complete']and post['complete']:
            assert len(primary['stages'])==12 and len(post['stages'])==4
            assert all(row['status']=='passed'for row in primary['stages']+post['stages']);break
        for state,marker in [(primary,'run_serial_experiments.py'),(post,'run_post_serial_experiments.py')]:
            if not state['complete']:
                process=psutil.Process(state['pid']);assert process.is_running()and marker in ' '.join(process.cmdline()),'required owner stopped before completing'
        assert time.monotonic()<deadline,'previous serialized suites still unfinished'
        time.sleep(15)
    status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    receipt=base/'qwen-serving-preparation.json'
    if not receipt.exists():subprocess.run([sys.executable,'-B',str(base/'prepare_qwen_serving.py'),'--workspace',str(workspace)],cwd=root,check=True)
    prepared=json.loads(receipt.read_text(encoding='utf-8'));assert Path(prepared['workspace']).resolve()==workspace
    for name,digest in prepared['modified_source_sha256'].items():assert sha(workspace/name)==digest,name
    rust_binding={p.relative_to(workspace).as_posix():sha(p)for p in(workspace/'crates').rglob('*.rs')}
    native_binding={p.relative_to(workspace).as_posix():sha(p)for folder in ['src','include']for p in(workspace/'native/cuda_quant'/folder).iterdir()if p.is_file()}
    cargo_binding={p.relative_to(workspace).as_posix():sha(p)for p in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]}
    env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
    def run(name,command,cwd=workspace,extra=None,marker=None):
        if command[0]=='cargo': ensure_project_build(env, workspace, out)
        print('Qwen draft serving:',name,flush=True)
        status['current_step']=name;save()
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=cwd,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
        text=(out/(name+'.log')).read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-6000:])
        if marker:assert marker in text and 'running 0 tests'not in text,(name,text[-3000:])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')])
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets'])
    run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    env['CARGO_TARGET_DIR']=str(base/'check-release');library=out/'cuda/rbitnet_cuda_quant64.dll'
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_MAX_SEQ='2048',RBITNET_CUDA_QWEN_FULL='1',RBITNET_REQUIRE_QWEN_FULL='1',RBITNET_QWEN_DRAFT_TEST='1',RBITNET_CUDA_QWEN_PREFILL='1',RBITNET_CUDA_PREFILL_TF32X3='0',RBITNET_PREFIX_KV='0',RBITNET_QWEN_SPECULATIVE='0',RBITNET_QWEN_TEST_TOKENIZER=str(root/'target/engine-benchmark/tokenizers/Qwen3.5-2B/tokenizer.json'),RBITNET_QWEN_TEST_GGUF='D:/Rbitnet-benchmark-models/qwen35-2b/Qwen3.5-2B-Q8_0.gguf',RBITNET_QWEN_DRAFT_GGUF='D:/Rbitnet-benchmark-models/qwen35-08b/Qwen3.5-0.8B-Q8_0.gguf')
    for split in ['0','1']:
        for proposals in ['greedy','coupled']:
            run('actual-draft-split'+split+'-'+proposals,['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_qwen_draft_seed_penalties_stream_cancel_and_rejection_exact','--','--nocapture','--test-threads=1'],extra=gpu|dict(RBITNET_CUDA_SPLIT_KV=split,RBITNET_QWEN_SPEC_DRAFT_SAMPLING=proposals),marker='QWEN_SPEC_DECODE_DONE')
    run('release',['cargo','build','--release','-p','rbitnet-cli']);binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
    common=['--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(binary),'--library',str(library),'--port','18138','--device-mib','12288']
    run('default-four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'default-finish')],cwd=root)
    run('network',['C:/Users/azero/anaconda3/python.exe','-B',str(base/'qwen-serving-harness/live.py'),*common,'--qwen-draft',gpu['RBITNET_QWEN_DRAFT_GGUF'],'--output-dir',str(out/'live')],cwd=root)
    capture=json.loads((out/'live/results.json').read_text(encoding='utf-8'));assert len(capture['cases'])==20 and capture['target_only_references']
    run('quiet',['C:/Users/azero/anaconda3/python.exe','-B',str(base/'qwen-serving-harness/benchmark.py'),*common,'--qwen-draft',gpu['RBITNET_QWEN_DRAFT_GGUF'],'--model','qwen35-2b','--backend','gpu','--cycles','3','--notes','8','--max-tokens','128','--output-dir',str(out/'ablation')],cwd=root)
    capture=json.loads((out/'ablation/results.json').read_text(encoding='utf-8'));assert len(capture['rows'])==36 and len(capture['sse'])==12 and len(capture['stops'])==4 and all(row['matches_baseline']for row in capture['rows'])
    for name,digest in (rust_binding|native_binding|cargo_binding).items():assert sha(workspace/name)==digest,name
    manifest=dict(preparation=prepared,binary_sha256=sha(binary),library_sha256=sha(library),rust_source_sha256=rust_binding,native_source_sha256=native_binding,cargo_source_sha256=cargo_binding,model_sha256={name:sha(Path(gpu[name]))for name in ['RBITNET_QWEN_TEST_GGUF','RBITNET_QWEN_DRAFT_GGUF','RBITNET_QWEN_TEST_TOKENIZER']},limits=['Speculative Qwen remains optional, two dense full-GPU checkpoints only.','Target-only greedy/sampled IDs must remain identical; performance and acceptance decision still needs publication.','Prefix coexistence and structured subword grammar are refused rather than inferred supported.'])
    write_json(out/'manifest.json',manifest)
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    print('QWEN_SERVING_PROOF_DONE optional real draft HTTP/SSE and exact target sampler validated; performance decision and publication pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
