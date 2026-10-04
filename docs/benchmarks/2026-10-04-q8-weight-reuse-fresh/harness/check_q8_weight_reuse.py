"""Fourteenth exclusive owner: optional Qwen Q8 weight reuse, correctness first."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,statistics,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=a.workspace.resolve();out=base/'q8-weight-reuse-proof';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=workspace,text=True).strip()
binding={path.relative_to(workspace).as_posix():sha(path)for path in(workspace/'crates').rglob('*.rs')}
binding.update({path.relative_to(workspace).as_posix():sha(path)for folder in ['src','include']for path in(workspace/'native/cuda_quant'/folder).iterdir()if path.is_file()})
binding.update({path.relative_to(workspace).as_posix():sha(path)for path in [workspace/'Cargo.toml',workspace/'Cargo.lock',*(workspace/'crates').rglob('Cargo.toml')]})
harnesses={str(path.relative_to(root)):sha(path)for path in [base/'q8-reuse-harness/benchmark.py',base/'q8-reuse-harness/live.py',base/'diagnostic_logs.py',base/'project_build_cache.py',base/'finish_reason_live.py',root/'scripts/benchmark_engines.py']}
journal=base/'q8-weight-reuse-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text(encoding='utf-8'))
    try:assert psutil.Process(prior['pid']).pid==os.getpid()or Path(__file__).name not in ' '.join(psutil.Process(prior['pid']).cmdline())
    except psutil.NoSuchProcess:pass
status=dict(pid=os.getpid(),status='waiting',complete=False,workspace=str(workspace),head=head,checker_sha256=sha(Path(__file__)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save():write_json(journal,status)
def unchanged():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()==head
    for name,digest in binding.items():assert sha(workspace/name)==digest,name
    for name,digest in harnesses.items():assert sha(root/name)==digest,name
save()
try:
    deadline=time.monotonic()+43200
    while True:
        previous=json.loads((base/'context-namespace-experiment.json').read_text(encoding='utf-8'))
        assert previous['status']!='failed','preceding namespace owner failed; Q8 hardware work refused'
        if previous['complete']:
            assert previous['status']=='passed';break
        process=psutil.Process(previous['pid']);assert process.is_running()and 'check_context_namespace.py'in ' '.join(process.cmdline())
        assert time.monotonic()<deadline;time.sleep(15)
    states={name:json.loads((base/(name+'-experiment.json')).read_text(encoding='utf-8'))for name in ['performance-stack','stack-parity','context-namespace']}
    assert all(state['complete']and state['status']=='passed'for state in states.values())
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=workspace,extra=None,marker=None):
        if command[0]=='cargo':ensure_project_build(env,workspace,out)
        unchanged();status['current_step']=name;save();print('Q8 weight reuse:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        text=diagnostic_text(out/(name+'.log'));commands.append(dict(name=name,command=command,cwd=str(cwd),returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,text[-6000:])
        if marker:assert marker in text and 'running 0 tests'not in text,(name,text[-3000:])
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets']);run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    run('native-build',['pwsh','-NoProfile','-File',str(workspace/'scripts/build_cuda_quant.ps1'),'-OutDir',str(out/'cuda')]);library=out/'cuda/rbitnet_cuda_quant64.dll'
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_MAX_SEQ='2048',RBITNET_CUDA_QUANT_SMOKE='1',RBITNET_CUDA_QWEN_FULL='1',RBITNET_REQUIRE_QWEN_FULL='1',RBITNET_CUDA_QWEN_PREFILL='1',RBITNET_CUDA_PREFILL_TF32X3='0',RBITNET_PREFIX_KV='0',RBITNET_QWEN_SPECULATIVE='0',RBITNET_QWEN_SPEC_Q8_TILE='4',RBITNET_QWEN_TEST_TOKENIZER=str(root/'target/engine-benchmark/tokenizers/Qwen3.5-2B/tokenizer.json'),RBITNET_QWEN_TEST_GGUF='D:/Rbitnet-benchmark-models/qwen35-2b/Qwen3.5-2B-Q8_0.gguf',RBITNET_QWEN_DRAFT_GGUF='D:/Rbitnet-benchmark-models/qwen35-08b/Qwen3.5-0.8B-Q8_0.gguf')
    run('projection-oracle',['cargo','test','-p','bitnet-core','--release','--lib','opt_in_ordered_gemm_matches_original_bits_all_formats_tails_and_f64','--','--nocapture','--test-threads=1'],extra=gpu,marker='ORDERED_GEMM_ORACLE_DONE cases=300 q8_four_token_cases=20')
    for model,path in [('08b',gpu['RBITNET_QWEN_DRAFT_GGUF']),('2b',gpu['RBITNET_QWEN_TEST_GGUF'])]:
        for split in ['0','1']:
            run('actual-vectors-'+model+'-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_qwen_all_position_verify_and_gdn_rollback_exact','--','--nocapture','--test-threads=1'],extra=gpu|dict(RBITNET_QWEN_SPEC_TEST='1',RBITNET_QWEN_TEST_GGUF=path,RBITNET_CUDA_SPLIT_KV=split),marker='QWEN_SPEC_VERIFY graphs=1')
    for split in ['0','1']:
        for proposal in ['greedy','coupled']:
            run('actual-draft-split'+split+'-'+proposal,['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_qwen_draft_seed_penalties_stream_cancel_and_rejection_exact','--','--nocapture','--test-threads=1'],extra=gpu|dict(RBITNET_QWEN_DRAFT_TEST='1',RBITNET_CUDA_SPLIT_KV=split,RBITNET_QWEN_SPEC_DRAFT_SAMPLING=proposal),marker='QWEN_SPEC_DECODE_DONE')
    run('release',['cargo','build','--release','-p','rbitnet-cli']);binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
    common=['--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(binary),'--library',str(library),'--qwen-draft',gpu['RBITNET_QWEN_DRAFT_GGUF'],'--port','18138','--device-mib','12288']
    captures={};identities={}
    for tile in ['0','4']:
        folder=out/('quiet-tile'+tile)
        run('quiet-tile'+tile,[sys.executable,'-B',str(base/'q8-reuse-harness/benchmark.py'),*common,'--qwen-q8-tile',tile,'--model','qwen35-2b','--backend','gpu','--cycles','3','--notes','8','--max-tokens','128','--output-dir',str(folder)],cwd=root)
        data=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert data['qwen_q8_tile']==int(tile)and len(data['rows'])==36 and len(data['sse'])==12 and len(data['stops'])==4 and all(row['matches_baseline']for row in data['rows'])
        captures[tile]=data;identities[tile]=[(r['mode'],r['cycle'],r['prompt'],r['response']['choices'],r['response']['usage'])for r in data['rows']];assert identities[tile]==identities['0']
    analysis=[]
    for mode in ['target','draft-1','draft-4','draft-8']:
        for prompt in [0,1]:
            row=dict(mode=mode,prompt=prompt,measurements={})
            for tile,data in captures.items():
                samples=[r for r in data['rows']if r['mode']==mode and r['prompt']==prompt and r['cycle']>0];assert len(samples)==2
                rates=[]
                for r in samples:
                    m=r['metrics_delta'];assert m['rbitnet_completion_tokens_total']==r['response']['usage']['completion_tokens']
                    rates.append(m['rbitnet_completion_tokens_total']*1000/m['rbitnet_inference_decode_ms_sum'])
                row['measurements'][tile]=dict(median_tps=statistics.median(rates),samples_tps=rates)
            row['reuse_gain_percent']=100*(row['measurements']['4']['median_tps']/row['measurements']['0']['median_tps']-1);analysis.append(row)
    write_json(out/'analysis.json',dict(rows=analysis,limits=['Two measured repeats, no confidence interval.','Two writing prompts; one-token capital reply excluded from throughput comparison.','Optional experiment, no default promotion or reference-engine parity claim.']))
    run('network-tile4',[sys.executable,'-B',str(base/'q8-reuse-harness/live.py'),*common,'--qwen-q8-tile','4','--output-dir',str(out/'live')],cwd=root)
    run('default-four-model-finish',[sys.executable,'-B',str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'default-finish')],cwd=root,marker='FINISH_REASON_NETWORK_DONE actual_models=4')
    unchanged();write_json(out/'manifest.json',dict(workspace=str(workspace),head=head,source_sha256=binding,harness_sha256=harnesses,checker_sha256=sha(Path(__file__)),commands=commands,binary_sha256=sha(binary),library_sha256=sha(library),models_sha256={name:sha(Path(gpu[name]))for name in ['RBITNET_QWEN_TEST_GGUF','RBITNET_QWEN_DRAFT_GGUF','RBITNET_QWEN_TEST_TOKENIZER']},preceding_owners=states,limits=['Q8 reuse only in speculative verification; default tile zero.','Successful tests do not themselves establish a net speculative speedup.','Only recorded dense .8B/2B pair; no MoE or classical q/p sampler.','Combined cache-stack source is a separate checkout and is not changed by this study.']))
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('Q8_WEIGHT_REUSE_PROOF_DONE exact projection/model/draft/HTTP tests; quiet performance decision remains to be reviewed',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
