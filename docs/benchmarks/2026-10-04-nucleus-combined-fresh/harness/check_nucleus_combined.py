"""Fifteenth exclusive owner: adaptive exact nucleus selection, no default promotion."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import psutil
from atomic_journal import write_json
from diagnostic_logs import diagnostic_text
from project_build_cache import ensure_project_build
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';w=a.workspace.resolve();out=base/'nucleus-combined-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=w)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
binding={p.relative_to(w).as_posix():sha(p)for p in(w/'crates').rglob('*.rs')}
binding.update({p.relative_to(w).as_posix():sha(p)for folder in ['src','include']for p in(w/'native/cuda_quant'/folder).iterdir()if p.is_file()})
binding.update({p.relative_to(w).as_posix():sha(p)for p in [w/'Cargo.toml',w/'Cargo.lock',*(w/'crates').rglob('Cargo.toml')]})
harnesses={str(p.relative_to(root)):sha(p)for p in [base/'nucleus_sampling_live.py',base/'project_build_cache.py',base/'diagnostic_logs.py',root/'scripts/benchmark_engines.py']}
inputs=json.loads((base/'top-p-actual-inputs.json').read_text());input_sha=sha(base/'top-p-actual-inputs.json')
journal=base/'nucleus-combined-experiment.json'
if journal.exists():
    previous=json.loads(journal.read_text())
    try:
        process=psutil.Process(previous['pid']);assert not(process.pid!=os.getpid()and process.is_running()and Path(__file__).name in ' '.join(process.cmdline()))
    except psutil.NoSuchProcess:pass
status=dict(pid=os.getpid(),status='waiting',complete=False,workspace=str(w),head=head,checker_sha256=sha(Path(__file__)),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save():write_json(journal,status)
def unchanged():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()==head
    for name,digest in binding.items():assert sha(w/name)==digest,name
    for name,digest in harnesses.items():assert sha(root/name)==digest,name
    assert sha(base/'top-p-actual-inputs.json')==input_sha
    assert sha(Path(inputs['origin_manifest']))==inputs['origin_manifest_sha256']
    for row in inputs['vectors']:assert sha(Path(row['path']))==row['sha256'],row['path']
save()
try:
    deadline=time.monotonic()+43200
    while True:
        previous=json.loads((base/'q8-weight-reuse-experiment.json').read_text())
        assert previous['status']!='failed','preceding Q8 owner failed; adaptive nucleus execution refused'
        if previous['complete']:assert previous['status']=='passed';break
        process=psutil.Process(previous['pid']);assert process.is_running()and 'check_q8_weight_reuse.py'in ' '.join(process.cmdline())
        assert time.monotonic()<deadline;time.sleep(15)
    unchanged();status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_TARGET_DIR=str(base/'check-build'))
    commands=[]
    def run(name,command,cwd=w,extra=None,marker=None):
        if command[0]=='cargo':ensure_project_build(env,w,out)
        unchanged();status['current_step']=name;save();print('Adaptive nucleus:',name,flush=True)
        with(out/(name+'.log')).open('w',encoding='utf-8')as log:result=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        text=diagnostic_text(out/(name+'.log'));commands.append(dict(name=name,command=command,cwd=str(cwd),returncode=result.returncode,log_sha256=sha(out/(name+'.log'))))
        assert result.returncode==0,(name,result.returncode,text[-6000:])
        if marker:assert marker in text and 'running 0 tests'not in text,(name,marker)
        unchanged()
    run('check',['cargo','check','--workspace','--all-targets']);run('clippy',['cargo','clippy','--workspace','--all-targets']);run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
    env['CARGO_TARGET_DIR']=str(base/'check-release')
    run('synthetic-exact',['cargo','test','-p','bitnet-core','--release','--lib','heap_top_p_preserves_stable_ties_f32_thresholds_fallback_and_rng','--','--nocapture','--test-threads=1'],marker='TOP_P_HEAP_ORIGINAL_EXACT_DONE cases=2028')
    run('actual-cost',['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_top_p_original_equality_and_quiet_cpu_cost','--','--nocapture','--test-threads=1'],extra=dict(RBITNET_TOP_P_ACTUAL_TEST='1',RBITNET_TOP_P_INPUTS=str(base/'top-p-actual-inputs.json')),marker='TOP_P_ACTUAL_DONE vectors=24 exact_cases=864')
    rows=[json.loads(line.split('TOP_P_COST ',1)[1])for line in diagnostic_text(out/'actual-cost.log').splitlines()if 'TOP_P_COST 'in line];assert len(rows)==20
    for row in rows:row['cost_change_percent']=100*(row['heap_median_ns']/row['original_median_ns']-1)
    write_json(out/'summary.json',dict(rows=rows,limits=['Pure selection cost, weights built before timing; actual frozen GPT-OSS vectors only.','Engine throughput and default promotion are separate decisions.']))
    proof=json.loads((base/'performance-stack-proof/manifest.json').read_text(encoding='utf-8'))
    native_paths=sorted(name for name in binding if name.startswith('native/'))
    assert native_paths==sorted(name for name in proof['source_sha256'] if name.startswith('native/'))
    for name in native_paths:
        historical=subprocess.check_output(['git','show',proof['head']+':'+name],cwd=w)
        assert (w/name).read_bytes().replace(b'\r\n',b'\n')==historical.replace(b'\r\n',b'\n'),name
    library=base/'performance-stack-proof/cuda/rbitnet_cuda_quant64.dll'
    assert sha(library)==proof['library_sha256']
    write_json(out/'native-reuse.json',dict(compiled_head=proof['head'],library_sha256=sha(library),identical_native_paths=native_paths,new_native_build=False))
    run('release',['cargo','build','--release','-p','rbitnet-cli']);binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
    run('actual-network',[sys.executable,'-B',str(base/'nucleus_sampling_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'live')],cwd=root,marker='NUCLEUS_SAMPLER_NETWORK_DONE actual_models=4 backends=2 exact_records=96 timed_cells=16')
    unchanged();write_json(out/'manifest.json',dict(workspace=str(w),head=head,source_sha256=binding,harness_sha256=harnesses,inputs=inputs,commands=commands,binary_sha256=sha(binary),library_sha256=sha(library),checker_sha256=sha(Path(__file__)),limits=['Opt-in default disabled; outputs/RNG equality and measured cost do not guarantee useful end-to-end speed.','Timing has one warmup and two measured samples; no statistical-significance or engine-parity claim.']))
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('ADAPTIVE_NUCLEUS_PROOF_DONE exact vectors/RNG, isolated selection cost and actual four-model CPU/GPU serving captured; adoption decision pending',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
