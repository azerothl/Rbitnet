"""Change one Native MLA normalization call; bind reused Rust, measure serial HTTP."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import psutil

root=Path.cwd();base=root/'target/performance-cache'
w=Path('C:/Users/azero/.codex/worktrees/gpt-ordered-normalization/Rbitnet')
out=base/'glm-ordered-norm-proof'
assert not out.exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=w)
assert not any(p.name().lower() in ('rbitnet.exe','bitnet-core-device-bytes.exe','bitnet-core-async.exe') for p in psutil.process_iter())
out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
previous=json.loads((base/'nucleus-combined-proof/manifest.json').read_text(encoding='utf-8'))
native_previous=json.loads((base/'performance-stack-proof/manifest.json').read_text(encoding='utf-8'))
test_previous=json.loads((base/'moe-device-payload-proof/manifest.json').read_text(encoding='utf-8'))
binary=base/'nucleus-combined-proof/rbitnet.exe'
frozen=base/'moe-device-payload-proof/bitnet-core-device-bytes.exe'
baseline_library=base/'performance-stack-proof/cuda/rbitnet_cuda_quant64.dll'
assert sha(binary)==previous['binary_sha256'] and sha(frozen)==test_previous['test_binary_sha256'] and sha(baseline_library)==native_previous['library_sha256']
changed='native/cuda_quant/src/mla_full.cuh'
binding={name:sha(w/name) for name in previous['source_sha256']}
for name in binding:
    before=subprocess.check_output(['git','show',previous['head']+':'+name],cwd=w)
    after=(w/name).read_bytes()
    if name==changed: assert after.replace(b'\r\n',b'\n')!=before.replace(b'\r\n',b'\n')
    else: assert after.replace(b'\r\n',b'\n')==before.replace(b'\r\n',b'\n'),name
build_script=w/'scripts/build_cuda_quant.ps1'
harness=base/'glm-policy-current-quiet-proof/quiet_harness.py'
manifest=dict(head=head,workspace=str(w),source_sha256=binding,checker_sha256=sha(Path(__file__)),build_script_sha256=sha(build_script),harness_sha256=sha(harness),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running',commands=[],binary_reuse=dict(head=previous['head'],sha256=sha(binary),all_rust_paths_equal=True,new_build=False),test_binary_reuse=dict(head=test_previous['head'],sha256=sha(frozen),new_build=False),baseline_library_sha256=sha(baseline_library),changed_native_paths=[changed],limits=['Only Native MLA normalization changes; no fresh Rust CLI or test build claimed.','One cold cycle and three warm measured cycles per prompt; baseline then candidate, no randomization or significance claim.','Normal HTTP generation at temperature zero, LRU 8192 MiB, async/prefetch disabled, prefix disabled.','A successful quiet repetition does not explain the historical Least-Stale divergence.'])
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
def unchanged():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()==head
    for name,digest in binding.items(): assert sha(w/name)==digest,name
    assert sha(binary)==manifest['binary_reuse']['sha256'] and sha(frozen)==manifest['test_binary_reuse']['sha256']
env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16')
save()
def run(name,command,cwd=root,extra=None,marker=None):
    unchanged();print('GLM ordered normalization:',name,flush=True)
    path=out/(name+'.log')
    with path.open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=cwd,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    output=path.read_text(encoding='utf-8',errors='replace')
    manifest['commands'].append(dict(name=name,command=command,cwd=str(cwd),extra_environment=extra or {},returncode=result.returncode,log_sha256=sha(path)))
    save()
    assert result.returncode==0,(name,output[-5000:])
    if marker: assert marker in output and 'running 0 tests' not in output,(name,output[-5000:])
    unchanged()
try:
    run('native-build',['pwsh','-NoProfile','-File',str(build_script),'-OutDir',str(out/'cuda')],cwd=w)
    library=out/'cuda/rbitnet_cuda_quant64.dll';manifest['library_sha256']=sha(library);save()
    gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_QUANT_SMOKE='1',RBITNET_GPT_ORDERED_FASTPATH_TEST='1')
    run('norm-exact-oracle',[str(frozen),'optional_gpt_staged_norm_preserves_original_bits_independent_f64_and_true_scratch','--nocapture','--test-threads=1'],extra=gpu,marker='GPT_NORM_ORACLE_DONE norms=108')
    run('mla-f64',[str(frozen),'quantized_full_mla_matches_f64_fixed_dynamic_cpu_fallback_graphs_reset_and_prefix','--nocapture','--test-threads=1'],extra=gpu,marker='1 passed; 0 failed')
    sanitizer=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/compute-sanitizer.bat')
    run('mla-synccheck',[str(sanitizer),'--tool','synccheck','--error-exitcode','99',str(frozen),'quantized_full_mla_matches_f64_fixed_dynamic_cpu_fallback_graphs_reset_and_prefix','--nocapture','--test-threads=1'],extra=gpu,marker='ERROR SUMMARY: 0 errors')
    for variant,dll in [('baseline',baseline_library),('staged',library)]:
        command=[sys.executable,'-B',str(harness),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--backend','gpu','--cycles','4','--notes','4','--max-tokens','128','--device-mib','12288','--port','18138','--binary',str(binary),'--library',str(dll),'--policies','--model','glm47-flash','--mla-full','--split-kv','--moe-cache','8192','--trace-dir',str(out/'disabled-traces'),'--no-trace','--modes','lru','--output-dir',str(out/variant)]
        run(variant,command)
    observations={v:json.loads((out/v/'results.json').read_text(encoding='utf-8'))['rows'] for v in ['baseline','staged']}
    assert all(len(rows)==12 for rows in observations.values())
    def exact(rows):return [(r['cycle'],r['prompt'],r['request'],r['response']['choices'],r['response']['usage']) for r in rows]
    assert exact(observations['baseline'])==exact(observations['staged']), 'Changed actual responses or usage'
    assert all(r['matches_baseline'] for rows in observations.values() for r in rows)
    import statistics
    timing=[]
    for prompt in [0,1]:
        values={}
        for variant,rows in observations.items():
            selected=[r for r in rows if r['prompt']==prompt and r['cycle']>0]
            values[variant]=dict(decode_tps=[r['response']['usage']['completion_tokens']*1000/r['metrics_delta']['rbitnet_inference_decode_ms_sum'] for r in selected],prefill_ms=[r['metrics_delta']['rbitnet_inference_prefill_ms_sum'] for r in selected],wall_ms=[r['wall_ms'] for r in selected])
        medians={v:statistics.median(data['decode_tps']) for v,data in values.items()}
        timing.append(dict(prompt=prompt,values=values,median_decode_tps=medians,decode_change_percent=100*(medians['staged']/medians['baseline']-1)))
    (out/'summary.json').write_text(json.dumps(dict(rows=timing,exact_http_records=24,limits=manifest['limits']),indent=2)+'\n',encoding='utf-8')
    manifest.update(status='passed',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    print(json.dumps(dict(status='passed',timing=timing),indent=2),flush=True)
except BaseException as error:
    manifest.update(status='failed',error=str(error));save();raise
