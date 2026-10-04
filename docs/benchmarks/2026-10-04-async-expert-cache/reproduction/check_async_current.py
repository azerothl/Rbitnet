"""Serialize rebased async pool type/lifetime/GGUF proof after host context checks."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';out=base/'check-async-current'
gate=base/'context-trained-chain.log';deadline=time.monotonic()+10800
while 'Isolated context/SentencePiece checks passed; production source and GPU unchanged.'not in gate.read_text(encoding='utf-8-sig'):
    assert time.monotonic()<deadline,'context checks did not finish; refusing concurrent compilation'
    time.sleep(15)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16')
for name in ['prepare_async_cache.py','prepare_async_runtime_test.py']:
    subprocess.run([sys.executable,str(base/name)],cwd=root,env=env,check=True)
out.mkdir(exist_ok=True)
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(root/name,out/name)
shutil.copytree(root/'crates',out/'crates',dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
shutil.copytree(root/'recipes',out/'recipes',dirs_exist_ok=True)
if(root/'.cargo').exists():shutil.copytree(root/'.cargo',out/'.cargo',dirs_exist_ok=True)
core=out/'crates/bitnet-core/src';src=base/'async-integration'
copies={'backend.rs':'backend.rs','async_upload.rs':'backend/async_upload.rs','weights.rs':'native/weights.rs',
 'expert_cache.rs':'native/expert_cache.rs','async_cache.rs':'native/expert_cache/async_cache.rs',
 'tests.rs':'native/expert_cache/async_cache/tests.rs','moe.rs':'native/moe.rs','moe_metrics.rs':'native/moe_metrics.rs','perf.rs':'perf.rs'}
for source,dest in copies.items():
    p=core/dest;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/source,p)
shutil.copy2(base/'async_runtime_tests_draft.rs',core/'native/async_runtime_tests.rs')
shutil.copy2(base/'async_profile_tests_draft.rs',core/'native/async_profile_tests.rs')
p=core/'native/graph.rs';p.write_text(p.read_text(encoding='utf-8')+'\n#[cfg(test)]\n#[path="async_runtime_tests.rs"]\nmod async_runtime_tests;\n',encoding='utf-8',newline='\n')
p.write_text(p.read_text(encoding='utf-8')+'\n#[cfg(test)]\n#[path="async_profile_tests.rs"]\nmod async_profile_tests;\n',encoding='utf-8',newline='\n')
proof=base/'async-current-proof';proof.mkdir(exist_ok=True)
env['CARGO_TARGET_DIR']=str(base/'check-build')
def run(name,command,extra=None,marker=None):
    print('Actual optional async proof',name,flush=True);path=proof/(name+'.log')
    with path.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=out,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
    text=path.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-5000:])
    if marker:assert marker in text and 'running 0 tests'not in text,(name,text[-3000:])
run('check',['cargo','check','--workspace','--all-targets'])
run('clippy',['cargo','clippy','--workspace','--all-targets'])
env['CARGO_TARGET_DIR']=str(base/'check-release')
gpu={'RBITNET_CUDA_QUANT_LIB':str(root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'),
 'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_ASYNC_TEST':'1',
 'RBITNET_TEST_GGUF':'D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf'}
gpu['RBITNET_TEST_MIXED_GGUF']='D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf'
for name in ['optional_async_upload_publishes_only_after_events_and_drains_on_drop','optional_async_pool_uses_one_budget_protects_ready_and_pending_and_drains_unload','optional_async_mixed_q4_q6_refill_keeps_capacity_and_original_bytes']:
    run(name,['cargo','test','-p','bitnet-core','--release','--lib',name,'--','--nocapture','--test-threads=1'],gpu,'1 passed; 0 failed')
for name,family,path,tokenizer in [('gpt','gptoss',gpu['RBITNET_TEST_GGUF'],'gpt-oss-20b'),
    ('glm','mla','D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf','GLM-4.7-Flash')]:
    actual=gpu|{'RBITNET_MOE_POLICY_TEST':'1','RBITNET_MOE_POLICY_FAMILY':family,'RBITNET_MOE_POLICY_GGUF':path,
        'RBITNET_MOE_POLICY_TOKENIZER':str(root/'target/engine-benchmark/tokenizers'/tokenizer/'tokenizer.json'),'RBITNET_MOE_POLICY_CACHE':'8192'}
    for budget in ['8192','512']:
        case=actual|{'RBITNET_MOE_POLICY_CACHE':budget,'RBITNET_MOE_REQUIRE_PREFETCH':'1' if budget=='512' else '0'}
        run('real-'+name+'-cache'+budget,['cargo','test','-p','bitnet-core','--release','--lib','real_moe_async_cache_preserves_logits_generations_prefix_and_model_lifetime','--','--nocapture','--test-threads=1'],case,'MoE async real worst KL=')
manifest={'source_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in (out/'crates').rglob('*.rs')},
    'library_sha256':hashlib.sha256(Path(gpu['RBITNET_CUDA_QUANT_LIB']).read_bytes()).hexdigest()}
test_binary=max((base/'check-release/release/deps').glob('bitnet_core-*.exe'),key=lambda p:p.stat().st_mtime)
shutil.copy2(test_binary,proof/'bitnet-core-async.exe')
manifest['frozen_test_binary_sha256']=hashlib.sha256((proof/'bitnet-core-async.exe').read_bytes()).hexdigest()
(proof/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Optional async current sources, actual GPU lifetime and two GGUF numerical suites passed; live, overlap trace and performance still pending.',flush=True)
