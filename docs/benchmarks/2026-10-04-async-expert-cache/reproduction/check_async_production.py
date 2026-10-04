"""Adopt the bounded asynchronous expert cache in the next quiet GPU window."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time

root=Path.cwd();base=root/'target/performance-cache';dest=root/'target/async-expert-cache';proof=dest/'production-proof'
deadline=time.monotonic()+43200
for path,marker in [(base/'async-lifecycle-chain.log','Actual async GPT/GLM scoped READY/PENDING/pinned budget and unload/reload lifecycle suites passed.'),
                    (base/'kv-integration-chain.log','Private actual encoded Llama attention, quality, physical accounting and lifetime suites passed; serving/performance/adoption pending.')]:
    while not path.exists()or marker not in path.read_text(encoding='utf-8-sig'):
        assert time.monotonic()<deadline,'quiet/lifetime checks not complete; no adoption or build started'
        time.sleep(15)
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='codex/paged-kv'
for command in [['git','diff','--name-only'],['git','diff','--cached','--name-only']]:
    assert not subprocess.check_output(command,cwd=root,text=True).strip(),'deliver existing tracked work before adoption'
subprocess.run(['git','checkout','-b','codex/async-expert-cache'],cwd=root,check=True)
subprocess.run([sys.executable,str(base/'prepare_async_adoption.py')],cwd=root,check=True)
core=root/'crates/bitnet-core/src';src=base/'async-adoption';paths=[]
copies={'backend.rs':'backend.rs','async_upload.rs':'backend/async_upload.rs','weights.rs':'native/weights.rs',
        'expert_cache.rs':'native/expert_cache.rs','async_cache.rs':'native/expert_cache/async_cache.rs',
        'tests.rs':'native/expert_cache/async_cache/tests.rs','moe.rs':'native/moe.rs','moe_metrics.rs':'native/moe_metrics.rs','perf.rs':'perf.rs'}
for source,relative in copies.items():
    p=core/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/source,p);paths.append(p.relative_to(root).as_posix())
p=core/'native/expert_cache/async_cache.rs';s=p.read_text(encoding='utf-8')
s=s.replace('//! Opt-in asynchronous cache draft. One physical pool, bounded pinned staging.\n//! Child of expert_cache.rs; type/GPU/live validation is still pending.',
            '//! Opt-in asynchronous expert cache with one physical pool and bounded pinned staging.\n//! Pending copies become visible only after their completion event succeeds.')
p.write_text(s,encoding='utf-8',newline='\n')
p=core/'backend/async_upload.rs';s=p.read_text(encoding='utf-8')
s=s.replace('//! Draft #84: bounded pinned transfers with unpublished device owners.\n//! Include as a child of backend.rs; compile/GPU/live verification is pending.',
            '//! Bounded pinned transfers with unpublished device owners.\n//! Each transfer retains its source and destination until its event completes.')
p.write_text(s,encoding='utf-8',newline='\n')
for source,dest_name in [('async_runtime_tests_draft.rs','async_runtime_tests.rs'),('async_profile_tests_draft.rs','async_profile_tests.rs'),('async_config_tests_draft.rs','async_config_tests.rs')]:
    p=core/'native'/dest_name;shutil.copy2(base/source,p);paths.append(p.relative_to(root).as_posix())
p=core/'native/graph.rs';s=p.read_text(encoding='utf-8')
for name in ['async_runtime_tests','async_profile_tests','async_config_tests']:
    assert f'mod {name};'not in s
    s+=f'\n#[cfg(test)]\n#[path = "{name}.rs"]\nmod {name};\n'
p.write_text(s,encoding='utf-8',newline='\n');paths.append(p.relative_to(root).as_posix())
for relative in paths:subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true',str(root/relative)],cwd=root,check=True)
proof.mkdir(parents=True,exist_ok=True);(dest/'paths.json').write_text(json.dumps(paths,indent=2)+'\n',encoding='utf-8')
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16')
def run(name,command,extra=None,marker=None):
    print('Adopted asynchronous expert cache:',name,flush=True);path=proof/(name+'.log')
    with path.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
    text=path.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-6000:])
    if marker:assert marker in text and 'running 0 tests'not in text,(name,text[-3000:])
run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
run('clippy',['cargo','clippy','--workspace','--all-targets'])
lib=root/'target/paged-kv/cuda/rbitnet_cuda_quant64.dll'
page_manifest=json.loads((root/'target/paged-kv/production-proof/manifest.json').read_text(encoding='utf-8'))
assert hashlib.sha256(lib.read_bytes()).hexdigest()==page_manifest['library_sha256']
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in page_manifest['native_source_sha256'].items())
gpu={'RBITNET_CUDA_QUANT_LIB':str(lib),'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_ASYNC_TEST':'1',
     'RBITNET_TEST_GGUF':'D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',
     'RBITNET_TEST_MIXED_GGUF':'D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf'}
for name in ['optional_async_upload_publishes_only_after_events_and_drains_on_drop','optional_async_pool_uses_one_budget_protects_ready_and_pending_and_drains_unload','optional_async_mixed_q4_q6_refill_keeps_capacity_and_original_bytes']:
    run(name,['cargo','test','-p','bitnet-core','--release','--lib',name,'--','--nocapture','--test-threads=1'],gpu,'1 passed; 0 failed')
run('actual-config-refusals',['cargo','test','-p','bitnet-core','--release','--lib','optional_async_configuration_refuses_silent_fallback_and_releases','--','--nocapture','--test-threads=1'],gpu|{'RBITNET_ASYNC_CONFIG_TEST':'1'},'nine actual admission checks passed')
for name,family,path,tokenizer in [('gpt','gptoss',gpu['RBITNET_TEST_GGUF'],'gpt-oss-20b'),('glm','mla',gpu['RBITNET_TEST_MIXED_GGUF'],'GLM-4.7-Flash')]:
    for budget in ['8192','512']:
        actual=gpu|{'RBITNET_MOE_POLICY_TEST':'1','RBITNET_MOE_POLICY_FAMILY':family,'RBITNET_MOE_POLICY_GGUF':path,
                    'RBITNET_MOE_POLICY_TOKENIZER':str(root/'target/engine-benchmark/tokenizers'/tokenizer/'tokenizer.json'),
                    'RBITNET_MOE_POLICY_CACHE':budget,'RBITNET_MOE_REQUIRE_PREFETCH':'1'if budget=='512'else'0'}
        run('real-'+name+'-cache'+budget,['cargo','test','-p','bitnet-core','--release','--lib','real_moe_async_cache_preserves_logits_generations_prefix_and_model_lifetime','--','--nocapture','--test-threads=1'],actual,'MoE async real worst KL=')
run('release',['cargo','build','--release','-p','rbitnet-cli'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(root/'target/release/rbitnet.exe',proof/'rbitnet.exe')
test_binary=max((root/'target/release/deps').glob('bitnet_core-*.exe'),key=lambda p:p.stat().st_mtime)
shutil.copy2(test_binary,proof/'bitnet-core-async.exe')
manifest={'source_sha256':{p:sha(root/p)for p in paths},'binary_sha256':sha(proof/'rbitnet.exe'),'frozen_test_binary_sha256':sha(proof/'bitnet-core-async.exe'),
          'library_sha256':sha(lib),'native_source_sha256':{p.relative_to(root).as_posix():sha(p)for folder in ['native/cuda_quant/src','native/cuda_quant/include']for p in(root/folder).iterdir()if p.is_file()}}
(proof/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
common=['--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--backend','gpu','--cycles','3','--notes','8','--max-tokens','32','--device-mib','12288','--port','18138',
        '--binary',str(proof/'rbitnet.exe'),'--library',str(lib),'--async']
# Recheck every policy on the adopted binary. The earlier full budget sweep
# remains a separate prototype result, with distinct source and binary hashes.
for model in ['gpt-oss-20b','glm47-flash']:
    opts=['--gpt-full','--gpt-segmented']if model=='gpt-oss-20b'else['--mla-full','--split-kv']
    run('quiet-'+model+'-cache8192',[sys.executable,str(base/'followup-harness/async_benchmark.py'),*common,*opts,'--model',model,'--moe-cache','8192','--output-dir',str(dest/'ablation'/model)])
    r=json.loads((dest/'ablation'/model/'results.json').read_text(encoding='utf-8'));assert len(r['rows'])==27 and all(x['matches_baseline']for x in r['rows'])and len(r['sse'])==9 and len(r['stops'])==3
    run('network-'+model,[sys.executable,str(base/'followup-harness/async_live.py'),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
                         '--binary',str(proof/'rbitnet.exe'),'--library',str(lib),'--async','--split-kv',*opts,'--moe-cache','512','--output-dir',str(dest/'live'/model),'--port','18138','--device-mib','12288'])
# Adapt the exact lifecycle driver and profiler to the frozen adopted artifacts.
scripts=dest/'reproduction';scripts.mkdir(exist_ok=True)
s=(base/'run_async_lifecycle.py').read_text(encoding='utf-8');at=s.index('sys.path.insert(0,')
s=s[:s.index("root=Path.cwd();")]+"root=Path.cwd();base=root/'target/performance-cache'\n"+s[at:]
s=s.replace("out=base/'async-lifecycle';out.mkdir(exist_ok=True);proof=base/'async-current-proof'", "out=root/'target/async-expert-cache/lifecycle';out.mkdir(exist_ok=True);proof=root/'target/async-expert-cache/production-proof'")
s=s.replace("root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'", "root/'target/paged-kv/cuda/rbitnet_cuda_quant64.dll'").replace("m['cli_sha256']", "m['binary_sha256']")
(scripts/'lifecycle.py').write_text(s,encoding='utf-8',newline='\n');run('lifecycle',[sys.executable,str(scripts/'lifecycle.py')],marker='Actual async GPT/GLM scoped READY/PENDING/pinned budget and unload/reload lifecycle suites passed.')
s=(base/'profile_async_current.py').read_text(encoding='utf-8');at=s.index('manifest=json.loads(')
s=s[:s.index("root=Path.cwd();")]+"root=Path.cwd();base=root/'target/performance-cache';proof=root/'target/async-expert-cache/production-proof'\n"+s[at:]
s=s.replace("root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'", "root/'target/paged-kv/cuda/rbitnet_cuda_quant64.dll'")
(scripts/'profile.py').write_text(s,encoding='utf-8',newline='\n');run('profile',[sys.executable,str(scripts/'profile.py')],marker='Actual warmed async CUDA report and SQLite exported;')
s=(base/'analyze_async_trace.py').read_text(encoding='utf-8').replace("base=Path(__file__).resolve().parent/'async-current-proof'", "base=Path.cwd()/'target/async-expert-cache/production-proof'")
(scripts/'analyze_trace.py').write_text(s,encoding='utf-8',newline='\n');run('trace-analysis',[sys.executable,str(scripts/'analyze_trace.py')])
assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items()),'adopted source changed during validation'
print('Adopted async expert cache workspace, actual GPU/GGUF, quiet, network, lifecycle and overlap suites passed.',flush=True)
