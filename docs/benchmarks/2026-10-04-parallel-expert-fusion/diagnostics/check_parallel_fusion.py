"""Serialize the parallel-expert decode experiment after the page ablation."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'page-fastpath-chain.log';deadline=time.monotonic()+43200
while not gate.exists()or 'Private exclusive-page and pointer-hoisting ablations passed exact ownership, quiet and network suites; adoption pending.'not in gate.read_text(encoding='utf-8-sig'):
    assert time.monotonic()<deadline,'previous serial GPU experiment not complete'
    time.sleep(15)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0')
for script in ['prepare_parallel_fusion.py','prepare_parallel_fusion_harness.py']:subprocess.run([sys.executable,str(base/script)],cwd=root,env=env,check=True)
src=base/'parallel-fusion';out=base/'parallel-fusion-proof';out.mkdir(exist_ok=True);native=out/'native';native.mkdir(exist_ok=True)
for folder in [root/'native/cuda_quant/src',root/'native/cuda_quant/include']:
    for p in folder.iterdir():
        if p.is_file():shutil.copy2(p,native/p.name)
for name in ['moe_fused.cuh','moe_resident.cuh','moe_parallel_fused.cuh']:shutil.copy2(src/name,native/name)
script=(root/'scripts/build_cuda_quant.ps1').read_text(encoding='utf-8-sig').replace('$Src = Join-Path $Root "native\\cuda_quant\\src\\quant_matvec.cu"','$Src = Join-Path $PSScriptRoot "quant_matvec.cu"').replace('$Inc = Join-Path $Root "native\\cuda_quant\\include"','$Inc = $PSScriptRoot')
(native/'build.ps1').write_text(script,encoding='utf-8',newline='\n')
def run(name,command,cwd=root,extra=None,marker=None):
    print('Parallel selected-expert experiment:',name,flush=True)
    with(out/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=cwd,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
    text=(out/(name+'.log')).read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-6000:])
    if marker:assert marker in text and 'running 0 tests'not in text,(name,text[-3000:])
run('native-build',['pwsh','-NoProfile','-File',str(native/'build.ps1'),'-OutDir',str(native/'cuda')])
crate=base/'check-parallel-fusion';crate.mkdir(exist_ok=True)
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(root/name,crate/name)
for name in ['crates','recipes','.cargo','tests']:
    if(root/name).exists():shutil.copytree(root/name,crate/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
core=crate/'crates/bitnet-core/src/native';shutil.copy2(src/'moe.rs',core/'moe.rs')
for name in ['parallel_fused_tests.rs','parallel_gpt_tests.rs','parallel_mla_tests.rs']:shutil.copy2(src/name,core/name)
for filename,modules in [('moe.rs',['parallel_fused_tests']),('graph.rs',['parallel_gpt_tests','parallel_mla_tests'])]:
    p=core/filename;s=p.read_text(encoding='utf-8')
    for name in modules:s+=f'\n#[cfg(test)]\n#[path="{name}.rs"]\nmod {name};\n'
    p.write_text(s,encoding='utf-8',newline='\n')
for p in [core/'moe.rs',core/'graph.rs',*core.glob('parallel_*tests.rs')]:subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true',str(p)],cwd=root,check=True)
env['CARGO_TARGET_DIR']=str(base/'check-build');run('check',['cargo','check','--workspace','--all-targets'],crate);run('clippy',['cargo','clippy','--workspace','--all-targets'],crate)
env['CARGO_TARGET_DIR']=str(base/'check-release');library=native/'cuda/rbitnet_cuda_quant64.dll'
gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_MAX_SEQ='2048',RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off')
run('actual-f64-all-formats',['cargo','test','-p','bitnet-core','--release','--lib','parallel_fused_fixed_dynamic_all_formats_mixed_bias_graph_refill_matches_original_bits_and_f64','--','--nocapture','--test-threads=1'],crate,gpu|{'RBITNET_CUDA_QUANT_SMOKE':'1'},'1 passed; 0 failed')
run('actual-gpt',['cargo','test','-p','bitnet-core','--release','--lib','actual_gpt_parallel_fusion_logits_seeded_outputs_prefixes_and_cancellation','--','--nocapture','--test-threads=1'],crate,
    gpu|{'RBITNET_GPT_BLOCK_RUNTIME_TEST':'1','RBITNET_GPT_BLOCK_GGUF':'D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',
         'RBITNET_GPT_BLOCK_TOKENIZER':str(root/'target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json')},'1 passed; 0 failed')
run('actual-glm',['cargo','test','-p','bitnet-core','--release','--lib','actual_mla_parallel_fusion_teacher_forcing_greedy_seed_penalty_and_reset','--','--nocapture','--test-threads=1'],crate,
    gpu|{'RBITNET_MLA_FULL_TEST':'1','RBITNET_MLA_TEST_GGUF':'D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf',
         'RBITNET_MLA_TEST_TOKENIZER':str(root/'target/engine-benchmark/tokenizers/GLM-4.7-Flash/tokenizer.json')},'1 passed; 0 failed')
run('release',['cargo','build','--release','-p','rbitnet-cli'],crate);shutil.copy2(base/'check-release/release/rbitnet.exe',out/'rbitnet.exe')
common=['--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--backend','gpu','--cycles','3','--notes','8','--max-tokens','128','--device-mib','12288','--port','18138',
        '--binary',str(out/'rbitnet.exe'),'--library',str(library),'--parallel-fusion']
for label,model,budget,opts in [('gpt-fixed','gpt-oss-20b','0',['--gpt-full']),('gpt-segmented','gpt-oss-20b','8192',['--gpt-full','--gpt-segmented']),
                               ('glm-fixed','glm47-flash','0',['--mla-full','--split-kv']),('glm-cache','glm47-flash','8192',['--mla-full','--split-kv'])]:
    folder=out/'ablation'/label
    run('quiet-'+label,[sys.executable,str(base/'parallel-fusion-harness/benchmark.py'),*common,*opts,'--model',model,'--moe-cache',budget,'--output-dir',str(folder)])
    r=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert len(r['rows'])==27 and len(r['sse'])==9 and len(r['stops'])==3 and all(x['matches_baseline']for x in r['rows'])
for model,opts in [('gpt-oss-20b',['--gpt-full','--gpt-segmented']),('glm47-flash',['--mla-full','--split-kv'])]:
    run('network-'+model,[sys.executable,str(base/'parallel-fusion-harness/live.py'),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(out/'rbitnet.exe'),'--library',str(library),
         '--async','--fusion','2','--split-kv',*opts,'--moe-cache','8192','--output-dir',str(out/'live'/model),'--port','18138','--device-mib','12288'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(dict(binary_sha256=sha(out/'rbitnet.exe'),library_sha256=sha(library),
    native_source_sha256={p.name:sha(p)for p in native.iterdir()if p.is_file()},rust_source_sha256={p.relative_to(crate).as_posix():sha(p)for p in(crate/'crates').rglob('*.rs')},
    limits=['Private parallel fusion, not adopted production.','Gate/up retains existing fusion; only the selected-expert down/combine is parallelized.','Original warp and slot reduction order is preserved; exact/F64 fixtures run separately from performance.']),indent=2)+'\n',encoding='utf-8')
print('Private parallel expert down/combine passed independent exact/F64, actual GPT/GLM, quiet and network suites; adoption pending.',flush=True)
