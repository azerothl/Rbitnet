"""Ablate a fused encoded-KV store only after the serial lazy-host proof."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'llama-lazy-host-chain.log';deadline=time.monotonic()+43200
marker='Private lazy host Llama removed unused CPU KV planes and preserved exact actual format/page/graph, CPU fallback and cancel/replay fixtures; production serving pending.'
while not gate.exists()or marker not in gate.read_text(encoding='utf-8-sig'):
    assert time.monotonic()<deadline,'previous serialized device proof is incomplete'
    time.sleep(15)
out=base/'encoded-store-fusion-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=json.loads((base/'quantized-kv-canonical-proof/manifest.json').read_text(encoding='utf-8'))
baseline=base/'quantized-kv-canonical-native/cuda/rbitnet_cuda_quant64.dll';assert sha(baseline)==canonical['library_sha256']
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_TARGET_DIR=str(base/'check-release'))
def run(name,command,cwd=root,extra=None,required=None):
    print('Private encoded store fusion:',name,flush=True)
    p=out/(name+'.log')
    with p.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=cwd,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
    text=p.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-5000:])
    if required:assert required in text and 'running 0 tests'not in text,(name,text[-3000:])
    return text
run('prepare',[sys.executable,str(base/'prepare_encoded_store_fusion.py')])
native=base/'encoded-store-fusion-native';library=native/'cuda/rbitnet_cuda_quant64.dll'
run('native-build',['pwsh','-NoProfile','-File',str(native/'build_cuda.ps1'),'-OutDir',str(native/'cuda')])
run('independent-f64',[sys.executable,str(base/'check_encoded_attention_oracle.py'),str(library),str(out)],required='64 cases passed.')
crate=base/'check-encoded-fusion';crate.mkdir(exist_ok=True);source=base/'check-quantized-canonical'
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(source/name,crate/name)
for name in ['crates','recipes','.cargo','tests']:
    if(source/name).exists():shutil.copytree(source/name,crate/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
    elif name=='tests':shutil.copytree(root/name,crate/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
core=crate/'crates/bitnet-core/src';shutil.copy2(base/'encoded_guard_tests_draft.rs',core/'llama/resident/encoded_guard_tests.rs')
p=core/'llama/resident.rs';p.write_text(p.read_text(encoding='utf-8')+'\n#[cfg(test)]\n#[path="resident/encoded_guard_tests.rs"]\nmod encoded_guard_tests;\n',encoding='utf-8',newline='\n')
subprocess.run(['rustfmt','--edition','2021','--config','skip_children=true',str(core/'llama/resident/encoded_guard_tests.rs')],cwd=root,check=True)
run('check',['cargo','check','--workspace','--all-targets'],crate)
run('clippy',['cargo','clippy','--workspace','--all-targets'],crate)
gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_CUDA_PREFILL_TF32X3='0',
    RBITNET_KV_CANONICAL_TEST='1',RBITNET_CUDA_KV_TEST='1',RBITNET_CUDA_PAGES_TEST='1',RBITNET_ENCODED_GUARD_TEST='1',RBITNET_TEST_GGUF=str(root/'models/exported-llama/model.gguf'),
    RBITNET_TOKENIZER=str(root/'models/exported-llama/tokenizer.json'),RBITNET_KV_QUALITY_PLAN=str(base/'kv-quality-plan.json'))
tests=[('guard','optional_actual_encoded_tf32_mutable_configuration_refused','ENCODED_GUARD_DONE'),
    ('canonical','optional_actual_encoded_prefill_partition_prefix_restore_exact','KV_CANONICAL_DONE'),('quality-lifetime','resident::quantized_tests','ACTUAL_KV_LIFETIME')]
for split in ['0','1']:
    for label,test,required in tests:run(label+'-split'+split,['cargo','test','-p','bitnet-core','--release','--lib',test,'--','--nocapture','--test-threads=1'],crate,gpu|{'RBITNET_CUDA_SPLIT_KV':split},required)
    for fmt in ['f16','q8']:run('pages-'+fmt+'-split'+split,['cargo','test','-p','bitnet-core','--release','--lib','resident::paged_tests','--','--nocapture','--test-threads=1'],crate,gpu|dict(RBITNET_CUDA_SPLIT_KV=split,RBITNET_CUDA_KV_FORMAT=fmt),'PAGED_OWNER dense foreign/')
# Cross-library comparison: exact serial and cached runtime full-vocabulary
# vectors from the immediately preceding lazy-host proof are authoritative.
host_proof=base/'llama-lazy-host-proof';hm=json.loads((host_proof/'manifest.json').read_text(encoding='utf-8'))
test_binary=host_proof/'lazy-bitnet-core.exe';assert sha(test_binary)==hm['test_binary_sha256']['lazy']
reference=json.loads((host_proof/'results.json').read_text(encoding='utf-8'))['lazy'];actual={}
common=gpu|dict(RBITNET_MAX_SEQ='8192',RBITNET_CUDA_PREFILL='1',RBITNET_CUDA_SPLIT_KV='1',RBITNET_PREFIX_KV='1',RBITNET_CUDA_PREFIX_MB='256',RBITNET_CUDA_PREFIX_ENTRIES='8',
    RBITNET_LAZY_HOST_TEST='1',RBITNET_LAZY_HOST_EXPECT='1',RBITNET_LAZY_BACKEND='cuda',RBITNET_LLAMA_SPECULATIVE='0')
for fmt in ['f32','f16','q8']:
    for layout in ['dense','paged']:
        for graph in ['0','1']:
            name=f'{fmt}-{layout}-graph{graph}';text=run('cross-library-'+name,[str(test_binary),'optional_actual_llama_lazy_host_planes_runtime_replay','--nocapture','--test-threads=1'],extra=common|dict(RBITNET_CUDA_KV_FORMAT=fmt,RBITNET_CUDA_KV_PAGE_LIMIT='256'if layout=='paged'else'0',RBITNET_CUDA_RESIDENT_GRAPH=graph),required='LAZY_HOST_DONE')
            rows=re.findall(r'LAZY_HOST_RESULT (\{[^\r\n]+\})',text);assert len(rows)==1;row=json.loads(rows[0]);old=reference[name]
            for key in ['logits_bits','next_bits']:row[key+'_sha256']=hashlib.sha256(json.dumps(row.pop(key),separators=(',',':')).encode()).hexdigest()
            for key in old:
                if key!='peak_process_rss':assert row[key]==old[key],(name,key)
            actual[name]=row
binary=base/'quantized-kv-canonical-proof/rbitnet.exe';assert sha(binary)==canonical['binary_sha256']
for variant,lib in [('baseline',baseline),('fused',library)]:
    for context,notes in [('2048','24'),('8192','96')]:
        for layout in ['dense','paged']:
            folder=out/'ablation'/variant/(layout+'-context'+context)
            run('quiet-'+variant+'-'+layout+'-context'+context,[sys.executable,str(base/'quantized-harness/benchmark.py'),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
                '--backend','gpu','--cycles','3','--max-tokens','128','--device-mib','12288','--port','18138','--binary',str(binary),'--library',str(lib),'--model','llama32-1b','--paged',
                '--notes',notes,'--context',context,'--layout',layout,'--output-dir',str(folder)])
            r=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert len(r['rows'])==54 and len(r['sse'])==18 and len(r['stops'])==6 and all(x['matches_baseline']for x in r['rows'])
            if variant=='fused':
                old=json.loads((out/'ablation/baseline'/(layout+'-context'+context)/'results.json').read_text(encoding='utf-8'))
                for x,y in zip(r['rows'],old['rows'],strict=True):assert x['response']['choices']==y['response']['choices'],(layout,context,x['mode'],x['cycle'],x['prompt'])
for layout in ['dense','paged']:
    for fmt in ['f16','q8']:
        run('network-'+layout+'-'+fmt,[sys.executable,str(base/'quantized-harness/live.py'),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
            '--binary',str(binary),'--library',str(library),'--paged','--split-kv','--layout',layout,'--format',fmt,'--port','18138','--device-mib','12288','--output-dir',str(out/'live'/(layout+'-'+fmt))])
(out/'exact-cross-library.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'manifest.json').write_text(json.dumps(dict(canonical_baseline=canonical,lazy_host_baseline=hm,library_sha256=sha(library),binary_sha256=sha(binary),
    native_source_sha256={p.name:sha(p)for p in native.iterdir()if p.is_file()},source_sha256={p.relative_to(crate).as_posix():sha(p)for p in(crate/'crates').rglob('*.rs')},
    protocol=dict(contexts=[2048,8192],notes=[24,96],output_tokens=128,cycles=3),
    limits=['Private fused native library, no production adoption.','F32 retains its existing kernels. Encoded numerical outputs must remain bit exact to the preceding canonical library.',
    'One fused store launch replaces separate RoPE and store launches per encoded layer. The original quantization, scales and reduction order are retained.',
    'Mutable encoded TF32 configuration is refused as well as encoded TF32 construction.','Quiet tests are separate from all correctness fixtures; default selection requires the measured result.']),indent=2)+'\n',encoding='utf-8')
print('Private fused encoded RoPE/store passed exact canonical and cross-library logits, F64, ownership, quiet and network suites; adoption pending.',flush=True)
