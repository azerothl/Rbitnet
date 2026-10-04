"""Run actual driver/thread/network proofs after the preceding serial owner."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys
root=Path.cwd();base=root/'target/performance-cache';out=base/'llama-continuous-proof';out.mkdir(exist_ok=True)
state=json.loads((base/'serial-experiments.json').read_text(encoding='utf-8'))
assert state['complete']and all(row['status']=='passed'for row in state['stages']),'earlier serial hardware owner is not complete'
native=base/'llama-batch-native';library=native/'cuda/rbitnet_cuda_quant64.dll'
batch=json.loads((base/'llama-batch-proof/manifest.json').read_text(encoding='utf-8'))
finish=json.loads((base/'finish-reason-proof/manifest.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(library)==batch['library_sha256']
assert all(sha(native/name)==digest for name,digest in batch['native_source_sha256'].items())
reference=base/'finish-reason-proof/rbitnet.exe';reference_library=base/'finish-reason-baseline-native.dll'
assert sha(reference)==finish['binary_sha256']and sha(reference_library)==finish['library_sha256']
env={key:value for key,value in os.environ.items()if not key.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',RAYON_NUM_THREADS='16',RBITNET_CUDA='0',CARGO_INCREMENTAL='0',CARGO_TARGET_DIR=str(base/'check-build'))
def run(name,command,cwd=root,extra=None,markers=(),timeout=3600):
 print('Private continuous Llama:',name,flush=True);p=out/(name+'.log')
 with p.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=cwd,env=env|(extra or{}),stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 text=p.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-6000:])
 for marker in markers:assert marker in text and 'running 0 tests'not in text,(name,marker,text[-3000:])
 return text
run('prepare',[sys.executable,str(base/'prepare_llama_continuous.py')])
crate=base/'check-llama-continuous'
run('check',['cargo','check','--workspace','--all-targets'],crate)
run('clippy',['cargo','clippy','--workspace','--all-targets'],crate)
run('workspace',['cargo','test','--workspace','--','--test-threads=1'],crate)
env['CARGO_TARGET_DIR']=str(base/'check-release')
gpu=dict(RBITNET_CUDA_QUANT_LIB=str(library),RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_LLAMA_CONTINUOUS_TEST='1',
 RBITNET_TEST_GGUF=str(root/'models/exported-llama/model.gguf'),RBITNET_TOKENIZER=str(root/'models/exported-llama/tokenizer.json'))
text=run('actual-driver-and-threads',['cargo','test','-p','bitnet-core','--release','--lib','optional_actual_llama_continuous','--','--nocapture','--test-threads=1'],crate,gpu,
 ('LLAMA_CONTINUOUS_DONE cases=24','LLAMA_CONTINUOUS_THREAD_DONE layouts=2','2 passed; 0 failed'))
rows=[json.loads(row)for row in re.findall(r'LLAMA_CONTINUOUS_CASE (\{[^\r\n]+\})',text)]
threads=[json.loads(row)for row in re.findall(r'LLAMA_CONTINUOUS_THREAD_CASE (\{[^\r\n]+\})',text)]
assert len(rows)==24 and len(threads)==2
run('release',['cargo','build','-p','rbitnet-cli','--release'],crate)
binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
run('actual-network-and-quiet-arrivals',[sys.executable,str(base/'llama_continuous_live.py'),'--binary',str(binary),'--library',str(library),
 '--reference-binary',str(reference),'--reference-library',str(reference_library),'--output',str(out/'live')],
 markers=('LLAMA_CONTINUOUS_HTTP_DONE layouts=2 capacities=3 serial_cases=54 waves=21 owned_disconnects=6 explicit_stops=6',))
live=json.loads((out/'live/results.json').read_text(encoding='utf-8'));assert live['complete']and len(live['waves'])==21 and len(live['cases'])==66
manifest=dict(binary_sha256=sha(binary),library_sha256=sha(library),finish_reason_proof=finish,native_batch_proof=batch,
 source_sha256={p.relative_to(crate).as_posix():sha(p)for p in(crate/'crates').rglob('*.rs')},harness_sha256=sha(base/'llama_continuous_live.py'),
 actual_driver_cases=rows,thread_cases=threads,
 limits=['Private opt-in individual HTTP requests route to a worker with real Native shared forwards; production adoption pending.',
 'Only CUDA Llama F32 mono, normal 128-token prefill partitions, dense or shared-page-pool owners; split-KV/TF32/prefix/speculative/legacy batching combinations refused.',
 'No batch CUDA graph, no true multi-request prefill GEMM, no Qwen/GPT/GLM scheduler here; token budget covers actual single-owner prefill chunks plus shared decode rows.',
 'Request-local channel cancellation covers callback failure/disconnection and drop; server unary timeout and explicit stop may allow bounded remaining work before retirement.',
 'Structured JSON/tool grammar requests are refused before GPU admission; the legacy token-ID-as-ASCII mask is incompatible with real subword tokenizers.',
 'Quiet wave latencies include eight clients, 20ms delayed arrivals and CPU sampling; measured cycles=2 after one warmup, no Nsight trace.',
 'SSE chunk intervals are reported separately from logical GPU token intervals; no claim of parity with Ollama/llama.cpp without fresh comparison.',
 'Engine.complete_batch still uses the legacy scheduler path; the individual concurrent HTTP executor path is implemented by this private prototype.'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Private Llama continuous Native decode passed exact variable arrivals, request-local ownership, thread shutdown and live concurrent JSON/SSE/quiet suites; adoption and reference-engine benchmark pending.',flush=True)
