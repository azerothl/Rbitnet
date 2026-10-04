"""Capture a warmed actual expert-copy/compute CUDA timeline, outside benchmarks."""
from pathlib import Path
import hashlib,json,os,subprocess,time,uuid
root=Path.cwd();base=root/'target/performance-cache';proof=root/'target/async-expert-cache/production-proof'
manifest=json.loads((proof/'manifest.json').read_text());binary=proof/'bitnet-core-async.exe'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==manifest['frozen_test_binary_sha256']
nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe')
report=proof/'warmed-async-cuda'
nonce=str(uuid.uuid4());child_proof=proof/('profile-child-'+nonce+'.json')
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')}
env.update(PYTHONIOENCODING='utf-8',RAYON_NUM_THREADS='16',RBITNET_CUDA_QUANT_LIB=str(root/'target/paged-kv/cuda/rbitnet_cuda_quant64.dll'),
 RBITNET_CUDA_DEVICE_BUDGET_MB='12288',RBITNET_CUDA_DEVICE_MARGIN_MB='256',RBITNET_MOE_ASYNC_PROFILE='1',
 RBITNET_MOE_POLICY_GGUF='D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',
 RBITNET_MOE_POLICY_TOKENIZER=str(root/'target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json'),
 RBITNET_CUDA_PROFILE_RUNTIME='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64/cudart64_13.dll',
 RBITNET_CUDA_PROFILE_PROOF=str(child_proof),RBITNET_CUDA_PROFILE_NONCE=nonce)
command=[str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-memory-usage=true','--kill=false','--wait=primary','--show-output=true',
 '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output='+str(report),
 str(binary),'optional_async_real_gpt_warmed_cuda_profiler_range','--nocapture','--test-threads=1']
print('Nsight warmed actual GPT expert-copy range starting; not a throughput measurement.',flush=True)
with(proof/'nsight-profile.log').open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
text=(proof/'nsight-profile.log').read_text(encoding='utf-8');assert r.returncode==0,(r.returncode,text[-5000:])
assert child_proof.is_file(),('profiler child did not write its completion proof',text[-3000:])
completed=json.loads(child_proof.read_text(encoding='utf-8'))
assert completed['nonce']==nonce and completed['same_output'] and completed['profiler_stop_succeeded'] and completed['async_failed']==0 and completed['prefetch_copies']>0,completed
(proof/'profile-child-proof.json').write_text(json.dumps(completed,indent=2)+'\n',encoding='utf-8')
with(proof/'nsight-export.log').open('w',encoding='utf-8')as log:
    r=subprocess.run([str(nsys),'export','--type=sqlite','--force-overwrite=true','--output='+str(report)+'.sqlite',str(report)+'.nsys-rep'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert r.returncode==0
print('Actual warmed async CUDA report and SQLite exported; overlap and allocation analysis still required.',flush=True)
