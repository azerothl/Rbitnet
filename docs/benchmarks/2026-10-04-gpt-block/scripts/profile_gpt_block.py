"""Actual multi-token CUDA launch evidence, outside throughput measurements."""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3,subprocess,time,uuid
root=Path.cwd();base=root/'target/performance-cache';gate=base/'followups-chain.log';deadline=time.monotonic()+7200
while not gate.exists()or 'Actual context network and native F32 page quiet/live experiments passed;'not in gate.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'paging performance not complete; no competing build or profiling started'
 time.sleep(15)
proof=root/'target/gpt-block/profile';proof.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_INCREMENTAL='0')
def run(name,command,extra=None):
 print('Actual GPT block trace:',name,flush=True)
 with(proof/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
 s=(proof/(name+'.log')).read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,s[-3000:]);return s
workspace=run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
import re
assert sum(int(n)for n in re.findall(r'test result: ok\. (\d+) passed;',workspace))==271
run('clippy',['cargo','clippy','--workspace','--all-targets'])
run('profile-build',['cargo','test','--release','-p','bitnet-core','--test','gpt_cuda_profile','--no-run'])
binary=max((root/'target/release/deps').glob('gpt_cuda_profile-*.exe'),key=lambda p:p.stat().st_mtime)
shutil.copy2(binary,proof/'gpt-cuda-profile.exe');binary=proof/'gpt-cuda-profile.exe'
nonce=str(uuid.uuid4());child=proof/('completed-'+nonce+'.json');report=proof/'warmed-gpt-block'
gpu={'RBITNET_GPT_BLOCK_PROFILE':'1','RBITNET_CUDA_QUANT_LIB':str(root/'target/gpt-block/cuda/rbitnet_cuda_quant64.dll'),
 'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256',
 'RBITNET_GPT_BLOCK_GGUF':'D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf',
 'RBITNET_GPT_BLOCK_TOKENIZER':str(root/'target/engine-benchmark/tokenizers/gpt-oss-20b/tokenizer.json'),
 'RBITNET_CUDA_PROFILE_RUNTIME':'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64/cudart64_13.dll',
 'RBITNET_CUDA_PROFILE_PROOF':str(child),'RBITNET_CUDA_PROFILE_NONCE':nonce}
nsys='C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe'
run('nsight',[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-memory-usage=true','--kill=false','--wait=primary','--show-output=true','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output='+str(report),str(binary),'optional_gpt_block_warmed_cuda_profile','--nocapture','--test-threads=1'],gpu)
completed=json.loads(child.read_text(encoding='utf-8'));assert completed['nonce']==nonce and completed['same_output']and completed['completed_prefill_blocks']>1 and completed['profiler_stop_succeeded']
run('export',[nsys,'export','--type=sqlite','--force-overwrite=true','--output='+str(report)+'.sqlite',str(report)+'.nsys-rep'])
db=sqlite3.connect(str(report)+'.sqlite')
rows=db.execute('select s.value,k.gridX,k.gridY,k.gridZ,k.blockX,count(*),sum(k.end-k.start)from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value,k.gridX,k.gridY,k.gridZ,k.blockX').fetchall()
ordered=[x for x in rows if 'ordered_warp_gemm'in x[0]and x[2]>1]
grouped=[x for x in rows if 'moe_group_ordered_matrix'in x[0]]
assert ordered and grouped,(len(rows),ordered,grouped)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
analysis={'completed_child':completed,'kernel_events':db.execute('select count(*)from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0],
 'ordered_multi_token_projection_launches':ordered,'grouped_expert_launches':grouped,
 'all_kernel_grid_counts':rows,'hashes':{p.name:sha(p)for p in [binary,Path(str(report)+'.sqlite'),Path(str(report)+'.nsys-rep'),root/'crates/bitnet-core/tests/gpt_cuda_profile.rs']},
 'limits':['Warmed profiled generation; trace timing is not throughput.','gridY>1 shows multiple tokens in one projection kernel; grouped kernels process the joint routed-token worklist.']}
(proof/'trace-analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
print('Actual GPT block warmed trace proves multi-token projections and grouped FFNs; profiler child completed exactly.',flush=True)
