"""Matched previous/current binary memory and performance, after policy runs."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path.cwd();base=root/'target/moe-placement';chain=root/'target/moe-placement-benchmark-chain.log'
deadline=time.monotonic()+7200
while 'All ten matched MoE policy/cache ablations passed.'not in chain.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'policy benchmark did not complete; refusing concurrent measurements'
 time.sleep(15)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=root/'target/gpt-segmented/ablation-cache0-cap12288/rbitnet.exe'
old_manifest=json.loads((root/'target/gpt-segmented/live/manifest.json').read_text())
assert sha(old)==old_manifest['cli_sha256']=='016ed0f518a0125b3441ab2945f3ece0f2ce99be95e0c9cc59788801c77e6f45'
lib=root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll';assert sha(lib)==old_manifest['dll_sha256']
new_manifest=json.loads((base/'validation-manifest.json').read_text())
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(RAYON_NUM_THREADS='16',PYTHONIOENCODING='utf-8')
for model in ['gpt-oss-20b','glm47-flash']:
 assert all(sha(root/p)==h for p,h in new_manifest['source_sha256'].items())
 folder=base/'ablation'/f'{model}-previous-cache0'
 command=[sys.executable,'scripts/benchmark_cache_stack.py','--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
  '--model',model,'--binary',str(old),'--library',str(lib),'--output-dir',str(folder),'--split-kv','--moe-cache','0',
  '--moe-execution','cache','--device-mib','12288','--cycles','3','--notes','24','--max-tokens','128','--port','18131']
 if model=='gpt-oss-20b':command+=['--gpt-full','--gpt-segmented','--modes','full-split']
 else:command+=['--mla-full','--modes','mla-split']
 print('Starting previous/current matched binary comparison',model,flush=True)
 with(base/(model+'-previous-benchmark.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(model,r.returncode)
 data=json.loads((folder/'results.json').read_text());new=json.loads((base/'ablation'/f'{model}-cache-cache0/results.json').read_text())
 texts=lambda d:[row['response']['choices'][0]['message']['content']for row in d['rows']]+[row['text']for row in d['sse']]
 assert texts(data)==texts(new)
 (folder/'validation-manifest.json').write_text(json.dumps(old_manifest,indent=2)+'\n',encoding='utf-8')
 assert sha(old)==old_manifest['cli_sha256']and sha(lib)==old_manifest['dll_sha256']
 assert all(sha(root/p)==h for p,h in new_manifest['source_sha256'].items())
 print('Completed previous/current matched comparison',model,'with identical outputs.',flush=True)
print('All twelve policy and previous/current matched ablations completed.',flush=True)
