"""Quiet same-build MoE cost ablation; launched after live checks and prototypes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path.cwd();base=root/'target/moe-placement'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((base/'validation-manifest.json').read_text())
binary=root/'target/release/rbitnet.exe';library=root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'
def unchanged():
 assert sha(binary)==manifest['cli_sha256'] and sha(library)==manifest['dll_sha256']
 assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items())
assert 'All four MoE live policy/metrics/lifecycle/streaming suites passed.'in(root/'target/moe-placement-live-chain.log').read_text(encoding='utf-8-sig')
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(RAYON_NUM_THREADS='16',PYTHONIOENCODING='utf-8')
reference={}
for model in ['gpt-oss-20b','glm47-flash']:
 for policy,cache in [('cache',0),('cpu',0),('adaptive',0),('cache',8192),('adaptive',8192)]:
  unchanged();name=f'{model}-{policy}-cache{cache}';folder=base/'ablation'/name
  command=[sys.executable,'scripts/benchmark_cache_stack.py','--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
   '--model',model,'--binary',str(binary),'--library',str(library),'--output-dir',str(folder),'--split-kv',
   '--moe-cache',str(cache),'--moe-execution',policy,'--device-mib','12288','--cycles','3','--notes','24','--max-tokens','128','--port','18131']
  if model=='gpt-oss-20b':command+=['--gpt-full','--gpt-segmented','--modes','full-split']
  else:command+=['--mla-full','--modes','mla-split']
  print('Starting quiet MoE',name,'three cycles, same inputs; no other build/inference.',flush=True)
  with(base/(name+'-benchmark.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(name,r.returncode)
  unchanged();data=json.loads((folder/'results.json').read_text())
  texts=[row['response']['choices'][0]['message']['content']for row in data['rows']]+[row['text']for row in data['sse']]
  if model not in reference:reference[model]=texts
  assert texts==reference[model],(name,'same-policy benchmark output differs')
  (folder/'validation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
  print('Completed quiet MoE',name,'with identical outputs and unchanged source/CLI/DLL.',flush=True)
print('All ten matched MoE policy/cache ablations passed.',flush=True)
