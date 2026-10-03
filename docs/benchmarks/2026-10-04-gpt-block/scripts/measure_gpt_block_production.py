from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'block-production-chain.log';deadline=time.monotonic()+28800
while not gate.exists()or 'Adopted GPT block/fusion native, workspace, Clippy, release, numerical and real GGUF validation passed.'not in gate.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'production correctness did not finish; refusing competing inference'
 time.sleep(15)
proof=root/'target/gpt-block/production-proof';m=json.loads((proof/'manifest.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();lib=root/'target/gpt-block/cuda/rbitnet_cuda_quant64.dll'
assert sha(proof/'rbitnet.exe')==m['binary_sha256']and sha(lib)==m['library_sha256']
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',RAYON_NUM_THREADS='16')
common=['--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(proof/'rbitnet.exe'),'--library',str(lib),'--gpt-full','--device-mib','12288','--port','18135']
def run(name,command):
 print('Adopted GPT quiet/live:',name,flush=True)
 with(proof/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
out=root/'target/gpt-block/ablation'
run('quiet-ablation',[sys.executable,'scripts/benchmark_cache_stack.py',*common,'--model','gpt-oss-20b','--backend','gpu','--gpt-prefill','--modes','serial,block16,block32,block16-prefix','--cycles','3','--notes','24','--max-tokens','128','--output-dir',str(out)])
r=json.loads((out/'results.json').read_text(encoding='utf-8'));assert len(r['rows'])==36 and all(x['matches_baseline']for x in r['rows'])and len(r['sse'])==12
run('network',[sys.executable,'scripts/validate_cache_streaming.py',*common,'--gpt-prefill','16','--split-kv','--output-dir',str(root/'target/gpt-block/live')])
print('Adopted GPT block quiet ablations and real network suites passed.',flush=True)
