from pathlib import Path
import hashlib,json,os,subprocess,sys
import psutil
root=Path.cwd();base=root/'target/performance-cache';out=base/'glm-ordered-norm-proof'
manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));assert manifest['status']=='passed'
destination=out/'least-stale';assert not destination.exists()
assert not any(p.name().lower()=='rbitnet.exe' for p in psutil.process_iter())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=base/'nucleus-combined-proof/rbitnet.exe';library=out/'cuda/rbitnet_cuda_quant64.dll';harness=base/'glm-policy-current-quiet-proof/quiet_harness.py'
assert sha(binary)==manifest['binary_reuse']['sha256'] and sha(library)==manifest['library_sha256'] and sha(harness)==manifest['harness_sha256']
command=[sys.executable,'-B',str(harness),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--backend','gpu','--cycles','4','--notes','4','--max-tokens','128','--device-mib','12288','--port','18138','--binary',str(binary),'--library',str(library),'--policies','--model','glm47-flash','--mla-full','--split-kv','--moe-cache','8192','--trace-dir',str(out/'disabled-traces'),'--no-trace','--modes','least-stale','--output-dir',str(destination)]
env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16')
path=out/'least-stale.log'
with path.open('w',encoding='utf-8') as log: result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
receipt=dict(command=command,returncode=result.returncode,checker_sha256=sha(Path(__file__)),log_sha256=sha(path),status='failed')
def save(): (out/'least-stale-check.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
save();assert result.returncode==0,path.read_text(encoding='utf-8')[-6000:]
rows=json.loads((destination/'results.json').read_text(encoding='utf-8'))['rows']
reference=json.loads((out/'baseline/results.json').read_text(encoding='utf-8'))['rows']
def exact(rows): return [(r['cycle'],r['prompt'],r['request'],r['response']['choices'],r['response']['usage']) for r in rows]
assert len(rows)==12 and all(r['matches_baseline'] for r in rows) and exact(rows)==exact(reference)
receipt.update(status='passed',exact_records=12,limitation='Passing this additional quiet Least-Stale repetition does not explain or erase the historical failure.');save()
print('GLM_STAGED_NORM_LEAST_STALE_DONE exact_records=12')
