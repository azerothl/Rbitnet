from pathlib import Path
import hashlib,json,os,subprocess,sys
import psutil
root=Path.cwd();base=root/'target/performance-cache';out=base/'glm-ordered-norm-proof'
manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));assert manifest['status']=='passed'
assert json.loads((out/'least-stale-check.json').read_text(encoding='utf-8'))['status']=='passed'
destination=out/'streaming';assert not destination.exists()
assert not any(p.name().lower()=='rbitnet.exe' for p in psutil.process_iter())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=base/'nucleus-combined-proof/rbitnet.exe';library=out/'cuda/rbitnet_cuda_quant64.dll';harness=root/'scripts/validate_cache_streaming.py'
assert sha(binary)==manifest['binary_reuse']['sha256'] and sha(library)==manifest['library_sha256']
command=[sys.executable,'-B',str(harness),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--binary',str(binary),'--library',str(library),'--mla-full','--split-kv','--moe-cache','8192','--device-mib','12288','--port','18138','--output-dir',str(destination)]
env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16')
path=out/'streaming.log'
with path.open('w',encoding='utf-8') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
receipt=dict(command=command,returncode=result.returncode,checker_sha256=sha(Path(__file__)),harness_sha256=sha(harness),log_sha256=sha(path),status='failed')
def save(): (out/'streaming-check.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
save();assert result.returncode==0,path.read_text(encoding='utf-8')[-6000:]
report=json.loads((destination/'results.json').read_text(encoding='utf-8'))
assert len(report['cases'])==5
assert all(c.get('equal',True) for c in report['cases'])
assert report['cases'][3]['kind']=='explicit_stop_http_sse' and report['cases'][3]['done']
receipt.update(status='passed',cases=5,limits=['Prefix cache enabled for correctness checks only; not used in quiet performance measurements.','Four simultaneous requests are serialized by the model runtime; no GLM continuous-batching claim.']);save()
print('GLM_STAGED_NORM_STREAMING_DONE cases=5')
