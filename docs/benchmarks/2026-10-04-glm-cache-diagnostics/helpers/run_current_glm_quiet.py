"""Quiet HTTP reproduction on source-equivalent current main, one GPU owner."""
from pathlib import Path
import subprocess,json,hashlib,sys,os,time
import psutil
b=Path(__file__).resolve().parent;root=b.parent.parent
out=b/'glm-policy-current-quiet-proof';assert not out.exists();out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not [p.info for p in psutil.process_iter(['name','pid']) if (p.info['name'] or '').lower() in ['rbitnet.exe','rbitnet-server.exe','rbitnet-runner.exe']], 'another inference owner is active'
proof=json.loads((b/'nucleus-combined-proof/manifest.json').read_text())
main=subprocess.check_output(['git','rev-parse','origin/main'],cwd=root,text=True).strip()
for name in proof['source_sha256']:
    a=subprocess.check_output(['git','show',main+':'+name],cwd=root)
    z=subprocess.check_output(['git','show',proof['head']+':'+name],cwd=root)
    assert a.replace(b'\r\n',b'\n')==z.replace(b'\r\n',b'\n'),name
binary=b/'nucleus-combined-proof/rbitnet.exe';library=b/'performance-stack-proof/cuda/rbitnet_cuda_quant64.dll'
assert sha(binary)==proof['binary_sha256'] and sha(library)==proof['library_sha256']
source=b/'glm-policy-focused-harness.py';harness=out/'quiet_harness.py'
code=source.read_text(encoding='utf-8')
old="                    if mode == reference_mode: baseline[cycle, index] = text"
new="""                    if mode == reference_mode:
                        baseline[cycle, index] = text
                    # Compare both policies and repeated cycles to the same
                    # first cold LRU response, never a potentially drifting cycle.
                    stable_reference = baseline[0, index]"""
assert code.count(old)==1;code=code.replace(old,new)
code=code.replace('matches_baseline=text == baseline[cycle, index]','matches_baseline=text == stable_reference')
harness.write_text(code,encoding='utf-8',newline='\n')
command=[sys.executable,'-B',str(harness),'--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json','--backend','gpu','--cycles','4','--notes','4','--max-tokens','128','--device-mib','12288','--port','18138','--binary',str(binary),'--library',str(library),'--policies','--model','glm47-flash','--mla-full','--split-kv','--moe-cache','8192','--trace-dir',str(out/'disabled-traces'),'--no-trace','--modes','lru,least-stale','--output-dir',str(out/'http')]
env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='16')
binding=dict(main=main,compiled_head=proof['head'],binary_sha256=sha(binary),library_sha256=sha(library),harness_sha256=sha(harness),original_harness_sha256=sha(source),command=command,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),limitations=['No full-logit readback and no router trace; normal HTTP requests and metrics only.','A passing repeat does not explain or erase the historical failure.'])
(out/'manifest.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
with (out/'run.log').open('w',encoding='utf-8') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
binding.update(returncode=result.returncode,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
if (out/'http/results.json').exists():
    rows=json.loads((out/'http/results.json').read_text())['rows']
    binding.update(observations=len(rows),mismatches=[dict(mode=r['mode'],cycle=r['cycle'],prompt=r['prompt']) for r in rows if not r['matches_baseline']])
(out/'manifest.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
print(json.dumps(binding),flush=True)
assert result.returncode==0,'quiet reproduction failed; raw failure preserved'
