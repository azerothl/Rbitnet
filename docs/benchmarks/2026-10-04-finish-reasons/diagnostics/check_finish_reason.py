"""Serialise compilation and network proof after the actual cache policy study."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'policy-measurements-chain.log';deadline=time.monotonic()+43200
marker='Expert policy study captured replay and quiet/network suites; prior GLM Least-Stale divergence remains under investigation.'
while not gate.exists()or marker not in gate.read_text(encoding='utf-8-sig'):
    assert time.monotonic()<deadline,'preceding serial native proof not complete'
    time.sleep(15)
out=base/'finish-reason-proof';out.mkdir(exist_ok=True);crate=base/'check-finish-reason';crate.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_TARGET_DIR=str(base/'check-build'))
def run(name,command,cwd=root,required=None):
    print('Private actual finish-reason:',name,flush=True);path=out/(name+'.log')
    with path.open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    text=path.read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,text[-6000:])
    if required:assert required in text and 'running 0 tests'not in text,(name,text[-3000:])
    return text
receipt=base/'policy-delivery-verification.json'
if receipt.exists():
    published=json.loads(receipt.read_text(encoding='utf-8'))
    delivered=Path('C:/Users/azero/.codex/worktrees/moe-replay-delivery/Rbitnet/docs/benchmarks/2026-10-04-expert-policies/manifest.json')
    assert published['complete'] and all(row['returncode']==0 for row in published['commands'])
    assert hashlib.sha256(delivered.read_bytes()).hexdigest()==published['manifest_sha256']
    print('Previously verified and published policy evidence retained; no hardware study rerun.',flush=True)
else:
    run('policy-delivery',[sys.executable,'-B',str(base/'deliver_policy_evidence.py')],required='POLICY_DELIVERY_READY exact raw bytes and 12 replay counters verified; 10 Python regressions passed; commit/push pending.')
run('prepare',[sys.executable,str(base/'prepare_finish_reason.py')])
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(root/name,crate/name)
for name in ['crates','recipes','.cargo','tests']:
    if(root/name).exists():shutil.copytree(root/name,crate/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'))
for p in(base/'finish-reason-integration').rglob('*.rs'):
    dest=crate/p.relative_to(base/'finish-reason-integration');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
run('check',['cargo','check','--workspace','--all-targets'],crate)
run('clippy',['cargo','clippy','--workspace','--all-targets'],crate)
run('termination-decision',['cargo','test','-p','bitnet-core','--lib','finish_tests','--','--nocapture','--test-threads=1'],crate,'2 passed; 0 failed')
run('queue-eos',['cargo','test','-p','bitnet-core','--test','scheduler_speculative','--','--nocapture','--test-threads=1'],crate)
run('tiny-actual-http',['cargo','test','-p','bitnet-server','--test','finish_reason','--','--nocapture','--test-threads=1'],crate,'FINISH_REASON_HTTP_DONE cases=42')
run('workspace',['cargo','test','--workspace','--','--test-threads=1'],crate)
env['CARGO_TARGET_DIR']=str(base/'check-release')
run('release',['cargo','build','-p','rbitnet-cli','--release'],crate)
binary=out/'rbitnet.exe';shutil.copy2(base/'check-release/release/rbitnet.exe',binary)
library=base/'finish-reason-baseline-native.dll';assert library.is_file()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
production=json.loads((root/'target/async-expert-cache/production-proof/manifest.json').read_text(encoding='utf-8'));assert sha(library)==production['library_sha256']
run('actual-four-model-network',[sys.executable,str(base/'finish_reason_live.py'),'--binary',str(binary),'--library',str(library),'--output',str(out/'live')],required='FINISH_REASON_NETWORK_DONE actual_models=4')
(out/'manifest.json').write_text(json.dumps(dict(binary_sha256=sha(binary),library_sha256=sha(library),harness_sha256=sha(base/'finish_reason_live.py'),preparer_sha256=sha(base/'prepare_finish_reason.py'),logs={p.name:sha(p)for p in out.glob('*.log')},sources={p.relative_to(crate).as_posix():sha(p)for p in(crate/'crates').rglob('*.rs')},
    limits=['Private source integration; production adoption pending.','Actual EOS observation, token budget and explicit stop strings remain distinct, including zero visible tokens.','Unknown legacy/toy timings produce null instead of an invented stop reason.','Static stubs finish by definition and retain stop/end_turn.','Queue zero-output EOS cannot retry indefinitely; no claim of genuine GPU serving batching here.','Same prompt repeated exactly at EOS output length and one token beyond must return length then stop with equal visible text.']),indent=2)+'\n',encoding='utf-8')
print('Private finish reasons passed EOS/budget, queue no-progress, actual tiny OpenAI/Anthropic and four-model loopback JSON/SSE boundaries; production adoption pending.',flush=True)
