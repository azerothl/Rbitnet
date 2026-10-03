"""Validate adopted context/SentencePiece/shared tokenizers between experiments."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'followups-chain.log';deadline=time.monotonic()+28800
while not gate.exists()or 'Passed serial follow-up: async-benchmark-gpt-oss-20b-cache512'not in gate.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'first quiet async ablation did not finish; no competing work started'
 time.sleep(15)
proof=root/'target/context-tokenizer/production-proof';proof.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16')
def run(name,command,extra=None,marker=None):
 print('Adopted context/tokenizer validation:',name,flush=True)
 with(proof/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(command,cwd=root,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
 s=(proof/(name+'.log')).read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,s[-6000:])
 if marker:assert marker in s and 'running 0 tests'not in s,(name,s[-3000:])
run('workspace',['cargo','test','--workspace','--','--test-threads=1'])
run('clippy',['cargo','clippy','--workspace','--all-targets'])
gpu={'RBITNET_SENTENCEPIECE_REAL_TEST':'1','RBITNET_SENTENCEPIECE_REAL_GGUF':'D:/Rbitnet-benchmark-models/mistral7b-sp/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
 'RBITNET_SENTENCEPIECE_TEST_MODEL':'C:/Users/azero/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/sentencepiece-0.11.3/testdata/toy.model',
 'RBITNET_CUDA_QUANT_LIB':str(root/'target/gpt-block/cuda/rbitnet_cuda_quant64.dll'),'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_SPLIT_KV':'1'}
run('actual-sp',['cargo','test','--release','-p','bitnet-core','--lib','sentencepiece_codec::tests','--','--nocapture','--test-threads=1'],gpu,'ACTUAL_SP_MISTRAL backend=Cuda:')
run('release',['cargo','build','--release','-p','rbitnet-cli'])
shutil.copy2(root/'target/release/rbitnet.exe',proof/'rbitnet.exe');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=json.loads((base/'context-production-paths.json').read_text(encoding='utf-8'))
m={'source_sha256':{p:sha(root/p)for p in paths},'binary_sha256':sha(proof/'rbitnet.exe'),'library_sha256':sha(Path(gpu['RBITNET_CUDA_QUANT_LIB']))}
(proof/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
run('network',[sys.executable,str(base/'run_context_live.py')],{'RBITNET_CONTEXT_PROOF_DIR':str(proof),'RBITNET_CONTEXT_LIVE_DIR':str(root/'target/context-tokenizer/live')})
print('Adopted context capacity, SentencePiece, immutable tokenizers, actual Mistral and HTTP validation passed.',flush=True)
