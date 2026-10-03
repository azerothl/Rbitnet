from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
root=Path.cwd();base=root/'target/performance-cache';gate=base/'async-profile-chain.log';deadline=time.monotonic()+18000
while 'Actual warmed async CUDA report and SQLite exported;' not in gate.read_text(encoding='utf-8-sig'):
 assert time.monotonic()<deadline,'profiling did not finish; no competing compilation/inference started'
 time.sleep(15)
for name in ['prepare_context_capacity.py','prepare_sentencepiece.py','prepare_tokenizer_sharing.py']:subprocess.run([sys.executable,str(base/name)],cwd=root,check=True)
out=base/'check-context'
for name in ['Cargo.toml','Cargo.lock']:shutil.copy2(root/name,out/name)
shutil.copytree(root/'crates',out/'crates',dirs_exist_ok=True,ignore=shutil.ignore_patterns('target','__pycache__'));shutil.copytree(root/'recipes',out/'recipes',dirs_exist_ok=True)
for source in ['context-integration','sentencepiece-integration','tokenizer-sharing']:
 for p in (base/source).rglob('*'):
  if p.is_file():q=out/p.relative_to(base/source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env.update(PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0',RBITNET_CUDA='0',RAYON_NUM_THREADS='16',CARGO_TARGET_DIR=str(base/'check-build'))
proof=base/'context-real-proof';proof.mkdir(exist_ok=True)
def run(name,cmd,extra=None):
 print('Actual trained SentencePiece/GGUF:',name,flush=True)
 with (proof/(name+'.log')).open('w',encoding='utf-8')as log:r=subprocess.run(cmd,cwd=out,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
 s=(proof/(name+'.log')).read_text(encoding='utf-8');assert r.returncode==0,(name,r.returncode,s[-5000:]);return s
run('check',['cargo','check','--workspace','--all-targets']);run('frozen-tokenizer-context',['cargo','test','-p','bitnet-server','--test','context_capacity','--','--nocapture','--test-threads=1']);run('clippy',['cargo','clippy','--workspace','--all-targets'])
env['CARGO_TARGET_DIR']=str(base/'check-release')
gpu={'RBITNET_SENTENCEPIECE_REAL_TEST':'1','RBITNET_SENTENCEPIECE_REAL_GGUF':'D:/Rbitnet-benchmark-models/mistral7b-sp/mistral-7b-instruct-v0.1.Q4_K_M.gguf','RBITNET_SENTENCEPIECE_TEST_MODEL':'C:/Users/azero/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/sentencepiece-0.11.3/testdata/toy.model',
 'RBITNET_CUDA_QUANT_LIB':str(root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'),'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_SPLIT_KV':'1'}
s=run('actual-sp',['cargo','test','--release','-p','bitnet-core','--lib','sentencepiece_codec::tests','--','--nocapture','--test-threads=1'],gpu)
assert 'ACTUAL_SP_MISTRAL backend=Cpu:'in s and 'ACTUAL_SP_MISTRAL backend=Cuda:'in s and '8 passed; 0 failed'in s
run('cli',['cargo','build','--release','-p','rbitnet-cli']);shutil.copy2(base/'check-release/release/rbitnet.exe',proof/'rbitnet.exe')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m={'source_sha256':{p.relative_to(out).as_posix():sha(p)for p in(out/'crates').rglob('*.rs')},'binary_sha256':sha(proof/'rbitnet.exe'),'library_sha256':sha(Path(gpu['RBITNET_CUDA_QUANT_LIB'])),'model_inputs':json.loads((base/'model-inputs-manifest.json').read_text())}
(proof/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print('Actual Mistral GGUF SentencePiece/HF CPU/CUDA generation and trained normalization checks passed; network live and serving capacity follow-up pending.',flush=True)
