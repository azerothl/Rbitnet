"""Adopt exactly the successful private source bytes onto finish-reasons base."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=workspace,text=True).strip(),'delivery checkout must be clean'
manifest=json.loads((base/'llama-continuous-proof/manifest.json').read_text(encoding='utf-8'))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()=='86731958cadb7c0f6f48833cd6a47ff7fc83012d'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
changes=[]
for name,digest in manifest['source_sha256'].items():
    source=base/'check-llama-continuous'/name;assert sha(source)==digest,name
    target=workspace/name
    if not target.exists()or sha(target)!=digest:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);changes.append(name)
native={}
for name,digest in manifest['native_batch_proof']['native_source_sha256'].items():
    source=base/'llama-batch-native'/name;assert sha(source)==digest,name
    if name=='build.ps1':continue
    destination='native/cuda_quant/'+('include/'if name.endswith('.h')else'src/')+name
    target=workspace/destination;native[destination]=digest
    if not target.exists()or sha(target)!=digest:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);changes.append(destination)
for name,digest in(manifest['source_sha256']|native).items():assert sha(workspace/name)==digest,name
receipt=dict(workspace=str(workspace),base_head='86731958cadb7c0f6f48833cd6a47ff7fc83012d',modified_paths=changes,
             proven_rust_source_sha256=manifest['source_sha256'],proven_native_source_sha256=native,
             private_binary_sha256=manifest['binary_sha256'],private_library_sha256=manifest['library_sha256'],
             limit='Source bytes match the successful private build. A fresh build from this public checkout is not yet performed.')
(base/'llama-continuous-delivery-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'modified_paths':changes,'exact_rust_files':len(manifest['source_sha256']),'exact_native_files':len(native)},indent=2))
