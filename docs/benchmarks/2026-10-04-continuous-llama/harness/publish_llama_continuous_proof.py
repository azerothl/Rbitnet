"""Publish source-bound raw continuous Llama evidence without binaries/models."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,statistics
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
root=Path.cwd();base=root/'target/performance-cache';workspace=args.workspace.resolve()
preparation=json.loads((base/'llama-continuous-delivery-preparation.json').read_text())
manifest=json.loads((base/'llama-continuous-proof/manifest.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for name,digest in(preparation['proven_rust_source_sha256']|preparation['proven_native_source_sha256']).items():assert sha(workspace/name)==digest,name
live=json.loads((base/'llama-continuous-proof/live/results.json').read_text(encoding='utf-8'))
assert live['complete']and len(live['cases'])==66 and len(live['waves'])==21
assert len(manifest['actual_driver_cases'])==24 and len(manifest['thread_cases'])==2
dest=workspace/'docs/benchmarks/2026-10-04-continuous-llama';dest.mkdir(parents=True,exist_ok=False)
index={};objects=[]
for proof_name in ['llama-continuous-proof','llama-batch-proof']:
    proof=base/proof_name
    for path in sorted(proof.rglob('*')):
        if not path.is_file()or path.suffix not in ['.json','.log','.txt','.py','.sql']:continue
        relative=Path('raw')/proof_name/path.relative_to(proof)
        target=dest/(relative.as_posix()+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
        raw=path.read_bytes();target.write_bytes(gzip.compress(raw,mtime=0))
        assert gzip.decompress(target.read_bytes())==raw
        objects.append(dict(source=str(path.relative_to(root)),saved=target.relative_to(dest).as_posix(),source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw)))
for name in ['llama_continuous_live.py','check_llama_continuous.py','prepare_llama_continuous.py','prepare_llama_continuous_delivery.py','publish_llama_continuous_proof.py']:
    source=base/name;target=dest/'harness'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
summary=[]
reference=statistics.median(row['aggregate_tokens_per_second']for row in live['waves']if row['config']=='reference'and not row['warmup'])
for config in sorted({row['config']for row in live['waves']}):
    rows=[row for row in live['waves']if row['config']==config and not row['warmup']]
    tps=statistics.median(row['aggregate_tokens_per_second']for row in rows)
    summary.append(dict(config=config,measured_cycles=len(rows),aggregate_tokens_per_second=tps,
                        relative_to_reference_percent=(tps/reference-1)*100,
                        wave_wall_ms=statistics.median(row['wall_ms']for row in rows),
                        note='Eight actual HTTP clients; CPU sampling, prefill and delayed arrivals included. Not per-request decode tokens/s.'))
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(dest/'receipt.json').write_text(json.dumps(dict(preparation=preparation,raw_objects=objects,
    private_proof_source=str((base/'llama-continuous-proof/manifest.json').relative_to(root)),
    fresh_public_build=False,limits=manifest['limits']),indent=2)+'\n')
for path in dest.rglob('*'):
    if path.is_file():index[path.relative_to(dest).as_posix()]=sha(path)
(dest/'SHA256.json').write_text(json.dumps(index,indent=2)+'\n')
print('Continuous proof published:',len(objects),'exact raw gzip objects; 24 driver/2 threads/66 HTTP/21 waves; public rebuild still pending.')
