"""Quiet same-build GPT fixed/segmented/cache/prefix ablations after live checks."""
from pathlib import Path
import argparse,hashlib,json,os,psutil,subprocess,sys,time
root=Path.cwd();p=argparse.ArgumentParser();p.add_argument('--validation-pid',type=int,required=True);p.add_argument('--check-pid',type=int,required=True);p.add_argument('--prefill-build-pid',type=int,required=True);a=p.parse_args()
for pid,expected in [(a.validation_pid,'target/gpt-segmented/final_validation.py'),(a.check_pid,'target/performance-cache/check_next.py'),(a.prefill_build_pid,'target/performance-cache/build_prefill.py')]:
    try:previous=psutil.Process(pid)
    except psutil.NoSuchProcess:continue
    assert expected in[s.replace('\\','/')for s in previous.cmdline()],previous.cmdline()
    began=previous.create_time()
    print('Waiting for',expected,'with no competing benchmark.',flush=True)
    while previous.is_running()and previous.create_time()==began:time.sleep(15)
log=(root/'target/gpt-segmented-final-validation.log').read_text(encoding='utf-8-sig')
assert 'Final output-readiness build passed correctness and live checks.'in log,log[-1600:]
manifest=json.loads((root/'target/gpt-segmented/live/manifest.json').read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
binary=root/'target/release/rbitnet.exe';library=root/'target/gpt-segmented/cuda/rbitnet_cuda_quant64.dll'
def unchanged():
    assert sha(binary)==manifest['cli_sha256']and sha(library)==manifest['dll_sha256']
    assert all(sha(root/path)==expected for path,expected in manifest['source_sha256'].items()),'benchmark source changed'
env={k:v for k,v in os.environ.items()if not k.startswith('RBITNET_')};env['RAYON_NUM_THREADS']='16'
for cache in [0,8192]:
    unchanged();folder=root/f'target/gpt-segmented/ablation-cache{cache}-cap12288'
    print('Starting quiet GPT ablation cache',cache,'cap 12288, three cycles; no builds or other inference.',flush=True)
    args=[sys.executable,'scripts/benchmark_cache_stack.py','--config','docs/benchmarks/2026-10-03-parity-round2/manifest.json',
        '--model','gpt-oss-20b','--binary',str(binary),'--library',str(library),'--output-dir',str(folder),
        '--gpt-full','--gpt-segmented','--split-kv','--moe-cache',str(cache),'--device-mib','12288','--cycles','3','--port','18105']
    with(root/f'target/gpt-segmented-ablation-cache{cache}.log').open('w',encoding='utf-8')as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
    print('Completed GPT ablation cache',cache,'exit',r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
    unchanged();(folder/'validation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Both GPT same-build ablations completed with unchanged source/CLI/DLL.',flush=True)
