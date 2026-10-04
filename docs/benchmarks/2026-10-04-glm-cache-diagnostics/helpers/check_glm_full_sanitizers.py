"""Expanded historical GLM diagnosis; immutable binary and library, serial GPU ownership."""
from pathlib import Path
import hashlib,json,os,subprocess,time
import psutil

root=Path.cwd()
out=root/'target/performance-cache/glm-full-sanitizers-proof'
assert not out.exists(), 'Preserve previous observations'
assert not any(p.name().lower() in ('rbitnet.exe','rbitnet-server.exe','rbitnet-runner.exe') for p in psutil.process_iter()), 'Inference owns the GPU'
out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
production_path=root/'target/async-expert-cache/production-proof/manifest.json'
production=json.loads(production_path.read_text(encoding='utf-8'))
binary=root/'target/async-expert-cache/production-proof/bitnet-core-async.exe'
library=root/'target/paged-kv/cuda/rbitnet_cuda_quant64.dll'
assert sha(binary)==production['frozen_test_binary_sha256']
assert sha(library)==production['library_sha256']
sanitizer=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/compute-sanitizer.bat')
fixture='quantized_full_mla_matches_f64_fixed_dynamic_cpu_fallback_graphs_reset_and_prefix'
env={k:v for k,v in os.environ.items() if not k.startswith('RBITNET_')}
env.update(RBITNET_CUDA_QUANT_SMOKE='1',RBITNET_CUDA_QUANT_LIB=str(library),PYTHONIOENCODING='utf-8',RAYON_NUM_THREADS='16')
manifest=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),binary_sha256=sha(binary),library_sha256=sha(library),production_manifest_sha256=sha(production_path),production=production,fixture=fixture,commands=[],logs={},limits=['Historical frozen Native library and test binary bound to the original failed HTTP capture; no fresh build claimed.','Synthetic full MLA fixture covers fixed/dynamic expert pointers, graph/eager modes, reset, prefix and CPU FFN fallback.','Sanitizer instrumentation changes scheduling; passing cannot establish a fix of the retained full-model quiet HTTP divergence.','No full 30B model under Compute Sanitizer; no performance measurement.'])
def save():
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
save()
for tool in ['memcheck','racecheck','initcheck','synccheck']:
    command=[str(sanitizer),'--tool',tool,'--error-exitcode','99',str(binary),fixture,'--nocapture','--test-threads=1']
    print('Full MLA sanitizer:',tool,flush=True)
    manifest['commands'].append(command)
    save()
    path=out/(tool+'.log')
    with path.open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    output=path.read_text(encoding='utf-8')
    manifest['logs'][path.name]=dict(sha256=sha(path),returncode=result.returncode)
    save()
    assert result.returncode==0,output[-8000:]
    assert '1 passed; 0 failed' in output and 'running 0 tests' not in output,output[-8000:]
    marker='RACECHECK SUMMARY: 0 hazards' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
    assert marker in output,output[-8000:]
manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
manifest['status']='passed'
save()
print('All four full MLA sanitizer modes passed.',flush=True)
