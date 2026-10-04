"""Twelfth serial owner: real four-model CPU/GPU comparison after combined proof."""
from pathlib import Path
import hashlib,json,os,shutil,socket,subprocess,sys,time
import psutil,requests
from atomic_journal import write_json

root=Path.cwd();base=root/'target/performance-cache';out=base/'native-llama-parity-proof';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checker=Path(__file__);harness=root/'scripts/benchmark_engines.py';harness_sha=sha(harness)
journal=base/'native-llama-parity-experiment.json'
if journal.exists():
    prior=json.loads(journal.read_text(encoding='utf-8'))
    try:
        owner=psutil.Process(prior['pid']);assert owner.pid==os.getpid() or checker.name not in ' '.join(owner.cmdline())
    except psutil.NoSuchProcess:pass
status={'pid':os.getpid(),'status':'waiting','complete':False,'checker_sha256':sha(checker),'harness_sha256':harness_sha,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def save():write_json(journal,status)
save();owned=None;owned_log=None
try:
    deadline=time.monotonic()+43200
    while True:
        state=json.loads((base/'adaptive-nucleus-experiment.json').read_text(encoding='utf-8'))
        assert state['status']!='failed','preceding adaptive-nucleus owner failed; hardware comparison refused'
        if state['complete']:
            assert state['status']=='passed';break
        owner=psutil.Process(state['pid']);assert owner.is_running() and 'check_adaptive_nucleus.py' in ' '.join(owner.cmdline())
        assert time.monotonic()<deadline;time.sleep(15)
    proof_path=base/'performance-stack-proof/manifest.json';proof=json.loads(proof_path.read_text(encoding='utf-8'))
    binary=base/'performance-stack-proof/rbitnet.exe';library=base/'performance-stack-proof/cuda/rbitnet_cuda_quant64.dll'
    assert sha(binary)==proof['binary_sha256']['rbitnet'] and sha(library)==proof['library_sha256'] and sha(harness)==harness_sha
    status.update(status='running',hardware_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    original=json.loads((root/'docs/benchmarks/2026-10-03-parity-round2/manifest.json').read_text(encoding='utf-8'))
    config=dict(original);config['models']=[m for m in original['models'] if m['id']=='llama32-1b'];assert len(config['models'])==1;config.update(rbitnet=str(binary),cuda_quant_library=str(library),context=2048,threads=16,cwd=str(root),port=18138,startup_timeout=600)
    ollama=Path('C:/Users/azero/AppData/Local/Programs/Ollama/ollama.exe');store=Path('D:/Rbitnet-benchmark-models/ollama')
    paths={'rbitnet':binary,'cuda_quant_library':library,'ollama':ollama,'llama_cpu':Path(config['llama_cpu']),'llama_gpu':Path(config['llama_gpu'])}
    versions={}
    for name in ['ollama','llama_cpu','llama_gpu']:
        r=subprocess.run([str(paths[name]),'--version'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        assert r.returncode==0,(name,r.returncode,r.stdout,r.stderr)
        versions[name]=(r.stdout+r.stderr).strip()
    models={}
    for model in config['models']:
        digest=sha(Path(model['gguf']));alias,tag=model['ollama'].split(':',1)
        manifest=store/'manifests/registry.ollama.ai/library'/alias/tag
        meta=json.loads(manifest.read_text(encoding='utf-8'))
        layer=next(l for l in meta['layers'] if l['mediaType']=='application/vnd.ollama.image.model')
        assert layer['digest']=='sha256:'+digest and layer['size']==Path(model['gguf']).stat().st_size,model['id']
        blob=store/'blobs'/layer['digest'].replace(':','-')
        assert sha(blob)==digest,blob
        models[model['id']]={'gguf_sha256':digest,'tokenizer_sha256':sha(Path(model['tokenizer'])),'ollama_manifest_sha256':sha(manifest),'ollama_model_layer':layer}
    # Own a separate local daemon, never unload or stop the user's main Ollama.
    with socket.socket() as probe:probe.bind(('127.0.0.1',11439))
    opts={'OLLAMA_HOST':'127.0.0.1:11439','OLLAMA_MODELS':str(store),'OLLAMA_NUM_PARALLEL':'1','OLLAMA_MAX_LOADED_MODELS':'1','OLLAMA_CONTEXT_LENGTH':'2048','OLLAMA_FLASH_ATTENTION':'1','OLLAMA_KV_CACHE_TYPE':'f16'}
    env={k:v for k,v in os.environ.items() if not k.startswith(('RBITNET_','OLLAMA_'))};env.update(opts,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    owned_log=(out/'ollama-private.log').open('w',encoding='utf-8')
    owned=subprocess.Popen([str(ollama),'serve'],env=env,cwd=root,stdout=owned_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    config.update(ollama_url='http://127.0.0.1:11439',ollama_pid=owned.pid)
    listener_observations=[]
    deadline=time.monotonic()+60
    while True:
        assert owned.poll() is None,('private Ollama exited',owned.returncode)
        connections=psutil.Process(owned.pid).connections(kind='tcp')
        owns_port=any(c.laddr.ip=='127.0.0.1' and c.laddr.port==11439 and not c.raddr and c.status in [psutil.CONN_LISTEN, psutil.CONN_NONE] for c in connections)
        if owns_port:
            listener_observations=[dict(pid=owned.pid,local_ip=c.laddr.ip,local_port=c.laddr.port,remote=list(c.raddr),reported_state=c.status) for c in connections if c.laddr.port==11439 and not c.raddr]
            try:
                r=requests.get(config['ollama_url']+'/api/version',timeout=2);r.raise_for_status();api_version=r.json();break
            except requests.RequestException:pass
        assert time.monotonic()<deadline,'private Ollama startup timed out';time.sleep(.2)
    tags=requests.get(config['ollama_url']+'/api/tags',timeout=30);tags.raise_for_status();known={m['name'] for m in tags.json()['models']}
    assert all(m['ollama'] in known for m in config['models']),known
    environment={'rbitnet_head':proof['head'],'binary_sha256':{k:sha(v) for k,v in paths.items()},'versions':versions,'ollama_api_version':api_version,
        'ollama_private_pid':owned.pid,'ollama_listener_observations':listener_observations,'ollama_private_options':opts,'models':models,'source_proof_sha256':sha(proof_path),
        'context_capacity':2048,'threads':16,'profile':'Native Llama dense versus Native F32 paged KV; host paging disabled; graph and split counters required',
        'kv_dtype':{'rbitnet':'f32','llama.cpp':'f32','ollama':'f16'},'limits':['Different KV dtypes; no strict dtype-identical comparison.','Three short quality probes are not a general model quality evaluation.','Single-client throughput; continuous multi-client benefit is a separate experiment.','GPU memory samples are machine-global; other applications can affect them.','Only this NVIDIA machine is validated.']}
    config['environment']=environment;write_json(out/'environment.json',environment)
    all_rows=[];commands=[]
    for model in config['models']:
        name=model['id']
        for profile in ['dense','paged']:
            backend='gpu'
            status['current_step']=name+'-'+profile;save()
            folder=out/(name+'-'+profile);folder.mkdir(exist_ok=True)
            assert not (folder/'results.json').exists(),'choose a fresh archive before repeating a parity attempt'
            if backend=='gpu':
                fixture=out/(name+'-cpu')/(name+'-prompts.json')
                if fixture.exists():shutil.copy2(fixture,folder/fixture.name)
            common={'RBITNET_MAX_SEQ':'2048','RBITNET_PREFIX_KV':'0','RBITNET_CONTEXT_TIERS':'0','RBITNET_CUDA_KV_FORMAT':'f32','RBITNET_KV_POOL':'0',
                'RBITNET_MOE_CACHE_POLICY':'lru','RBITNET_MOE_PREFETCH':'off','RBITNET_MOE_EXECUTION':'cache','RBITNET_QWEN_SPECULATIVE':'0','RBITNET_CONTINUOUS_BATCHING':'0',
                'RBITNET_CUDA_DEVICE_BUDGET_MB':'12288','RBITNET_CUDA_DEVICE_MARGIN_MB':'256','RBITNET_CUDA_PREFILL_TF32X3':'0',
                'RBITNET_CUDA_QWEN_FULL':'0','RBITNET_REQUIRE_QWEN_FULL':'0','RBITNET_CUDA_GPT_FULL':'0','RBITNET_REQUIRE_GPT_FULL':'0',
                'RBITNET_CUDA_MLA_FULL':'0','RBITNET_REQUIRE_MLA_FULL':'0','RBITNET_CPU_DIRECT_ROWS':'1' if backend=='cpu' and name=='llama32-1b' else '0',
                'RBITNET_MOE_ASYNC':'0','RBITNET_MOE_ARENA':'0','RBITNET_MOE_CACHE_MB':'0'}
            if backend=='gpu':
                common.update(RBITNET_CUDA_PREFILL='1',RBITNET_CUDA_PREFILL_TOKENS='128',RBITNET_CUDA_SPLIT_KV='1')
                if name=='llama32-1b':common.update(RBITNET_LLAMA_PAGED_KV='0',RBITNET_CUDA_KV_PAGE_LIMIT='256' if profile=='paged' else '0',RBITNET_CUDA_RESIDENT_GRAPH='1')
                elif name=='qwen35-2b':common.update(RBITNET_CUDA_QWEN_FULL='1',RBITNET_REQUIRE_QWEN_FULL='1',RBITNET_CUDA_QWEN_PREFILL='1',RBITNET_QWEN_ORDERED_BLOCK_TEST='1')
                else:
                    common.update(RBITNET_MOE_ASYNC='1',RBITNET_MOE_ARENA='1',RBITNET_MOE_CACHE_MB='8192')
                    if name=='gpt-oss-20b':common.update(RBITNET_CUDA_GPT_FULL='1',RBITNET_REQUIRE_GPT_FULL='1',RBITNET_CUDA_GPT_SEGMENTED='1')
                    else:common.update(RBITNET_CUDA_MLA_FULL='1',RBITNET_REQUIRE_MLA_FULL='1')
            cell=dict(config,models=[model],rbitnet_env=common);manifest=folder/'manifest.json';write_json(manifest,cell)
            command=[sys.executable,'-B',str(harness),'--manifest',str(manifest),'--output',str(folder/'results.json'),'--backends',backend,'--repeats','3','--tokens','128','--long-notes','24']
            with(folder/'checker.log').open('w',encoding='utf-8') as log:
                r=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
            assert r.returncode==0,(name,backend,r.returncode)
            captured=json.loads((folder/'results.json').read_text(encoding='utf-8'))
            assert len(captured['rows'])==3,(name,backend,captured.get('fixture_errors'),captured['rows'])
            for row in captured['rows']:
                row['native_profile']=profile
                if row['status']=='ok' and row['engine']=='rbitnet':
                    assert row['metrics'].get('rbitnet_core_cuda_graph_replays_total',0)>0,('required native Llama graph did not run',profile)
                    assert row['metrics'].get('rbitnet_core_gpu_split_attention_queries_total',0)>0,('required split attention did not run',profile)
                    tokens=sum(row['metrics'].get('rbitnet_core_gpu_'+f+'_tokens_total',0) for f in ['qwen_full','gpt_full','mla_full'])
                    if backend=='cpu':assert tokens==0 and row['metrics'].get('rbitnet_core_gpu_gemv_calls_total',0)==0,row
                    elif name!='llama32-1b':assert tokens>0,('required full GPU path did not run',name)
                if row['status']=='ok' and row['engine']=='ollama':
                    loaded=row['loaded_model']['models'];assert len(loaded)==1
                    actual_vram=loaded[0].get('size_vram',0)
                    row['backend_observed']='gpu' if actual_vram>0 else 'cpu'
                    if row['backend_observed']!=backend:row['status']='backend_fallback';row['error']='requested backend differs from observed Ollama VRAM placement'
                all_rows.append(row)
            commands.append({'command':command,'manifest_sha256':sha(manifest),'capture_sha256':sha(folder/'results.json'),'checker_log_sha256':sha(folder/'checker.log')})
            write_json(out/'summary.json',{'environment':environment,'rows':all_rows,'commands':commands,'complete':False})
            assert sha(binary)==proof['binary_sha256']['rbitnet'] and sha(library)==proof['library_sha256'] and sha(harness)==harness_sha
    assert len(all_rows)==6
    write_json(out/'summary.json',{'environment':environment,'rows':all_rows,'commands':commands,'complete':True})
    status.update(status='passed',complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();print('NATIVE_LLAMA_PARITY_CAPTURE_DONE models=1 engines=3 native_profiles=2 rows=6; inspect quality and per-engine limits before parity claims',flush=True)
except BaseException as error:
    status.update(status='failed',error=str(error),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();raise
finally:
    if owned is not None and owned.poll() is None:
        process=psutil.Process(owned.pid);children=[(p.pid,p.create_time()) for p in process.children(recursive=True)]
        owned.terminate()
        try:owned.wait(10)
        except subprocess.TimeoutExpired:owned.kill();owned.wait(10)
        for pid,created in children:
            try:
                child=psutil.Process(pid)
                if child.create_time()==created:
                    child.terminate()
                    try:child.wait(10)
                    except psutil.TimeoutExpired:child.kill()
            except psutil.NoSuchProcess:pass
    if owned_log:owned_log.close()
