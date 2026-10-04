"""Validate completed captures and preserve incomplete ones before resuming."""
from pathlib import Path
import hashlib,json,shutil,time

def reuse_or_archive(name,command,out,binary_sha,library_sha,records):
    if '--output-dir'not in command:return False
    directory=Path(command[command.index('--output-dir')+1]);result=directory/'results.json'
    complete=False;reason='capture absent'
    try:
        r=json.loads(result.read_text(encoding='utf-8'))
        assert r['binary_sha256']==binary_sha and r['library_sha256']==library_sha,'executable/library changed'
        if name.startswith(('trace-','quiet-')):
            assert r['notes']==4 and r['max_tokens']==128
            assert len(r['rows'])==27 and len(r['sse'])==9 and len(r['stops'])==3
            assert set(r['memory'])=={'lru','lfu','least-stale'}
            assert set(r['scoped_metrics'])=={'lru','lfu','least-stale'}
            for policy in r['scoped_metrics']:
                assert r['scoped_metrics'][policy]['before'] and r['scoped_metrics'][policy]['after']
            for x in r['rows']:
                assert x['request']['max_tokens']==128 and x['matches_baseline']
                text=x['response']['choices'][0]['message']['content'];assert text and '\ufffd'not in text
                assert x['env']['RBITNET_MOE_CACHE_POLICY']==x['mode']
                assert x['env']['RBITNET_MOE_ASYNC']=='0' and x['env']['RBITNET_MOE_PREFETCH']=='off'
                assert x['env']['RBITNET_MAX_SEQ']=='2048' and x['env']['RBITNET_CUDA_DEVICE_BUDGET_MB']=='12288'
            for x in r['sse']:
                assert x['done'] and x['matches_baseline'] and x['first_content_ms']is not None
                assert x['text']==x['sse_text'] and x['text'] and '\ufffd'not in x['text']
            assert all('Paris'not in x['response']['choices'][0]['message']['content']for x in r['stops'])
            if name.startswith('trace-'):
                traces=Path(command[command.index('--trace-dir')+1])
                for policy in ['lru','lfu','least-stale']:assert len(list((traces/policy).glob('*.jsonl')))==1
        else:
            assert name.startswith('network-') and r['moe_cache_mib']==512 and r['split_kv']
            policy=command[command.index('--policy')+1]
            assert r['harness_sha256']==hashlib.sha256(Path(command[1]).read_bytes()).hexdigest(), 'network harness changed or its source was not recorded'
            assert r['policy']==policy and r['moe_async']is False and r['prefetch']=='off', 'network capture did not record the synchronous policy protocol'
            assert r['effective_moe_env']==dict(RBITNET_MOE_CACHE_POLICY=policy,RBITNET_MOE_ASYNC='0',RBITNET_MOE_PREFETCH='off',RBITNET_MOE_CACHE_MB='512'), 'actual child options changed'
            assert r['native_async_gauges']==dict(async_enabled=[0],async_pool_bytes=[0],pinned_bytes=[0],pinned_slots=[0]), 'Native storage was not proved synchronous'
            assert len(r['cases'])==5
            assert [x['kind']for x in r['cases']]==['client_disconnect_then_resume']*3+['explicit_stop_http_sse','four_concurrent_requests_serialized_runtime']
            for x in r['cases'][:3]:
                assert x['equal'] and len(x['cancelled_fragments'])>=2
                a=x['expected']['choices'][0]['message']['content'];b=x['resumed']['choices'][0]['message']['content']
                assert a==b and a and '\ufffd'not in b
            x=r['cases'][3];assert x['done'] and x['sse_text']==x['response']['choices'][0]['message']['content'] and 'Paris'not in x['sse_text']
            x=r['cases'][4];assert x['equal'] and len(x['actual'])==len(x['expected'])==4
            assert all(a['choices'][0]['message']['content']==b['choices'][0]['message']['content']for a,b in zip(x['actual'],x['expected']))
            usage=x['all_cases_metrics_delta'];assert usage['rbitnet_core_expert_cache_evictions_total']>0 and usage['rbitnet_core_native_moe_resident_layers_total']>0
            assert usage['rbitnet_core_gpu_gpt_full_tokens_total'if r['gpt_segmented']else'rbitnet_core_gpu_mla_full_tokens_total']>0
            log=out/(name+'.log');assert 'stop and concurrency passed'in log.read_text(encoding='utf-8')
        complete=True
    except (OSError,ValueError,KeyError,AssertionError)as error:reason=str(error)or'capture incomplete'
    if complete:
        record=dict(name=name,result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),action='reused_after_full_capture_validation')
        records.append(record);print('Validated preserved capture:',name,flush=True);return True
    candidates=[directory]
    if name.startswith('trace-'):candidates.append(Path(command[command.index('--trace-dir')+1]))
    log=out/(name+'.log')
    if log.exists():candidates.append(log)
    if any(p.exists()for p in candidates):
        archive=out/'interrupted'/f'{name}-{time.time_ns()}';archive.mkdir(parents=True)
        moved=[]
        for index,p in enumerate(candidates):
            assert p.resolve().is_relative_to(out.resolve()),p
            if p.exists():
                dest=archive/f'{index}-{p.name}';shutil.move(str(p),str(dest));moved.append(str(dest))
        records.append(dict(name=name,action='preserved_incomplete_capture',reason=reason,paths=moved))
    return False
