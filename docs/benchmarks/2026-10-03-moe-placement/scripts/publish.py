"""Publish bounded claims and original/published capture provenance for #110."""
from pathlib import Path
import hashlib,json,re,statistics,shutil
root=Path.cwd();base=root/'target/moe-placement';out=root/'docs/benchmarks/2026-10-03-moe-placement'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((base/'validation-manifest.json').read_text())
assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items())
assert 'All twelve policy and previous/current matched ablations completed.'in(root/'target/moe-placement-previous-benchmark-chain.log').read_text(encoding='utf-8-sig')
assert 'All four MoE live policy/metrics/lifecycle/streaming suites passed.'in(root/'target/moe-placement-live-chain.log').read_text(encoding='utf-8-sig')
assert 'running 4 tests'in(base/'ownership.log').read_text(encoding='utf-8')
out.mkdir(parents=True,exist_ok=True);published={};original={}
def capture(src,dest):
 p=out/dest;p.parent.mkdir(parents=True,exist_ok=True)
 raw=src.read_bytes();original[dest]=hashlib.sha256(raw).hexdigest()
 p.write_text(raw.decode('utf-8-sig').replace('\r\n','\n'),encoding='utf-8',newline='\n');published[dest]=sha(p)
for p in base.glob('*.log'):capture(p,'validation/'+p.name)
for p in base.glob('*manifest.json'):capture(p,'validation/'+p.name)
for p in (base/'live').rglob('*'):
 if p.is_file()and p.suffix in ['.json','.log']:capture(p,'live/'+p.relative_to(base/'live').as_posix())
for p in (base/'ablation').rglob('*'):
 if p.is_file()and p.suffix in ['.json','.log']:capture(p,'performance/'+p.relative_to(base/'ablation').as_posix())
for p in [root/'target/performance-cache/validate_cost.py',base/'live.py',base/'benchmark.py',base/'benchmark_previous.py',Path(__file__),root/'scripts/benchmark_cache_stack.py',root/'scripts/validate_cache_streaming.py',root/'scripts/benchmark_engines.py']:
 capture(p,'scripts/'+p.name)
for source,name in [('target/moe-placement-validation-chain.log','production-chain.log'),('target/moe-placement-live-chain.log','live-chain.log'),('target/moe-placement-benchmark-chain.log','policy-benchmark-chain.log'),('target/moe-placement-previous-benchmark-chain.log','previous-benchmark-chain.log')]:capture(root/source,'validation/'+name)
summaries=[]
for folder in sorted((base/'ablation').iterdir()):
 d=json.loads((folder/'results.json').read_text());rows=[r for r in d['rows']if r['cycle']>0];long=[r for r in rows if r['prompt']<2];short=[r for r in rows if r['prompt']==2]
 assert len(long)==4 and len(short)==2 and all(r['matches_baseline']for r in d['rows'])
 assert len(d['sse'])==3 and all(s['done']and s['text']==s['sse_text']and s['matches_baseline']for s in d['sse'])
 def values(rs,key):return[r['metrics_delta'][key]for r in rs]
 def stats(values):return dict(median=statistics.median(values),min=min(values),max=max(values),samples=values)
 tps=[r['response']['usage']['completion_tokens']*1000/r['metrics_delta']['rbitnet_inference_decode_ms_sum']for r in long]
 scoped={}
 for raw in d.get('scoped_metrics',{}).values():
  for n,_,v in re.findall(r'^(rbitnet_moe_layer_\w+)\{([^\n]+)\} ([0-9]+)$',raw['after'],re.M):scoped[n]=scoped.get(n,0)+int(v)
 mode=d['rows'][0]['mode'];summaries.append(dict(case=folder.name,mode=mode,
  actual_context=d['rows'][0]['env']['RBITNET_MAX_SEQ'],prompt_tokens=[r['response']['usage']['prompt_tokens']for r in long],
  completion_tokens=[r['response']['usage']['completion_tokens']for r in long],prefill_ms=stats(values(long,'rbitnet_inference_prefill_ms_sum')),
  decode_tokens_per_second=stats(tps),http_ms=stats([r['wall_ms']for r in long]),short_prefill_ms=stats(values(short,'rbitnet_inference_prefill_ms_sum')),
  first_visible_content_ms=d['sse'][0]['first_content_ms'],memory=d['memory'][mode],scoped_metrics_after_all_cases=scoped,
  cli_sha256=d['binary_sha256'],dll_sha256=d['library_sha256']))
workspace=(base/'workspace.log').read_text();total=sum(int(v)for v in re.findall(r'test result: ok\. ([0-9]+) passed;',workspace))
assert total==265,total
max_kl=max_nll=0.0
for p in base.glob('real-*.log'):
 m=re.search(r'worst KL=([0-9.e+-]+), worst absolute target NLL delta=([0-9.e+-]+)',p.read_text());assert m,p
 max_kl=max(max_kl,float(m[1]));max_nll=max(max_nll,float(m[2]))
index={s['case']:s for s in summaries}
memory_comparisons=[]
for model in ['gpt-oss-20b','glm47-flash']:
 new=index[f'{model}-cache-cache0'];old=index[f'{model}-previous-cache0']
 previous=old['memory']['peak_process_tree_rss_bytes'];current=new['memory']['peak_process_tree_rss_bytes']
 memory_comparisons.append(dict(model=model,previous_peak_rss_bytes=previous,current_peak_rss_bytes=current,
  reduction_bytes=previous-current,reduction_percent=100*(previous-current)/previous,
  current_over_previous_prefill_ratio=new['prefill_ms']['median']/old['prefill_ms']['median'],
  current_over_previous_decode_ratio=new['decode_tokens_per_second']['median']/old['decode_tokens_per_second']['median']))
policy_comparisons=[]
for model in ['gpt-oss-20b','glm47-flash']:
 for cache in [0,8192]:
  reference=index[f'{model}-cache-cache{cache}'];adaptive=index[f'{model}-adaptive-cache{cache}']
  policy_comparisons.append(dict(model=model,cache_mib=cache,
   adaptive_over_cache_prefill_ratio=adaptive['prefill_ms']['median']/reference['prefill_ms']['median'],
   adaptive_over_cache_decode_ratio=adaptive['decode_tokens_per_second']['median']/reference['decode_tokens_per_second']['median']))
result={'build':manifest,'workspace_passed':total,'workspace_ignored':1,'observed_logit_positions':240,
 'policy_generations':120,'reference_generations':30,'worst_kl':max_kl,'worst_absolute_target_nll_delta':max_nll,
 'ablation_rows':summaries,'mapped_previous_current_comparisons':memory_comparisons,'adaptive_cache_comparisons':policy_comparisons,
 'published_captures':published,'original_captures':original,
 'limits':['12 GiB managed cap / 256 MiB margin; single RTX 4080 SUPER and one server at a time',
 '24 shared notes, 2048 configured capacity, 128 maximum output tokens, one warmup plus two measured cycles',
 'Whole routed-FFN selection; mixed per-expert compute and async transfers remain separate work',
 'Previous binary capture Git fields describe the harness checkout; its validation-manifest.json identifies the previous engine build',
 'Global VRAM delta is not per-process physical VRAM; mmap pages still count in working set',
 'No new Ollama/llama.cpp parity claim or automatic enablement of the adaptive selector']}
(out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
(out/'.gitattributes').write_text('validation/** -text\nlive/** -text\nperformance/** -text\nscripts/*.py -text\n',encoding='utf-8',newline='\n')
lines=['# Routed FFN placement and mapped GGUF weights — 3 October 2026','',
 'This lot adds mapped host backing, weakly registered per-layer MoE counters, and opt-in CPU/adaptive whole-FFN policies. `cache` remains the default. CPU expert mode retains the placement of attention, head and shared FFNs; it is not an entirely CPU model.',
 '',f'Production validation: {total} workspace tests passed, one ignored, Clippy/release passed; 3 mapped tests and 4 transfer/lifetime tests executed, with the actual CUDA fixture enabled. The native suite has 32 passing tests, including optional tests whose model-specific flags were not enabled. Five actual GPT-OSS/GLM budget cases compare 240 observed positions and 120 generated/replayed outputs plus 30 reference outputs; worst KL {max_kl:.3e}, absolute target-NLL delta {max_nll:.3e}, all observed argmaxes and tested generations agree.',
 '', 'Four live CPU/adaptive model suites verify scoped metrics, output, unload/reload, disconnection/resume, seeded sampling, penalties, explicit stop and four concurrent requests on the serialized executor. After unload there are no model counter owners and no remaining managed weight/KV/expert/prefix allocations; reload gets a different model_id.',
 '', 'Same-build policy ablations use identical requests, a 12 GiB managed cap, 2048 capacity, 24 common notes and maximum 128 output tokens. Cycle 0 warms up; medians below cover four long observations (two prompts × two measured cycles). Raw JSON retains all requests, output token counts, timings, model counters and memory; exact strings and HTTP/SSE output were checked across policies. Previous/current binary comparisons use the same harness, DLL and requests; their build manifests distinguish source provenance. This is a smaller prompt corpus than the earlier 72-note reports.',
 '', '| Model / policy / cache | Long prefill median ms | Decode median tok/s | HTTP median ms | Peak process tree RSS GiB |', '|---|---:|---:|---:|---:|']
for s in summaries:lines.append(f"| {s['case']} | {s['prefill_ms']['median']:.1f} | {s['decode_tokens_per_second']['median']:.2f} | {s['http_ms']['median']:.1f} | {s['memory']['peak_process_tree_rss_bytes']/2**30:.2f} |")
lines += ['', 'Matched mapped-backing comparison (same requests, policy, native DLL and managed cap):', '',
 '| Model | Previous peak RSS GiB | Current peak RSS GiB | Peak RSS reduction | Current/previous decode |',
 '|---|---:|---:|---:|---:|']
for s in memory_comparisons:lines.append(f"| {s['model']} | {s['previous_peak_rss_bytes']/2**30:.2f} | {s['current_peak_rss_bytes']/2**30:.2f} | {s['reduction_percent']:.1f}% | {s['current_over_previous_decode_ratio']:.3f} |")
lines += ['', 'Policy interpretation is limited to these two measured cycles and these requests. Adaptive execution remains opt-in. In particular, a correct CPU/GPU decision mechanism does not by itself establish a throughput gain. Ratios below compare adaptive execution to the corresponding same-build cache policy (prefill below 1 and decode above 1 are favorable):', '',
 '| Model / cache MiB | Adaptive/cache prefill | Adaptive/cache decode |', '|---|---:|---:|']
for s in policy_comparisons:lines.append(f"| {s['model']} / {s['cache_mib']} | {s['adaptive_over_cache_prefill_ratio']:.3f} | {s['adaptive_over_cache_decode_ratio']:.3f} |")
lines += ['', 'The policy decision is per layer for the entire selected routed FFN. GPU allocation/fill calibration, warm-cache locality and CPU parallel work can change its costs; these data do not justify enabling the selector for every model. A 0 MiB cache retains fixed placement. Explicit CPU mode ignores the expert cache budget and avoids placing unused expert banks. The mapped backing avoids a second owned host copy in this shared planner, while mapped pages remain physically resident when touched.',
 '', 'This report does not implement async overlap (#84), mixed per-expert execution (#86), device KV pages/formats (#92/#93), persistent KV tiers (#94), true multi-sequence forwards (#96), or a real GGUF draft (#97). The ignored block/fusion prototypes are outside this build and outside these measurements.',
 '', 'Reproduction: [serialized checks](scripts/validate_cost.py), [live suites](scripts/live.py), [policy benchmark](scripts/benchmark.py), [previous/current benchmark](scripts/benchmark_previous.py). [Manifest](manifest.json) stores distributions, source/binary hashes, original and published LF-normalized capture hashes. Initial empty ownership-filter output is preserved separately; only the subsequent four-test run is counted.']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('Published mapped/placement validation and twelve matched captures, with bounded claims.')
