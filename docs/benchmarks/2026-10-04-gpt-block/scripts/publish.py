"""Publish bounded GPT block/fusion proofs, distributions and provenance."""
from pathlib import Path
import hashlib,json,re,statistics
root=Path.cwd();base=root/'target/gpt-block';draft=root/'target/performance-cache';out=root/'docs/benchmarks/2026-10-04-gpt-block'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert 'Adopted GPT block quiet ablations and real network suites passed.'in(draft/'block-production-measurements-chain.log').read_text(encoding='utf-8-sig')
build=json.loads((base/'production-proof/manifest.json').read_text(encoding='utf-8'))
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
out.mkdir(parents=True,exist_ok=True);published={};original={}
def capture(src,dest):
 p=out/dest;p.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes();original[dest]=hashlib.sha256(raw).hexdigest()
 p.write_text(raw.decode('utf-8-sig').replace('\r\n','\n'),encoding='utf-8',newline='\n');published[dest]=sha(p)
for folder,label in [(base/'profile','profile'),(base/'production-proof','production'),(base/'ablation','performance'),(base/'live','live'),(draft/'gpt-block-ablation','prototype-performance'),(draft/'gpt-block-live','prototype-live'),(draft/'gpt-prefill-proof','kernel-validation'),(draft/'gpt-block-runtime-proof','prototype-validation')]:
 for p in folder.rglob('*'):
  if p.is_file()and p.suffix in ['.json','.log']:capture(p,label+'/'+p.relative_to(folder).as_posix())
for p in [draft/'check_gpt_block_production.py',draft/'measure_gpt_block_production.py',draft/'profile_gpt_block.py',Path(__file__),root/'scripts/benchmark_cache_stack.py',root/'scripts/validate_cache_streaming.py',root/'scripts/benchmark_engines.py']:
 capture(p,'scripts/'+p.name)
summaries=[]
def stats(v):return {'median':statistics.median(v),'min':min(v),'max':max(v),'samples':v}
for folder,label,expected_count in [(base/'ablation','production',36),(draft/'gpt-block-ablation','prototype',45)]:
 r=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert len(r['rows'])==expected_count and all(x['matches_baseline']for x in r['rows'])
 assert all(x['done']and x['text']==x['sse_text']and x['matches_baseline']for x in r['sse'])
 for mode in r['memory']:
  rows=[x for x in r['rows']if x['mode']==mode and x['cycle']>0 and x['prompt']<2];short=[x for x in r['rows']if x['mode']==mode and x['cycle']>0 and x['prompt']==2]
  assert len(rows)==4 and len(short)==2
  s={'build':label,'mode':mode,'cli_sha256':r['binary_sha256'],'dll_sha256':r['library_sha256'],
   'prompt_tokens':[x['response']['usage']['prompt_tokens']for x in rows],
   'prefill_ms':stats([x['metrics_delta']['rbitnet_inference_prefill_ms_sum']for x in rows]),
   'decode_tokens_per_second':stats([1000*x['response']['usage']['completion_tokens']/x['metrics_delta']['rbitnet_inference_decode_ms_sum']for x in rows]),
   'http_ms':stats([x['wall_ms']for x in rows]),'short_prefill_ms':stats([x['metrics_delta']['rbitnet_inference_prefill_ms_sum']for x in short]),
   'first_visible_sse_content_ms':[x['first_content_ms']for x in r['sse']if x['mode']==mode],'memory':r['memory'][mode]}
  if label=='production':
   assert r['binary_sha256']==build['binary_sha256']and r['library_sha256']==build['library_sha256']
   assert all(x['managed_metrics']['rbitnet_core_cuda_managed_live_bytes']<=x['managed_metrics']['rbitnet_core_cuda_managed_peak_bytes']<=x['managed_metrics']['rbitnet_core_cuda_managed_limit_bytes']<=12288*2**20 for x in rows)
   s['managed_metrics']=rows[-1]['managed_metrics'];s['scoped_metrics_after']=r['scoped_metrics'][mode]['after']
  summaries.append(s)
workspace=(base/'production-proof/workspace.log').read_text(encoding='utf-8');total=sum(int(x)for x in re.findall(r'test result: ok\. ([0-9]+) passed;',workspace));assert total==270,total
real=(base/'production-proof/actual-gpt.log').read_text(encoding='utf-8');assert 'count=32 tile=0'in real and 'worst KL=0.000e0, absolute target NLL delta=0.000e0'in real
for name in ['ordered','grouped','fused','block']:assert '1 passed; 0 failed'in(base/'production-proof'/(name+'.log')).read_text(encoding='utf-8')
profile=json.loads((base/'profile/trace-analysis.json').read_text(encoding='utf-8'))
assert 'Actual GPT block warmed trace proves multi-token projections and grouped FFNs;'in(draft/'block-profile-chain.log').read_text(encoding='utf-8-sig')
record={'profile_trace':profile,'profile_workspace_passed':271,'build':build,'workspace_passed':total,'workspace_ignored':1,'actual_gguf_observed_positions':120,'compared_generations_and_replays':60,'reference_generations':6,'worst_kl':0,'worst_target_nll_delta':0,'modes':summaries,'original_captures':original,'published_captures':published,
 'limits':['Single RTX 4080 SUPER, original GPT-OSS-20B GGUF/tokenizer, fixed resident banks, 12 GiB managed cap / 256 MiB margin.','2048 configured capacity, 24 shared notes, 128 maximum output tokens, one warmup plus two measured cycles.','Four long observations per mode are two prompts times two measured cycles.','Prototype and production captures have separate binary/library hashes; compare modes within each build.','The initial five-mode prototype capture did not retain absolute managed-memory gauges; production capture does.','Global GPU memory deltas are not per-process physical VRAM.','The shared-byte tile is compared as a 32-token configuration; production also measures 32-token ordered warps.','MoE fusion is opt-in, does not fuse the block grouped kernels, and has no established generic speed gain.','Segmented/cache/partial GPT and all MLA block prefill remain pending.','No new Ollama/llama.cpp comparison or parity claim.']}
(out/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
(out/'.gitattributes').write_text('profile/** -text\nproduction/** -text\nperformance/** -text\nlive/** -text\nprototype-*/** -text\nkernel-validation/** -text\nscripts/*.py -text\n',encoding='utf-8',newline='\n')
lines=['# Exact GPT-OSS block prefill and optional MoE fusion — 4 October 2026','',
 'GPT fixed resident banks can now prefill up to 32 positions in one causal block. Projections consume multiple input rows per launch; routed work is grouped by expert and preserves selected-slot order, biases, OAI activation and original GGUF quantized bytes. Attention includes the original alternating windows and sinks. A native verification API returns all-position logits or greedy IDs; ordinary generation returns only its needed final output.',
 '',f'Production validation: {total} workspace tests passed, one ignored; Clippy and release passed with the existing warnings. Four explicit CUDA suites compare ordered projections, grouped experts, fusion and block forwarding against original GPU kernels and independently decoded F64 references. The actual GPT-OSS model compares 120 observed positions, 60 generated/replayed outputs and six references across 16/32 ordered blocks, 32 tiled blocks and optional fusion: observed KL and target-NLL differences are zero. Seeded sampling, penalties, exact prefix replay and cancellation/recovery pass.',
 '', 'The production ablation uses the same binary and DLL for every mode: 2048 configured capacity, 12 GiB managed budget, 24 common notes, maximum 128 output tokens. Cycle 0 warms up; distributions below cover two long prompts in cycles 1 and 2. Short prompts, original requests/responses, first visible SSE times, absolute managed gauges, model counters and process working set are retained in the raw captures. Production has 36 protocol observations, 12 unary/SSE pairs and four explicit stops; its additional network suite checks disconnect/resume in three sampling modes, stop, and four simultaneous requests on the serialized executor.',
 '', '| Production mode | Long prefill median ms | Decode median tok/s | HTTP median ms | Peak RSS GiB | Managed peak MiB |','|---|---:|---:|---:|---:|---:|']
for s in summaries:
 if s['build']=='production':lines.append(f"| {s['mode']} | {s['prefill_ms']['median']:.1f} | {s['decode_tokens_per_second']['median']:.2f} | {s['http_ms']['median']:.1f} | {s['memory']['peak_process_tree_rss_bytes']/2**30:.2f} | {s['managed_metrics']['rbitnet_core_cuda_managed_peak_bytes']/2**20:.1f} |")
index={s['mode']:s for s in summaries if s['build']=='production'}
ratio=index['serial']['prefill_ms']['median']/index['block32']['prefill_ms']['median'];wall=index['serial']['http_ms']['median']/index['block32']['http_ms']['median']
lines += ['',f'Ordered 32-token blocks reduce long prefill by {ratio:.2f}× and HTTP time by {wall:.2f}× in this corpus. Decode speed is close to serial and slightly lower in several observations; this is a prefill gain. Warm prefix medians reuse almost all input state and cannot describe cold requests. Block capacity defaults to 16 and remains configurable; all options remain opt-in.',
 '', 'Separate five-mode prototype experiment (same requests; independently frozen binary/library):','', '| Prototype mode | Long prefill median ms | Decode median tok/s | HTTP median ms |','|---|---:|---:|---:|']
for s in summaries:
 if s['build']=='prototype':lines.append(f"| {s['mode']} | {s['prefill_ms']['median']:.1f} | {s['decode_tokens_per_second']['median']:.2f} | {s['http_ms']['median']:.1f} |")
lines += ['', 'The 32-token shared-byte tiled configuration is slower than ordered blocks; fusion is close to serial. Neither is promoted to a default. Three additional prototype network suites exercise ordered blocks, tiled blocks and fusion. The early prototype greedy mismatch is preserved in kernel-validation; rebuilding from the current matching sources removed it, but its individual cause was not isolated. Empty/inactive early filters are preserved and are not counted as GPU proof.',
 '', 'An additional integration fixture brings the workspace to 271 passing tests (one ignored). Its warmed Nsight capture confirms multi-token ordered-projection grids and joint routed-token grouped-expert kernels; the child completion nonce proves that the generation finished and matched its warm reference. Full grid/count records and capture hashes are in [trace analysis](profile/trace-analysis.json). Profiler timings are excluded from throughput measurements.',
 '', 'Limits: fixed resident GPT expert banks only; cache/partial placement and CPU/adaptive routing keep their serial segments. GLM/MLA block forwarding, true multi-sequence execution, a real draft model and reduced-precision KV are separate work. Fusion changes launch structure and does not currently reclaim its retained intermediate buffers. No new comparison with Ollama or llama.cpp is made.',
 '', 'Reproduce with [production checks](scripts/check_gpt_block_production.py), [quiet/network measurements](scripts/measure_gpt_block_production.py) and the copied benchmark/streaming harnesses. [Manifest](manifest.json) records distributions, source/binary/library hashes and original/published LF-normalized capture hashes.']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('Published exact GPT block/fusion validation and bounded production/prototype distributions.')
