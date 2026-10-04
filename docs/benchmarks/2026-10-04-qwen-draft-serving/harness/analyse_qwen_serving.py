from pathlib import Path
import hashlib,json,statistics
base=Path(__file__).resolve().parent;capture=base/'qwen-serving-proof/ablation/results.json'
d=json.loads(capture.read_text(encoding='utf-8'));assert len(d['rows'])==36 and all(r['matches_baseline'] for r in d['rows'])
result={'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'binary_sha256':d['binary_sha256'],'library_sha256':d['library_sha256'],'rows':[]}
for prompt in sorted({r['prompt'] for r in d['rows']}):
    modes={}
    for mode in dict.fromkeys(r['mode'] for r in d['rows']):
        rows=[r for r in d['rows'] if r['prompt']==prompt and r['mode']==mode and r['cycle']>0];assert len(rows)==2
        rates=[]
        for r in rows:
            m=r['metrics_delta'];tokens=m['rbitnet_completion_tokens_total'];assert tokens==r['response']['usage']['completion_tokens'] and tokens>0
            rates.append(tokens*1000/m['rbitnet_inference_decode_ms_sum'])
        modes[mode]={'median_tokens_per_second':statistics.median(rates),'samples':rates,'median_ttft_ms':statistics.median(r['metrics_delta']['rbitnet_inference_ttft_ms_sum'] for r in rows),
            'proposed':sum(r['metrics_delta']['rbitnet_core_speculative_draft_tokens_total'] for r in rows),'accepted':sum(r['metrics_delta']['rbitnet_core_speculative_accepted_tokens_total'] for r in rows)}
    target=modes['target']['median_tokens_per_second']
    for mode,row in modes.items():row['decode_gain_percent']=100*(row['median_tokens_per_second']/target-1)
    result['rows'].append({'prompt':prompt,'modes':modes})
result['limits']=['Two measured cycles after one warmup, no confidence interval.','Same target IDs/RNG semantics are proved separately; coupled proposal sampling is not the classical q/p rejection sampler.','Same .8B draft and 2B target only; no extrapolation to large MoE or other drafts.','Keep speculative serving optional; promotion requires a demonstrated speed benefit.']
(base/'qwen-serving-proof/analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for row in result['rows']:
    print(row['prompt'],[(mode,round(r['median_tokens_per_second'],2),round(r['decode_gain_percent'],1)) for mode,r in row['modes'].items()])
