from pathlib import Path
import hashlib,json,statistics
base=Path(__file__).resolve().parent/'gpt-norm-delivery-proof'
records=[];captures={}
for layout in ['fixed','segmented']:
    data={}
    for variant in ['baseline','norm']:
        path=base/('quiet-'+layout)/variant/'results.json'
        data[variant]=json.loads(path.read_text(encoding='utf-8'))
        captures[layout+'-'+variant]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(data[variant]['rows'])==27 and all(r['matches_baseline']for r in data[variant]['rows'])
    assert data['baseline']['binary_sha256']==data['norm']['binary_sha256']
    for mode in dict.fromkeys(r['mode']for r in data['baseline']['rows']):
        for prompt in [0,1]:
            row={'layout':layout,'mode':mode,'prompt':prompt,'measurements':{}}
            for variant,capture in data.items():
                samples=[r for r in capture['rows']if r['mode']==mode and r['prompt']==prompt and r['cycle']>0]
                assert len(samples)==2
                rates=[]
                for r in samples:
                    m=r['metrics_delta'];assert m['rbitnet_completion_tokens_total']==r['response']['usage']['completion_tokens']
                    rates.append(m['rbitnet_completion_tokens_total']*1000/m['rbitnet_inference_decode_ms_sum'])
                row['measurements'][variant]={'tps':statistics.median(rates),'samples_tps':rates,'ttft_ms':statistics.median(r['metrics_delta']['rbitnet_inference_ttft_ms_sum']for r in samples)}
            row['gain_percent']=100*(row['measurements']['norm']['tps']/row['measurements']['baseline']['tps']-1)
            records.append(row)
summary={'captures_sha256':captures,'rows':records,'limits':['One warmup plus two measured repeats, no confidence interval.','Same CLI, original versus staged-normalization DLL; router unchanged.','Counter-duration deltas only, not instantaneous gauge differences.','Two writing prompts; no general parity or CPU claim.']}
(base/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
for row in records:print(row['layout'],row['mode'],row['prompt'],round(row['measurements']['baseline']['tps'],2),round(row['measurements']['norm']['tps'],2),round(row['gain_percent'],2))
