from pathlib import Path
import json, csv, statistics, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path(__file__).resolve().parent
j=json.loads((b/'nucleus-combined-experiment.json').read_text())
assert j['complete'] and j['status']=='passed'
w=Path(j['workspace']);dest=w/'docs/benchmarks/2026-10-04-nucleus-combined-fresh'
receipt=json.loads((dest/'receipt.json').read_text())
d=json.loads((b/'nucleus-combined-proof/live/results.json').read_text())
assert len(d['cases'])==96 and len(d['timings'])==16
models=['llama32-1b','qwen35-2b','gpt-oss-20b','glm47-flash']
names=['Llama 3.2 1B','Qwen3.5 2B','GPT-OSS 20B','GLM-4.7 Flash']
rows=[];cells={}
for model in models:
    prompts=list(dict.fromkeys(r['prompt'] for r in d['timings'] if r['model']==model))
    assert len(prompts)==2
    for pi,prompt in enumerate(prompts):
        for variant in ['0','1']:
            ts=[r for r in d['timings'] if (r['model'],r['prompt'],r['variant'])==(model,prompt,variant)]
            assert len(ts)==1
            measured=[s for s in ts[0]['samples'] if not s['warmup']]
            assert len(measured)==2
            values=[s['wall_tokens_per_second'] for s in measured]
            cells[model,pi,variant]=(statistics.median(values),min(values),max(values))
            for sample in measured:rows.append(dict(model=model,prompt_index=pi,variant=variant,cycle=sample['cycle'],wall_ms=sample['wall_ms'],completion_tokens=sample['response']['usage']['completion_tokens'],wall_tokens_per_second=sample['wall_tokens_per_second']))
with (dest/'sampling-http.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,model,name in zip(axes.flat,models,names):
    for vi,variant in enumerate(['0','1']):
        points=[cells[model,pi,variant] for pi in range(2)]
        yy=[p[0] for p in points]
        ax.bar([pi+(vi-.5)*.34 for pi in range(2)],yy,width=.32,color=['#64748b','#0891b2'][vi],label=['Tri original','Top-p adaptatif'][vi],yerr=[[p[0]-p[1] for p in points],[p[2]-p[0] for p in points]],capsize=4)
    ax.set_title(name);ax.set_xticks([0,1],['Récit','Musée']);ax.set_ylabel('Tokens de sortie / seconde HTTP');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(fontsize=9)
fig.suptitle('Rbitnet combiné : effet isolé du sampling top-p sur GPU',fontsize=14)
fig.text(.5,.018,'RTX 4080 SUPER · top-p 0,9 · température 0,7 · seed 42 · 128 tokens max\n1 chauffe + 2 mesures ; barres = médiane, traits = min–max ; préremplissage inclus.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(dest/'sampling-http.png',dpi=180);fig.savefig(dest/'sampling-http.pdf');plt.close(fig)
table=[]
for model,name in zip(models,names):
    for pi in range(2):
        a=cells[model,pi,'0'][0];z=cells[model,pi,'1'][0]
        table.append(f'| {name} | {pi+1} | {a:.2f} | {z:.2f} | {(z/a-1)*100:+.1f} % |')
readme='''# Exact nucleus sampling on the merged cache stack

Fresh validation binds the combined checkout at `HEAD_COMPILED`, after merging
main `ba2cd2fc42448794452f057c02a6c0af22e942d8`. The only compiled differences
from that main are the four reviewed sampling files from #131. Defaults remain
unchanged: activate with `RBITNET_CPU_TOP_P_HEAP=1`.

WORKSPACE_TESTS workspace tests passed, one ignored; check and Clippy passed.
2,028 synthetic cases and 864 comparisons on 24 actual frozen GPT-OSS vectors
preserve selected tokens and RNG state. The fresh release CLI produced 96
JSON/SSE/zero-budget records across four models and CPU/CUDA, with identical
outputs, usage and finish reasons between original and adaptive samplers.

The Native DLL is reused from the sealed integration proof, not rebuilt here.
Every Native source path matches that compiled source after CRLF normalization;
the DLL SHA-256 is verified. `raw/native-reuse.json.gz` records this boundary.

## Fresh GPU HTTP throughput

| Model | Prompt | Original tokens/s | Adaptive tokens/s | Change |
|---|---:|---:|---:|---:|
TABLE

![Isolated sampling effect](sampling-http.png)

These are median output-token / HTTP-wall-time rates, including prefill. One
warmup and two measured samples per cell; original then adaptive order, without
randomization or statistical-significance claim. The two prompts are a story
and a description of a future museum. Same-model sampled outputs match exactly between variants.
This is neither decode-only throughput nor a new comparison with other engines.
Broad nuclei can still incur sorting-fallback overhead; no default promotion.

## Evidence and reproduction

`receipt.json` indexes deterministic gzip captures and exact executed helpers;
`raw/manifest.json.gz` binds compiled sources and artifact identities.
Run the archived checker from a repository checkout with local model paths
adapted. It uses `top-p-actual-inputs.json` for the frozen input identities;
the local float archives, models, DLL and executable are omitted.
Previous standalone measurements remain in the adjacent
`2026-10-04-adaptive-nucleus-fresh` report; these new captures do not replace them.
`sampling-http.csv`, the PNG and PDF expose the current combined measurements.
'''.replace('HEAD_COMPILED',j['head']).replace('WORKSPACE_TESTS',str(receipt['workspace_tests']['passed'])).replace('TABLE','\n'.join(table))
(dest/'README.md').write_text(readme,encoding='utf-8')
print('\n'.join(table));print('tests',receipt['workspace_tests'],'captures',len(receipt['captures']))
