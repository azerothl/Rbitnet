"""Source-bound performance history and isolated ablations; no new benchmark."""
from pathlib import Path
import csv,hashlib,json,statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
root=next(p for p in Path(__file__).resolve().parents if (p/'docs/benchmarks/2026-10-03/summary.json').is_file());base=root/'target/performance-cache';out=base/'performance-evolution';out.mkdir(exist_ok=True)
sources={};records=[]
def load(path):
    path=Path(path);raw=path.read_bytes();key=str(path.resolve());sources[key]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));return json.loads(raw)
models=['llama32-1b','qwen35-2b','gpt-oss-20b','glm47-flash']
names={'llama32-1b':'Llama 3.2 1B','qwen35-2b':'Qwen3.5 2B','gpt-oss-20b':'GPT-OSS 20B','glm47-flash':'GLM 4.7 Flash (MoE)'}
colors=['#2563eb','#7c3aed','#059669','#ea580c']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#0f172a','axes.edgecolor':'#cbd5e1','figure.facecolor':'#f8fafc','axes.facecolor':'#ffffff','savefig.facecolor':'#f8fafc'})
history=[('Initial\n7eca9ad','2026-10-03'),('Architectures + SIMD\nc5e5a8a + modifications','2026-10-03-optimized'),('CUDA résident / FFN\nab520d0 + modifications','2026-10-03-parity-round2')]
loaded=[load(root/'docs/benchmarks'/folder/'summary.json')for _,folder in history]
assert all(d['protocol']['max_output_tokens']==32 and d['protocol']['threads']==16 and d['protocol']['repeats']==3 for d in loaded)
fig,axes=plt.subplots(1,2,figsize=(15,7));fig.subplots_adjust(top=.78,bottom=.27,left=.07,right=.97,wspace=.24)
fig.suptitle('Rbitnet — évolution des premières campagnes comparables',x=.07,ha='left',fontsize=20,fontweight='bold',y=.96)
fig.text(.07,.905,'3 octobre 2026 • Ryzen 7 9800X3D / RTX 4080 SUPER • médiane de décodage, 32 tokens maximum',fontsize=11,color='#475569')
for ax,backend in zip(axes,['cpu','gpu']):
    for model,color in zip(models,colors):
        values=[]
        for step,d in enumerate(loaded):
            row=next(r for r in d['rows']if r['model']==model and r['engine']=='rbitnet'and r['backend']==backend)
            value=row['decode_tokens_per_second_median'];values.append(np.nan if value is None else value)
            records.append(dict(figure='history',panel=backend,model=model,step=history[step][0].replace('\n',' / '),metric='decode_tokens_per_second',value=value,status=row['status'],source=str(root/'docs/benchmarks'/history[step][1]/'summary.json')))
            if value is not None:
                ax.vlines(step,row['decode_tokens_per_second_min'],row['decode_tokens_per_second_max'],color=color,alpha=.45,lw=4)
        ax.plot(range(3),values,color=color,lw=2.5,marker='o',markersize=7,label=names[model])
        ax.annotate(f'{values[-1]:.1f}',(2,values[-1]),xytext=(9,0),textcoords='offset points',color=color,fontsize=10,va='center',fontweight='bold')
    ax.set_yscale('log');ax.set_xlim(-.15,2.48);ax.set_xticks(range(3),[label for label,_ in history]);ax.set_ylabel('Tokens/s — échelle logarithmique');ax.set_title('CPU'if backend=='cpu'else'CUDA / placement hybride selon le modèle',loc='left',pad=18)
    ax.grid(axis='y',which='major',color='#e2e8f0',zorder=0);ax.tick_params(axis='x',length=0,pad=10)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.065,.868),ncol=4,frameon=False,fontsize=10)
fig.text(.07,.165,'Chargement initial de Qwen, GPT-OSS et GLM : échec. Aucune vitesse n’est attribuée à ces trois modèles avant leur prise en charge.',fontsize=10,color='#475569')
fig.text(.07,.12,'Même export par modèle, 16 threads, un client, chauffe exclue, trois mesures. Traits verticaux : min/max observés, pas un intervalle de confiance.',fontsize=10,color='#475569')
fig.text(.07,.075,'Contrôles courts : Llama/GPT-OSS/GLM 3/3 ; Qwen 2/3 après correction. Ils ne constituent pas une évaluation générale de qualité.',fontsize=10,color='#475569')
fig.text(.07,.03,'Les ablations à 128 tokens du 4 octobre et le comparatif final en cours sont présentés séparément ; aucune courbe ne mélange leurs protocoles.',fontsize=10,color='#475569')
fig.savefig(out/'historique_cpu_gpu.png',dpi=160);fig.savefig(out/'historique_cpu_gpu.svg')

abfig,axarr=plt.subplots(4,2,figsize=(16,19));abfig.subplots_adjust(top=.91,bottom=.055,left=.065,right=.97,hspace=.76,wspace=.29)
abfig.suptitle('Optimisations récentes — effets isolés, gains et régressions',x=.065,ha='left',fontsize=21,fontweight='bold',y=.975)
abfig.text(.065,.947,'Chaque panneau compare sa propre référence. Les gains ne s’additionnent pas et ne forment pas une chaîne de versions.',fontsize=11,color='#475569')
axes=iter(axarr.flat)
def bars(ax,title,labels,values,unit,source,model,subtitle='',barcolors=None):
    assert len(labels)==len(values) and all(np.isfinite(values))
    ax.bar(range(len(values)),values,color=barcolors or ['#94a3b8']+['#2563eb']*(len(values)-1),width=.64,zorder=3)
    ax.set_xticks(range(len(labels)),labels,fontsize=9);ax.set_ylabel(unit,fontsize=9);ax.set_title(title,loc='left',fontsize=12,pad=25)
    ax.text(0,1.035,subtitle,transform=ax.transAxes,fontsize=9,color='#64748b');ax.grid(axis='y',color='#e2e8f0',zorder=0);ax.set_ylim(0,max(values)*1.25)
    for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.1f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
    for label,value in zip(labels,values):records.append(dict(figure='ablations',panel=title,model=model,step=label.replace('\n',' / '),metric=unit,value=value,status='measured',source=str(source)))
    ax.tick_params(axis='x',length=0,pad=8)

p=root/'docs/benchmarks/2026-10-03-split-kv/llama-summary.json';d=load(p);v=[d['modes'][m]['decode_tokens_per_second']['median']for m in ['baseline','split']]
bars(next(axes),'1. Llama — attention par tuiles KV (#103)',['Référence','Split KV'],v,'Tokens/s de décodage',p,models[0],f'128 tokens • même campagne • {100*(v[1]/v[0]-1):+.1f}%')
p=root/'docs/benchmarks/2026-10-03-split-kv/qwen-summary.json';d=load(p);v=[d['modes'][m]['decode_tokens_per_second']['median']for m in ['baseline','split']]
bars(next(axes),'2. Qwen — attention par tuiles KV (#103)',['Référence','Split KV'],v,'Tokens/s de décodage',p,models[1],f'128 tokens • même campagne • {100*(v[1]/v[0]-1):+.1f}%')
p=root/'docs/benchmarks/2026-10-03-gpt-full/summary.json';d=load(p);v=[d['modes'][m]['decode_tokens_per_second']['median']for m in ['baseline','full','full-split']]
bars(next(axes),'3. GPT-OSS — pipeline résident (#106)',['Référence','Résident','Résident\n+ split KV'],v,'Tokens/s de décodage',p,models[2],'128 tokens • médianes des prompts longs de cette ablation')
p=base/'gpt-norm-delivery-proof/analysis.json';d=load(p);r=next(r for r in d['rows']if r['layout']=='fixed'and r['mode']=='serial'and r['prompt']==0);v=[r['measurements'][m]['tps']for m in ['baseline','norm']]
bars(next(axes),'4. GPT-OSS — normalisation ordonnée (#124)',['Avant','Après'],v,'Tokens/s de décodage',p,models[2],f'128 tokens • même CLI, DLL différente • {r["gain_percent"]:+.1f}%')
p=base/'cpu-direct-rows-proof/analysis.json';d=load(p);rows=[next(r for r in d['rows']if r['model']==model and r['prompt']==0)for model in models];ax=next(axes);values=[100+r['decode_gain_percent']for r in rows]
bars(ax,'5. CPU — sortie directe des lignes (prototype)',['Llama','Qwen','GPT-OSS','GLM'],values,'Indice de débit — référence = 100',p,'four-models','128 tokens • contexte 512 • prompt histoire • option désactivée par défaut',barcolors=['#2563eb'if v>=100 else'#dc2626'for v in values]);ax.axhline(100,color='#64748b',ls='--',lw=1.2);ax.set_ylim(85,115)
# Replace index annotations with signed gains so small regressions stay visible.
for text in list(ax.texts)[1:]:text.remove()
for i,row in enumerate(rows):ax.text(i,values[i]+1,f'{row["decode_gain_percent"]:+.1f}%',ha='center',fontsize=10,fontweight='bold')
p=base/'qwen-serving-proof/analysis.json';d=load(p);row=next(r for r in d['rows']if r['prompt']==0);keys=['target','draft-1','draft-4','draft-8'];v=[row['modes'][key]['median_tokens_per_second']for key in keys]
bars(next(axes),'6. Qwen — décodage spéculatif (#123)',['Cible seule','Draft 1','Draft 4','Draft 8'],v,'Tokens/s de décodage',p,models[1],'128 tokens • sorties identiques testées • ralentissement, option expérimentale',barcolors=['#2563eb','#dc2626','#dc2626','#dc2626'])
p=Path('C:/Users/azero/.codex/worktrees/llama-continuous-delivery/Rbitnet/docs/benchmarks/2026-10-04-llama-continuous-fresh/summary.json');d=load(p);keys=['reference','dense-slots1','dense-slots4','dense-slots8','paged-slots8'];v=[next(r['tokens_per_second']for r in d['rows']if r['config']==k)for k in keys]
bars(next(axes),'7. Llama — batching continu (#120)',['Sérialisé','Dense\n1 slot','Dense\n4 slots','Dense\n8 slots','Pagé\n8 slots'],v,'Débit TOTAL HTTP — tokens/s',p,models[0],'8 clients • 798 tokens par vague • mesure fraîche • pas le débit par requête')
p=base/'expert-arena-proof/quiet-analysis.json';d=load(p);v=[next(r['memory']['gpu_global_peak_delta_mib']for r in d['rows']if r['model']==model and r['expert_budget_mib']==8192 and r['mode']==mode)for model in models[2:]for mode in ['async-demand','arena-demand']]
bars(next(axes),'8. MoE — arène d’experts (#121)',['GPT-OSS\nasynchrone','GPT-OSS\n+ arène','GLM\nasynchrone','GLM\n+ arène'],v,'Pic GPU GLOBAL − niveau initial (MiB)',p,'GPT-OSS / GLM','Budget experts 8192 MiB • gain de capacité • ce n’est pas la VRAM du processus',barcolors=['#94a3b8','#059669','#94a3b8','#059669'])
abfig.text(.065,.021,'Ablations mono-client : chauffe exclue, deux mesures par prompt sauf mention contraire. Mesures descriptives ; aucune parité générale revendiquée.',fontsize=10,color='#475569')
abfig.savefig(out/'optimisations_isolees.png',dpi=150);abfig.savefig(out/'optimisations_isolees.svg')
with PdfPages(out/'evolution_performances_rbitnet.pdf')as pdf:pdf.savefig(fig);pdf.savefig(abfig)
plt.close('all')
dataset=dict(generated_date='2026-10-04',hardware='Ryzen 7 9800X3D, RTX 4080 SUPER 16 GiB',sources=sources,records=records,limits=['History covers three comparable 32-token campaigns, not current combined-stack throughput.','Each 128-token ablation has its own baseline and option set; do not compound gains or connect across panels.','Errors have no throughput value. Three short quality probes are not a general quality evaluation.','Continuous serving is aggregate HTTP throughput; arena memory is global GPU sampling.','CPU direct-row source is an isolated prototype; combined validation remains pending.','The latest four-engine CPU/GPU comparison is queued; no value is invented for it.'])
(out/'data.json').write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with(out/'data.csv').open('w',encoding='utf-8-sig',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
(out/'README.md').write_text('''# Évolution des performances Rbitnet

`historique_cpu_gpu.png` : trois campagnes du 3 octobre, 32 tokens maximum,
16 threads, un client, chauffe exclue et trois mesures. Les fins de lignes
identifient les versions ou modifications locales exécutées. Les erreurs de
chargement initiales ne reçoivent aucune vitesse. Les lignes verticales montrent
la variation min/max observée, pas une incertitude statistique estimée.

`optimisations_isolees.png` : huit ablations à référence propre. Les gains ne
s'additionnent pas et les panneaux ne sont pas les points d'une même courbe.
Le panneau CPU direct correspond au prototype isolé. Le batching continu est
un débit total HTTP. La mémoire de l'arène est un pic global GPU au-delà du
niveau initial, pas une mesure propre au processus.

Le PDF contient les deux figures ; les SVG sont vectoriels. `data.json` donne
les valeurs et les empreintes SHA-256 des sources, `data.csv` les points du
graphique. `generator.py` permet de régénérer les figures avec Python, NumPy et
Matplotlib, à condition de disposer des sources locales indiquées dans les données.
Les JSON sources conservent leurs protocoles et captures référencées.
Le comparatif final de la pile complète est encore en file : aucun résultat
ne lui est attribué. Aucun benchmark supplémentaire n'a été exécuté pour ces figures.
''',encoding='utf-8',newline='\n')
for source,identity in sources.items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==identity['sha256']
if Path(__file__).resolve()!=out/'generator.py':shutil.copy2(Path(__file__),out/'generator.py')
print('PERFORMANCE_FIGURES_DONE',out,'points',len(records),'bound sources',len(sources))
