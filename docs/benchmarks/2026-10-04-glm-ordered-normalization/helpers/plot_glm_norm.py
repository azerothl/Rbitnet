from pathlib import Path
import argparse,gzip,json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
data=a.summary.read_bytes();data=gzip.decompress(data) if a.summary.suffix=='.gz' else data
rows=json.loads(data)['rows']; assert len(rows)==2
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.8));colors={'baseline':'#8793A3','staged':'#2465B6'}
for axis,metric,title in zip(axes,['decode_tps','prefill_ms'],['Décodage · tokens/s','Prefill · secondes']):
    maximum=0
    for variant,offset,label in [('baseline',-.18,'Avant'),('staged',.18,'Normalisation optimisée')]:
        for i,row in enumerate(rows):
            values=row['values'][variant][metric];values=[v/1000 for v in values] if metric=='prefill_ms' else values
            median=statistics.median(values);maximum=max(maximum,max(values))
            axis.bar(i+offset,median,width=.32,color=colors[variant],label=label if i==0 else None,alpha=.88)
            axis.scatter([i+offset+o for o in [-.055,0,.055]],values,color='#192737',s=20,zorder=3)
            axis.text(i+offset,max(values)+.02*maximum,f'{median:.2f}',ha='center',va='bottom',fontsize=10)
    axis.set_xticks([0,1],['Récit','Code']);axis.set_ylim(0,maximum*1.18);axis.set_title(title,loc='left',fontweight='bold');axis.grid(axis='y',alpha=.16);axis.set_axisbelow(True)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,frameon=False,fontsize=10,ncol=2,loc='upper center',bbox_to_anchor=(.5,.9))
fig.suptitle('GLM-4.7-Flash Q4_K_M · RTX 4080 SUPER · cache LRU 8 GiB',fontsize=14,fontweight='bold')
fig.text(.5,.035,'Trois répétitions après chauffe ; points = mesures individuelles.\nDébit de décodage rapporté par le serveur. Pas de comparaison avec les autres moteurs.',ha='center',fontsize=9,color='#38485B')
fig.tight_layout(rect=[0,.11,1,.91]);a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output.with_suffix('.png'),dpi=180);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
