"""Publish adopted F32 pages with physical memory and bounded speed results."""
from pathlib import Path
import hashlib,json,re,statistics
root=Path.cwd();base=root/'target/paged-kv';draft=root/'target/performance-cache';proof=base/'production-proof';out=root/'docs/benchmarks/2026-10-04-paged-kv'
assert 'Adopted native Llama F32 pages, actual GPU ownership, quiet ablations and network suites passed.'in(draft/'paged-production-chain.log').read_text(encoding='utf-8-sig')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((proof/'manifest.json').read_text(encoding='utf-8'));assert all(sha(root/p)==h for p,h in m['source_sha256'].items())
out.mkdir(parents=True,exist_ok=True);original={};published={}
def capture(src,dest):
 p=out/dest;p.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes();original[dest]=hashlib.sha256(raw).hexdigest()
 p.write_text(raw.decode('utf-8-sig').replace('\r\n','\n'),encoding='utf-8',newline='\n');published[dest]=sha(p)
for folder,label in [(proof,'production'),(base/'ablation','ablation'),(base/'live','live'),(draft/'paged-current-proof','prototype'),(draft/'paged-ablation','prototype-ablation'),(draft/'paged-live','prototype-live')]:
 for p in folder.iterdir():
  if p.is_file()and p.suffix in ['.json','.log']:capture(p,label+'/'+p.name)
for name in ['check_paged_production.py','check_paged_current.py']:
 capture(draft/name,'reproduction/'+name)
for name in ['paged_benchmark.py','paged_live.py']:capture(draft/'followup-harness'/name,'reproduction/'+name)
capture(base/'publish.py','reproduction/publish.py');capture(draft/'paged-production-chain.log','logs/paged-production-chain.log')
r=json.loads((base/'ablation/results.json').read_text(encoding='utf-8'));assert len(r['rows'])==36 and all(x['matches_baseline']for x in r['rows'])and len(r['sse'])==12 and len(r['stops'])==4
summary=[]
for mode in ['dense','dense-prefix','paged','paged-prefix']:
 rows=[x for x in r['rows']if x['mode']==mode and x['cycle']>0 and x['prompt']<2]
 cats={k:int(v)for k,v in re.findall(r'^rbitnet_core_cuda_managed_category_bytes\{category="([^\"]+)"\} ([0-9]+)$',r['scoped_metrics'][mode]['after'],re.M)}
 assert 'kv_state'in cats and 'prefix'in cats
 summary.append({'mode':mode,'measured_long_rows':len(rows),'median_prefill_ms':statistics.median(x['metrics_delta']['rbitnet_inference_prefill_ms_sum']for x in rows),
  'median_decode_tps':statistics.median(x['response']['usage']['completion_tokens']*1000/x['metrics_delta']['rbitnet_inference_decode_ms_sum']for x in rows),'median_http_ms':statistics.median(x['wall_ms']for x in rows),
  'after_complete_protocol_categories':cats,'final_kv_plus_prefix_mib':(cats['kv_state']+cats['prefix'])/2**20,
  'median_long_managed_live_mib':statistics.median(x['managed_metrics']['rbitnet_core_cuda_managed_live_bytes']/2**20 for x in rows),'peak_managed_mib':max(x['managed_metrics']['rbitnet_core_cuda_managed_peak_bytes']/2**20 for x in r['rows']if x['mode']==mode)})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
table='\n'.join('| '+x['mode']+f" | {x['median_prefill_ms']:.1f} | {x['median_decode_tps']:.2f} | {x['median_http_ms']:.1f} | {x['final_kv_plus_prefix_mib']:.2f} |"for x in summary)
counts=re.findall(r'test result: ok\. (\d+) passed; 0 failed; (\d+) ignored',(proof/'workspace.log').read_text(encoding='utf-8'));passed=sum(int(a)for a,b in counts);ignored=sum(int(b)for a,b in counts)
speed=next(x for x in summary if x['mode']=='paged')['median_decode_tps']/next(x for x in summary if x['mode']=='dense')['median_decode_tps']-1
readme=f'''# KV device paginé F32 Llama — 4 octobre 2026

Le chemin optionnel `RBITNET_CUDA_KV_PAGE_LIMIT=128` remplace les deux allocations KV denses du pipeline natif Llama par un pool de pages physiques de 32 tokens. L'attention normale et split-KV lisent ces pages ; les forwards, blocs, vérifications spéculatives, graphes CUDA et snapshots utilisent la même table persistante. Le format reste F32 et le défaut reste dense.

Les snapshots partagent les pages immuables. Écrire dans une page encore référencée par un snapshot ou un autre contexte déclenche une copie avant écriture. Le pool réutilise les pages libérées et refuse les admissions dépassant sa limite ou le plafond CUDA géré. Les pages physiques et tables sont comptées dans `kv_state`, une seule fois par allocation ; les snapshots partagés ne créent pas une seconde charge `prefix`. Le budget logique du cache de snapshots reste conservateur.

## Mesures du binaire intégré

Même Llama-3.2-1B Q4_K_M, RTX 4080 SUPER, 24 notes communes, contexte 2048, sortie 128, plafond 12 Gio/marge 256 Mio, split-KV, blocs de 128. Trois cycles par mode : le premier est la chauffe ; la table utilise quatre observations longues des deux cycles suivants. Les réponses et prompts exacts sont conservés.

| Mode | Préremplissage ms | Décodage tok/s | HTTP ms | KV + snapshots à la fin du protocole, Mio |
|---|---:|---:|---:|---:|
{table}

La dernière colonne vient des catégories absolues après tout le corpus, y compris les vérifications SSE/sampling/arrêt ; ce n'est pas la moyenne des quatre requêtes longues. `summary.json` distingue mémoire live médiane de ces requêtes, pic géré et catégories finales. La mémoire du pilote et la RAM hôte sont conservées séparément dans les captures brutes. Les allocations hôtes F32 existantes ne sont pas supprimées par ce lot.

Le décodage paginé varie de {speed*100:+.2f} % face au dense dans cette ablation. Cette mesure borne une réduction de mémoire ; elle ne justifie pas de remplacer le dense par défaut ni une supériorité générale de tok/s. Aucun nouveau comparatif Ollama/llama.cpp n'est effectué ici.

## Correction et cycle de vie observés

- {passed} tests workspace réussis, {ignored} ignoré(s), Clippy, build CUDA multi-SM et CLI release réussis ; les avertissements existants sont visibles.
- Deux fixtures GGUF exécutées explicitement dans deux processus, split-KV 0 puis 1. Graphes 0/1, 1/4/8 contextes partageant un préfixe, page partielle à 33 tokens, branchements/COW, blocs et vérification multi-positions : logits F32 bit à bit identiques au dense de la même variante.
- Les refus d'un snapshot dense d'un autre contexte, d'un pool avec autre propriétaire de poids ou variante d'attention/TF32 sont vérifiés. Un peer et un snapshot peuvent survivre au contexte racine ; les allocations restent chargées une seule fois puis sont libérées.
- Limite de deux pages : refus d'une nouvelle écriture, libération puis reprise, sans dépasser la limite physique. La génération du propriétaire empêche la réutilisation accidentelle d'une adresse de contexte.
- 36 observations, 12 paires unary/SSE et quatre arrêts exacts sur le même binaire/DLL. Deux suites serveur, blocs 0/1 : déconnexion/reprise, glouton, seed/pénalités, arrêt et quatre requêtes concurrentes sérialisées.

Les 1/4/8 contextes de la fixture mesurent le partage/cycle de vie, pas un forward multi-séquences. Le serveur utilise encore un seul runtime de génération. Les formats F16/Q8, pages Qwen/GPT/MLA, le KV CPU/SSD et le continuous batching restent dans leurs lots distincts (#93, #94, #96). #92 reste ouvert pour ces architectures et l'intégration serveur multi-séquences.

## Provenance et reproduction

Les captures `production/`, `ablation/` et `live/` appartiennent à la version intégrée. Les captures `prototype-*` appartiennent au premier prototype et ne sont pas regroupées dans les résultats intégrés. `manifest.json` lie les sources, le binaire et la DLL. Les scripts `reproduction/` documentent les commandes et variables ; les captures publiées normalisent uniquement BOM/CRLF et conservent leurs empreintes originales.

Ancienne DLL, backend CPU ou KV paginé hôte avec l'option native demandée : refus explicite plutôt qu'un repli silencieux en dense. L'ABI dense existante reste disponible ; l'ABI paginée reçoit explicitement les variantes pour éviter une divergence de configuration Rust/CRT sous Windows.
'''
(out/'README.md').write_text(readme,encoding='utf-8',newline='\n');(out/'.gitattributes').write_text('*.log -text -whitespace\n*.json -text -whitespace\n',encoding='utf-8',newline='\n')
(out/'manifest.json').write_text(json.dumps({'production':m,'workspace_passed':passed,'workspace_ignored':ignored,'original_capture_sha256':original,'published_capture_sha256':published},indent=2)+'\n',encoding='utf-8',newline='\n')
doc=root/'docs/PERFORMANCE_CACHE_STACK.md';s=doc.read_text(encoding='utf-8');addition='Les [pages KV F32 natives Llama](benchmarks/2026-10-04-paged-kv/README.md) partagent les préfixes et copient seulement les pages encore référencées lors des écritures. Le pool et ses tables sont chargés une fois dans le registre physique. Les essais GGUF/serveur et ablations documentent la réduction de mémoire et le coût du décodage ; le dense reste le défaut. Les 1/4/8 contextes de la fixture ne constituent pas du batching GPU.\n\n';assert addition not in s;s=s.replace('## Mesurer et reproduire\n\n','## Mesurer et reproduire\n\n'+addition);doc.write_text(s,encoding='utf-8',newline='\n')
(base/'pr-body.md').write_text(f'''Le pipeline CUDA Llama allouait tout son KV dense et copiait les préfixes dans chaque snapshot. Il propose désormais des pages F32 de 32 tokens, des snapshots immuables partagés et une copie avant écriture lorsqu'une page est encore référencée. Attention/split-KV, blocs, graphes, vérification et rollback utilisent la table du contexte ; la limite physique et le registre CUDA couvrent pages et tables une seule fois.

Validation : {passed} tests workspace, {ignored} ignoré(s), Clippy, CUDA multi-SM/release, deux fixtures réelles dans deux processus split 0/1 avec graphes 0/1 et contextes partagés 1/4/8, propriétaires/refus/lifetimes/reprise ; 36 observations, 12 paires SSE, quatre arrêts, deux suites serveur avec déconnexion/sampling/pénalités/concurrence sérialisée. [Rapport et captures](docs/benchmarks/2026-10-04-paged-kv/README.md).

La pagination reste optionnelle (`RBITNET_CUDA_KV_PAGE_LIMIT`) : décodage {speed*100:+.2f} % dans l'ablation, bénéfice de mémoire publié sans promesse de tok/s. Références #92, #94 et #96 ; les autres architectures, les formats quantifiés et le forward multi-séquences restent ouverts. PR empilée sur la correction contexte/tokenizer.
''',encoding='utf-8',newline='\n')
print('Published adopted page memory/performance/quality proofs:',len(published),'captures;',passed,'workspace tests.')
