"""Publish adopted async-cache evidence without mixing prototype measurements."""
from pathlib import Path
import hashlib,json,re,statistics
root=Path.cwd();base=root/'target/async-expert-cache';draft=root/'target/performance-cache';proof=base/'production-proof'
out=root/'docs/benchmarks/2026-10-04-async-expert-cache'
assert 'Adopted async expert cache workspace, actual GPU/GGUF, quiet, network, lifecycle and overlap suites passed.'in(draft/'async-production-chain.log').read_text(encoding='utf-8-sig')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((proof/'manifest.json').read_text(encoding='utf-8'))
assert all(sha(root/p)==h for p,h in m['source_sha256'].items())
assert sha(proof/'rbitnet.exe')==m['binary_sha256']
out.mkdir(parents=True,exist_ok=True);original={};published={}
def capture(src,dest):
    p=out/dest;p.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes();original[dest]=hashlib.sha256(raw).hexdigest()
    p.write_text(raw.decode('utf-8-sig').replace('\r\n','\n'),encoding='utf-8',newline='\n');published[dest]=sha(p)
for folder,label in [(proof,'production'),(base/'ablation','ablation'),(base/'live','live'),(base/'lifecycle','lifecycle'),
                     (draft/'async-current-proof','prototype'),(draft/'async-ablation','prototype-ablation'),(draft/'async-live','prototype-live'),(draft/'async-lifecycle','prototype-lifecycle')]:
    for p in folder.rglob('*'):
        if p.is_file()and p.suffix in ['.json','.log']:capture(p,label+'/'+p.relative_to(folder).as_posix())
for name in ['check_async_production.py','check_async_current.py','run_async_lifecycle.py','profile_async_current.py','analyze_async_trace.py']:
    capture(draft/name,'reproduction/'+name)
for name in ['async_benchmark.py','async_live.py']:capture(draft/'followup-harness'/name,'reproduction/'+name)
for p in(base/'reproduction').glob('*.py'):capture(p,'reproduction/adopted-'+p.name)
capture(base/'publish.py','reproduction/publish.py');capture(draft/'async-production-chain.log','logs/async-production-chain.log')
summary=[]
def summarize(path,kind,model,budget):
    r=json.loads(path.read_text(encoding='utf-8'))
    assert len(r['rows'])==27 and len(r['sse'])==9 and len(r['stops'])==3 and all(x['matches_baseline']for x in r['rows'])
    for mode in ['sync','async-demand','async-previous-pass']:
        rows=[x for x in r['rows']if x['mode']==mode and x['cycle']>0 and x['prompt']<2]
        tps=[x['response']['usage']['completion_tokens']*1000/x['metrics_delta']['rbitnet_inference_decode_ms_sum']for x in rows]
        summary.append(dict(kind=kind,model=model,cache_mib=budget,mode=mode,measured_long_rows=len(rows),
            median_prefill_ms=statistics.median(x['metrics_delta']['rbitnet_inference_prefill_ms_sum']for x in rows),
            median_decode_tps=statistics.median(tps),min_decode_tps=min(tps),max_decode_tps=max(tps),
            median_http_ms=statistics.median(x['wall_ms']for x in rows)))
for model in ['gpt-oss-20b','glm47-flash']:
    summarize(base/'ablation'/model/'results.json','adopted',model,8192)
    for budget in [512,8192]:summarize(draft/'async-ablation'/(model+'-cache'+str(budget))/'results.json','prototype',model,budget)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
def table(kind):
    return '\n'.join(f"| {x['model']} | {x['cache_mib']} | {x['mode']} | {x['median_prefill_ms']:.1f} | {x['median_decode_tps']:.2f} | {x['median_http_ms']:.1f} |"for x in summary if x['kind']==kind)
counts=re.findall(r'test result: ok\. (\d+) passed; 0 failed; (\d+) ignored',(proof/'workspace.log').read_text(encoding='utf-8'))
passed=sum(int(a)for a,b in counts);ignored=sum(int(b)for a,b in counts)
assert 'nine actual admission checks passed'in(proof/'actual-config-refusals.log').read_text(encoding='utf-8')
trace=json.loads((proof/'trace-analysis.json').read_text(encoding='utf-8'))
groups=[v for v in trace['h2d_groups'].values()if v['copy_compute_overlap_ns']>0]
assert groups and trace['child_completion']['same_output']and trace['child_completion']['async_failed']==0
overlap=max(groups,key=lambda v:v['h2d_bytes'])
readme=f'''# Cache d'experts asynchrone GPT-OSS/GLM — 4 octobre 2026

Le cache optionnel garde des groupes d'experts READY et PENDING dans un seul pool physique borné. Les octets quantifiés GGUF restent inchangés. Un flux CUDA privé copie depuis une ou deux zones RAM épinglées ; un groupe ne devient visible qu'après la réussite de son événement de fin. Les experts sélectionnés et les copies en cours conservent leurs propriétaires jusqu'à la fin du calcul. Le déchargement draine les événements avant de libérer ces buffers.

`RBITNET_MOE_ASYNC=1` active cette variante avec `RBITNET_MOE_EXECUTION=cache`, CUDA/hybrid et un budget d'experts positif. `RBITNET_MOE_PINNED_SLOTS=1|2` borne la RAM épinglée. `RBITNET_MOE_PREFETCH=off|previous-pass` compare la demande seule aux experts observés à la couche suivante lors de la passe précédente. Il s'agit d'une heuristique ; les prédicteurs entraînés de [Mira](https://arxiv.org/abs/2609.38090) restent une étude séparée. Les combinaisons incompatibles et valeurs invalides demandées explicitement sont refusées. Le défaut reste synchrone, sans préchargement.

## Mesures intégrées

RTX 4080 SUPER 16 Gio, mêmes GGUF/tokenizers que le manifeste de référence, contexte 2048, plafond CUDA 12 Gio/marge 256 Mio. Huit notes communes, sortie 32 tokens, trois cycles : le premier chauffe chaque mode. La table utilise quatre observations longues des cycles suivants ; `summary.json` conserve aussi les bornes min/max du débit. Le protocole intégral conserve le prompt court, les réponses, métriques, températures, RAM/VRAM et erreurs éventuelles. GPT utilise son graphe segmenté ; GLM utilise MLA/split-KV. Aucun résultat de débit ne provient de Nsight.

| Modèle | Cache Mio | Mode | Préremplissage ms | Décodage tok/s | HTTP ms |
|---|---:|---|---:|---:|---:|
{table('adopted')}

Ces mesures portent sur le binaire intégré à 8192 Mio. Elles ne justifient pas un changement automatique du défaut. Le cache asynchrone réserve des slots ayant la capacité maximale des groupes de projections ; les allocations physiques, les payloads READY/PENDING et les octets utiles sont mesurés séparément. Un format mixte Q4/Q6 peut donc avoir une réserve supérieure aux seuls payloads utiles. Les métriques exposent préchargements demandés/utilisés/inutiles/tardifs, copies, octets gaspillés, attente, slots et RAM épinglée.

## Première ablation, prototype distinct

Même protocole, mais binaire/DLL et empreintes distincts des mesures intégrées. La comparaison des budgets a précédé l'adoption. Ces lignes ne sont pas agrégées aux lignes intégrées.

| Modèle | Cache Mio | Mode | Préremplissage ms | Décodage tok/s | HTTP ms |
|---|---:|---|---:|---:|---:|
{table('prototype')}

Le petit cache crée beaucoup de rechargements. L'augmentation de recouvrement ne garantit donc pas une baisse de latence : demande asynchrone et préchargement peuvent aider ou ralentir selon le modèle et le budget. Les timings courts ne démontrent pas la parité avec Ollama/llama.cpp ; aucun nouveau comparatif de ces moteurs n'est effectué ici.

## Correction, mémoire et serveur observés

- {passed} tests workspace réussis, {ignored} ignoré(s), Clippy et CLI release ; trois fixtures CUDA explicites vérifient événements/publication, protections READY/PENDING, budget unique, pointeurs stables, drainage et refill des octets originaux Q4/Q6. Les warnings sont conservés.
- Neuf refus de configuration exécutés sur le vrai GGUF GPT-OSS ; toutes les catégories CUDA restent inchangées et le chemin CPU désactivé reste chargeable.
- GPT-OSS et GLM, budgets 512/8192 Mio, références/variants réelles : teacher forcing des logits, générations avec seed/pénalités, préfixes, événements/prefetch et durée de vie des modèles. Tolérances et maxima sont dans les logs dédiés. La préférence `previous-pass` peut être inutile avec un grand cache ; aucune obligation artificielle d'obtenir un hit n'est imposée.
- Sur le binaire intégré : 54 observations, 18 paires unary/SSE et six arrêts exacts ; deux suites réseau à 512 Mio testent sampling, seed/pénalités, déconnexion/reprise et quatre requêtes concurrentes sérialisées.
- Deux cycles serveur de déchargement/rechargement à 512 Mio : budgets READY/PENDING/pool/pinned bornés, registre de modèles vide et toutes catégories libérées sauf scratch persistant ; requête sur modèle déchargé refusée, nouveau propriétaire après rechargement et même sortie.
- Nsight capture une seconde génération chaude GPT à 512 Mio, après sortie identique au témoin et nonce de fin validé : {overlap['h2d_copies']} copies H2D privées, {overlap['h2d_bytes']} octets ; {overlap['copies_overlapping_other_stream_compute']} copies croisent effectivement des kernels sur un autre flux du même GPU/contexte. L'intersection fusionnée vaut {overlap['copy_compute_overlap_ns']/1e9:.3f} s, soit {overlap['fraction_copy_time_overlapped']*100:.2f} % de la durée cumulée de ces copies. Le rapport de trace distingue temps cumulés et unions ; aucun débit n'en est déduit.

Les captures brutes et l'analyse de trace sont publiées avec leurs empreintes. Les lourds fichiers SQLite/NSYS restent locaux, leurs SHA sont dans l'analyse. L'absence d'allocation chaude, lorsqu'observée, ne porte que sur la génération capturée. Les requêtes concurrentes restent sérialisées : ce lot ne réalise pas le batching GPU. Une injection d'erreur matérielle CUDA n'a pas été réalisée.

## Reproduction et portée

`manifest.json` lie les sources intégrées, le binaire, la DLL et chaque capture. Les `.log`/`.json` ne normalisent que BOM/CRLF ; leurs SHA originaux sont conservés. Les commandes et scripts historiques sont archivés sous `reproduction/`. Pour répéter les mesures après checkout, construire la CLI de cette révision, placer les deux harnesses sous `target/performance-cache/followup-harness/`, puis passer `--binary`, `--library`, `--async`, les chemins GGUF/tokenizer du manifeste, `--moe-cache 512|8192`, les mêmes notes/sorties/cycles. Les générateurs de prototype sont des archives de la préparation précédente, pas des migrations à réappliquer sur ces sources déjà intégrées.

#84 reste ouvert pour l'étude séparée des prédicteurs entraînés et des autres architectures. #85 traite le choix de politique ; #86 conserve le placement mixte par expert ; #98 conserve le comparatif final des moteurs. Le préchargement n'altère jamais les IDs du routeur pour augmenter artificiellement les hits.
'''
(out/'README.md').write_text(readme,encoding='utf-8',newline='\n')
(out/'.gitattributes').write_text('*.log -text -whitespace\n*.json -text -whitespace\n',encoding='utf-8',newline='\n')
(out/'manifest.json').write_text(json.dumps(dict(production=m,workspace_passed=passed,workspace_ignored=ignored,original_capture_sha256=original,published_capture_sha256=published),indent=2)+'\n',encoding='utf-8',newline='\n')
doc=root/'docs/PERFORMANCE_CACHE_STACK.md';s=doc.read_text(encoding='utf-8')
addition='Le [cache d’experts asynchrone GPT-OSS/GLM](benchmarks/2026-10-04-async-expert-cache/README.md) copie les poids GGUF inchangés avec une RAM épinglée bornée et des événements CUDA. Les groupes READY/PENDING partagent un budget physique ; la demande et le préchargement précédent sont mesurés séparément. Les résultats dépendent du modèle et du budget : le défaut reste synchrone.\n\n'
assert addition not in s;s=s.replace('## Mesurer et reproduire\n\n','## Mesurer et reproduire\n\n'+addition);doc.write_text(s,encoding='utf-8',newline='\n')
s=doc.read_text(encoding='utf-8');line=next(x for x in s.splitlines()if x.startswith('| `RBITNET_MOE_TRACE_DIR='))
rows='| `RBITNET_MOE_ASYNC=1` | Pool asynchrone GPT-OSS/GLM optionnel ; exige CUDA/hybrid, `cache`, budget positif et API dynamique. READY/PENDING/slots libres partagent le plafond physique ; défaut `0`. |\n| `RBITNET_MOE_PINNED_SLOTS=2` | Une ou deux zones RAM épinglées pour les copies, allouées au chargement. Les sources/destinations restent possédées jusqu’aux événements de fin. |\n| `RBITNET_MOE_PREFETCH=off` | `off` ou `previous-pass` avec le cache asynchrone ; prédiction par passe précédente, sans entraînement. Une erreur de prédiction modifie les transferts, jamais le routage. |\n'
assert 'RBITNET_MOE_ASYNC=1'not in s;s=s.replace(line+'\n',line+'\n'+rows)
s=s.replace('sans exécution mixte par expert ni recouvrement de copies.', 'sans exécution mixte par expert. Le recouvrement de copies est une option séparée du mode `cache`, incompatible avec `cpu`/`adaptive`.')
doc.write_text(s,encoding='utf-8',newline='\n')
doc=root/'docs/STUBS_AND_MVP_AUDIT.md';s=doc.read_text(encoding='utf-8')
s=s.replace('Revu le **3 octobre 2026**, après les lots cache, spéculation, Qwen/GPT-OSS résidents et budget CUDA commun.', 'Revu le **4 octobre 2026**, après les lots MLA/GPT segmentés, préremplissage GPT fixe, contexte/tokenizer, pages KV Llama et cache d’experts asynchrone.')
s=s.replace('Cache dynamique GPT-OSS et pipeline MLA : #88/#89.', '[GPT segmenté](benchmarks/2026-10-03-gpt-segmented/README.md) et [MLA résident](benchmarks/2026-10-03-mla-full/README.md) ajoutent des chemins réels à budgets explicites ; autres layouts et variantes restent à valider.')
s=s.replace('Préchargement asynchrone et choix par coût absents (#84/#86) ; des petits caches ralentissent les modèles testés.', '[Cache asynchrone](benchmarks/2026-10-04-async-expert-cache/README.md) et [choix par coût du FFN entier](benchmarks/2026-10-03-moe-placement/README.md) opt-in ; placement mixte par expert encore absent (#86). Des petits caches peuvent ralentir les modèles.')
s=s.replace('GPT-OSS/GLM restent sériels (#95).', '[GPT-OSS à banques fixes](benchmarks/2026-10-04-gpt-block/README.md) calcule de vrais blocs ; les chemins GPT segmenté/cache et GLM restent sériels (#95).')
s=s.replace('Graphes GPU : K/V F32 denses. Pages/formats device : #92/#93.', '[Llama CUDA F32 paginé](benchmarks/2026-10-04-paged-kv/README.md) réel et optionnel ; autres pages/formats device : #92/#93.')
s=s.replace('Manquant/invalide bloque readiness ; templates propres au modèle.', '[SentencePiece, jetons spéciaux et capacité réelle](benchmarks/2026-10-04-context-tokenizer/README.md) validés sur GGUF et HTTP ; tokenizer partagé immuable au chargement. Manquant/invalide bloque readiness ; templates propres au modèle.')
doc.write_text(s,encoding='utf-8',newline='\n')
for target in re.findall(r'\]\((benchmarks/[^)]+)\)',s):
    assert (root/'docs'/target).is_file(),('audit link must resolve',target)
(base/'pr-body.md').write_text(f'''Le chargement des experts manquants bloquait la génération sur des copies synchrones. GPT-OSS et GLM disposent maintenant d’un pool asynchrone optionnel READY/PENDING, de buffers RAM épinglés bornés et d’un flux CUDA avec événements de publication. Les propriétaires sélectionnés ou en copie empêchent toute éviction/réutilisation prématurée. Les octets GGUF et les IDs du routeur restent inchangés.

Validation : {passed} tests workspace, {ignored} ignoré(s), Clippy/release, trois fixtures CUDA, neuf refus de configuration, quatre suites GGUF budgets 512/8192 avec logits/générations/préfixes/lifetimes ; 54 observations intégrées, 18 paires SSE, six arrêts, deux suites réseau et deux cycles déchargement/rechargement. Nsight démontre un recouvrement réel des copies et kernels pendant une génération chaude, séparément des mesures de débit. [Rapport et captures](docs/benchmarks/2026-10-04-async-expert-cache/README.md).

`RBITNET_MOE_ASYNC=1` reste opt-in, `RBITNET_MOE_PREFETCH=off|previous-pass`. Les petits caches montrent des gains et des régressions selon le modèle ; aucune promesse de parité générale. Références #83, #84, #85, #86 et #98 ; les prédicteurs entraînés et le batching GPU restent ouverts. PR empilée sur #113.
''',encoding='utf-8',newline='\n')
print('Published adopted async cache proofs:',len(published),'captures;',passed,'workspace tests.')
