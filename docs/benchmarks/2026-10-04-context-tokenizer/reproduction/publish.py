"""Publish adopted context/SentencePiece validation, including failed assumptions."""
from pathlib import Path
import hashlib,json,re,shutil
root=Path.cwd();base=root/'target/context-tokenizer';draft=root/'target/performance-cache'
proof=base/'production-proof';out=root/'docs/benchmarks/2026-10-04-context-tokenizer'
assert 'Adopted context capacity, SentencePiece, immutable tokenizers, actual Mistral and HTTP validation passed.'in(draft/'context-production-chain.log').read_text(encoding='utf-8-sig')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((proof/'manifest.json').read_text(encoding='utf-8'))
assert all(sha(root/p)==h for p,h in m['source_sha256'].items())
out.mkdir(parents=True,exist_ok=True);original={};published={}
def capture(src,dest):
 p=out/dest;p.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes();original[dest]=hashlib.sha256(raw).hexdigest()
 p.write_text(raw.decode('utf-8-sig').replace('\r\n','\n'),encoding='utf-8',newline='\n');published[dest]=sha(p)
for folder,label in [(proof,'production'),(base/'live','live'),(draft/'context-real-proof','prototype'),(draft/'context-live-proof','prototype-live')]:
 for p in folder.iterdir():
  if p.is_file()and p.suffix in ['.json','.log']:capture(p,label+'/'+p.name)
for name in ['run_context_live.py','check_context_production.py','check_sentencepiece_real.py','model-inputs-manifest.json']:
 p=draft/name
 if p.exists():capture(p,'reproduction/'+p.name)
capture(base/'publish.py','reproduction/publish.py')
for name in ['context-production-chain.log','context-real-chain.log','context-network-initial-concurrency1.log']:
 p=draft/name
 if p.exists():capture(p,'logs/'+name)
sp=(proof/'actual-sp.log').read_text(encoding='utf-8');assert '8 passed; 0 failed'in sp and 'ACTUAL_SP_MISTRAL backend=Cpu:'in sp and 'ACTUAL_SP_MISTRAL backend=Cuda:'in sp
live=json.loads((base/'live/results.json').read_text(encoding='utf-8'))
assert {x['backend']for x in live['cases']}=={'cpu','gpu'}and all(len(x['errors'])==6 and x['sse_done']and x['tokenizer_file_moved_before_first_generation']and len(x['concurrent_requests'])==4 for x in live['cases'])
workspace=(proof/'workspace.log').read_text(encoding='utf-8');counts=re.findall(r'test result: ok\. (\d+) passed; 0 failed; (\d+) ignored',workspace)
passed=sum(int(a)for a,b in counts);ignored=sum(int(b)for a,b in counts)
readme=f'''# Capacité allouée et tokenizers immuables — 4 octobre 2026

La capacité annoncée est désormais celle du runtime réellement configuré, distincte du contexte d'entraînement GGUF. Elle est figée avant le chargement paresseux des poids. Les requêtes `prompt + max_tokens` dépassant cette capacité reçoivent une erreur HTTP 400 JSON avant l'ouverture du SSE, sur chat/completions OpenAI et messages Anthropic. Les valeurs nulles, négatives, non numériques et les débordements sont refusés.

Les executors chargent une seule instance immuable du tokenizer partagée par `Arc` : comptage et génération utilisent le même propriétaire, même si le fichier est déplacé après chargement. Un nouveau chargement relit et valide son propre fichier. Cela retire les reparsings de comptage et les divergences avant l'initialisation des poids ; aucun gain de tok/s n'est revendiqué pour ce lot.

Le codec `tokenizer.model` conserve les octets protobuf, le normalizer original, les champs inconnus et la correspondance du vocabulaire GGUF. Il respecte les politiques BOS/EOS du modèle et préserve les contrôles visibles au milieu d'une conversation. La copie de décodage visible change seulement le type des pièces CONTROL en USER_DEFINED. Les modèles WORD qui ne peuvent reconnaître les contrôles intérieurs sont refusés explicitement. Les modèles, IDs, types et dimensions incompatibles sont refusés.

## Validation observée sur les sources intégrées

- {passed} tests workspace réussis, {ignored} ignoré(s), Clippy et CLI release réussis ; logs conservés dans `production/`. Les avertissements préexistants restent visibles.
- Huit tests SentencePiece exécutés explicitement en release. Mistral-7B-Instruct-v0.1 Q4_K_M réel, 32 000 pièces vérifiées, CPU puis CUDA ; par backend, six générations HF de référence, douze générations/replays SP sur les mêmes IDs et six générations/replays multi-tours SP. Paris est présent et aucun caractère de remplacement n'est admis.
- Modèle SentencePiece entraîné upstream de 1 000 pièces : IDs connus, normalisation, quatre politiques BOS/EOS et contrôles intérieurs confrontés au processeur original. Les tests sans fixtures externes sont également conservés.
- Serveur réseau Mistral CPU et CUDA : capacité 64, six refus par backend sur les trois routes en stream/non-stream, frontière exacte, déplacement du fichier avant la première génération, réponse unary/SSE identique et quatre requêtes concurrentes sérialisées identiques. Métadonnées, réponses et métriques brutes : `live/results.json`.
- Petit GGUF Llama F32 généré par le test serveur : capacité 16 figée, changement d'environnement à 4 avant la première allocation, dépassement puis reprise, nouveau chargement à 4 et refus du tokenizer remplacé par un fichier invalide. Ce modèle sert à vérifier la capacité, pas à mesurer les performances.

## Limites et échecs conservés

Le JSON HF converti de Mistral applique une politique différente sur certains espaces initiaux (`Metaspace` prepend-first contre dummy-prefix SentencePiece). Deux différences du corpus sont explicitement rapportées ; on ne force pas une égalité artificielle avec ce JSON lorsque les IDs diffèrent. Les générations HF/SP sont comparées sur les entrées dont les IDs concordent, tandis que la normalisation SP est confrontée au processeur original. L'hypothèse initiale erronée est conservée dans les logs du prototype.

Le premier essai réseau demandait quatre appels concurrents avec admission serveur par défaut à un : les 503 étaient le comportement attendu de cette configuration. L'essai final fixe explicitement l'admission à quatre ; le moteur exécute encore les requêtes une par une. Ce lot ne constitue pas un forward GPU multi-séquences ni du continuous batching. Le batching et le draft GGUF restent dans #96 et #97 ; les autres capacités du ticket #24 ne sont pas déclarées terminées.

## Reproduction et provenance

`cargo test --workspace -- --test-threads=1`, `cargo clippy --workspace --all-targets` et `cargo build --release -p rbitnet-cli` sont les vérifications ordinaires. Les scripts de `reproduction/` documentent les variables des tests Mistral, du modèle entraîné, de la DLL CUDA et les routes réseau. Les fichiers GGUF/tokenizers restent locaux ; leurs révisions et empreintes figurent dans les manifestes. Les scripts reflètent l'arborescence de travail et utilisent `target/performance-cache` pour les preuves.

Le manifeste de production lie les sources Rust, le CLI et la DLL utilisée. La suite réseau choisit exclusivement une DLL dont le SHA-256 correspond à ce manifeste. Les captures publiées normalisent seulement BOM/CRLF ; `manifest.json` conserve aussi leurs empreintes originales. Les résultats du prototype sont séparés des résultats intégrés.
'''
(out/'README.md').write_text(readme,encoding='utf-8',newline='\n')
(out/'.gitattributes').write_text('*.log -text -whitespace\n*.json -text -whitespace\n',encoding='utf-8',newline='\n')
(out/'manifest.json').write_text(json.dumps({'production':m,'workspace_passed':passed,'workspace_ignored':ignored,'original_capture_sha256':original,'published_capture_sha256':published},indent=2)+'\n',encoding='utf-8',newline='\n')
doc=root/'docs/PERFORMANCE_CACHE_STACK.md';s=doc.read_text(encoding='utf-8')
addition='Les [preuves de capacité et tokenizer](benchmarks/2026-10-04-context-tokenizer/README.md) distinguent le contexte GGUF de la capacité allouée et refusent les requêtes trop longues avant SSE. Le tokenizer immuable est partagé entre comptage et génération ; Mistral SentencePiece est confronté au processeur original et au JSON HF sur les mêmes IDs, sur CPU/CUDA et serveur réel. Ces corrections ne constituent pas un forward multi-séquences.\n\n'
assert addition not in s;s=s.replace('## Mesurer et reproduire\n\n','## Mesurer et reproduire\n\n'+addition)
doc.write_text(s,encoding='utf-8',newline='\n')
(base/'pr-body.md').write_text(f'''Les requêtes pouvaient annoncer le contexte d'entraînement au lieu de la capacité allouée, puis échouer après ouverture du SSE. Le runtime fige désormais sa capacité avant le chargement paresseux et vérifie prompt + sortie avant admission sur les routes OpenAI et Anthropic. Le comptage et la génération partagent un tokenizer immuable chargé une seule fois.

Le codec SentencePiece conserve le normalizer protobuf original, valide vocabulaire/IDs/dimensions GGUF et reconnaît les contrôles multi-tours avec BOS/EOS cohérents. Les différences de politique d'espaces du JSON HF sont documentées.

Validation : {passed} tests workspace, {ignored} ignoré(s), Clippy, CLI release, huit tests SentencePiece explicites dont Mistral réel CPU/CUDA et modèle entraîné upstream, puis serveur Mistral CPU/CUDA (trois routes, frontière/dépassement avant SSE, tokenizer déplacé avant première génération, unary/SSE et concurrence sérialisée). Données brutes et SHA-256 dans [le rapport](docs/benchmarks/2026-10-04-context-tokenizer/README.md).

Références #24 et #96. La PR est empilée sur #111. Le forward multi-séquences et les autres capacités de #24 restent ouverts ; aucun gain de débit n'est annoncé.
''',encoding='utf-8',newline='\n')
print('Published adopted context/tokenizer proof:',passed,'workspace tests;',len(published),'captures.')
