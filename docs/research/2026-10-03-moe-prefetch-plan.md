# Préchargement MoE : décision pour #84 et #86

## Sources et décision

[Mira, v1 du 29 septembre 2026](https://arxiv.org/html/2609.38090v1), section IV-E, entraîne hors ligne des prédicteurs par couche à partir des activations, logits de routage et historiques de sélection, pour anticiper la couche i+2. Sa hiérarchie HOT/STAGE et son format INT8 sont couplés. Reproduire son score avec nos GGUF Q4_K/MXFP4 inchangés et une RTX 4080 de 16 Go n'est pas établi.

[Fiddler](https://arxiv.org/html/2402.07033v3), sections 3.2–3.3, compare calcul CPU et transfert des poids puis calcul GPU selon le volume d'entrées par expert. Cette distinction est pertinente pour notre décodage à une entrée : le transfert d'un expert peut coûter plus cher que son calcul CPU.

**Décision locale : GO pour le préchargement simple et le modèle de coût ; NO-GO provisoire pour entraîner un prédicteur neuronal.** Les traces actuelles contiennent les experts choisis, pas les activations/logits nécessaires à son entraînement. Aucun prédicteur Mira n'a été entraîné ou évalué ici. La décision sera réexaminée après mesure du recouvrement, à budget égal, avec les octets GGUF actuels.

## Ordre d'implémentation

1. Un flux CUDA de copie non bloquant et une réserve RAM épinglée bornée. Chaque copie possède un événement et garde vivants source et destination jusqu'à sa complétion. La sélection réelle attend uniquement les événements des experts requis. Une copie inutile ne modifie jamais le routage.
2. Séparer HOT et STAGE en déduisant STAGE du même budget total de VRAM. Commencer par les sélections du passage précédent, une puis deux couches d'avance. Le premier passage fonctionne sans prédiction. Saturation, annulation ou faible confiance réduisent le staging et gardent un chemin CPU correct.
3. Mesurer temps de copie, attente au routeur, copies consommées/inutiles, temps GPU et CPU par format/forme. Comparer les traces aux événements du GPU pour prouver le recouvrement, puis les latences HTTP et le décodage sur GPT-OSS/GLM.
4. Choisir CPU ou GPU avec les coûts mesurés, protéger les experts en cours et sommer les résultats dans l'ordre du routeur. Tester d'abord le mode conservateur par couche, puis la combinaison de certains experts CPU avec les autres GPU. Le calcul CPU concurrent doit partager le budget de threads ; ajouter des workers sans limite serait une régression possible.

## Réexamen des prédicteurs appris

La calibration devra être propre à chaque empreinte de modèle/tokenizer. Séparer les conversations d'entraînement, de calibration et d'évaluation, pour éviter de tester une prédiction sur ses propres traces. Collecter des exemples i→i+1 et i→i+2, avec activations/logits, histogrammes et labels du routeur réel. Le coût de collecte et d'entraînement devra être publié avant de qualifier ce chemin d'amélioration utilisable.

Comparer trois bases : absence de prédiction, sélection précédente et affinité empirique entre couches. Mesurer précision/rappel des experts utiles, octets transférés inutilement, stalls évités, VRAM retirée de HOT, mémoire des prédicteurs et coût de leur exécution. L'acceptation dépendra d'un gain de latence après ces coûts, sur des requêtes nouvelles et plusieurs budgets ; un taux de prédiction seul ne suffit pas.

Les tests de génération doivent couvrir prédictions erronées, arrivées tardives, saturation, annulation pendant copie et rechargement du modèle. Garder les experts sélectionnés et les coefficients du routeur exacts. Le format INT8 spécifique de Mira constituerait une expérimentation distincte avec évaluation de qualité ; il ne fait pas partie de ce préchargement à octets GGUF constants.
