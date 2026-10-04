# Évolution des performances Rbitnet

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
