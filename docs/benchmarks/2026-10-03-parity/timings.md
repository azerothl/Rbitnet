# Phases et mesures complémentaires

| Modèle | Moteur | Backend | Décodage tok/s | Préremplissage médian ms | HTTP médian ms | Premier contenu SSE ms | RSS pic Gio | VRAM pic Δ Mio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama32-1b | rbitnet | cpu | 38.83 | 782.00 | 1676.52 | 863.21 | 1.33 | 0 |
| llama32-1b | ollama | cpu | 52.45 | 50.09 | 670.54 | 56.24 | 0.93 | 0 |
| llama32-1b | llama.cpp | cpu | 50.57 | 41.77 | 680.83 | 49.14 | 1.36 | 0 |
| llama32-1b | rbitnet | gpu | 477.61 | 58.00 | 142.56 | 74.92 | 1.85 | 1603 |
| llama32-1b | ollama | gpu | 487.00 | 44.97 | 141.00 | 73.76 | 0.68 | 1111 |
| llama32-1b | llama.cpp | gpu | 528.61 | 4.91 | 79.48 | 18.08 | 1.15 | 1140 |
| qwen35-2b | rbitnet | cpu | 16.53 | 1973.00 | 3908.27 | 2053.59 | 2.20 | 1 |
| qwen35-2b | ollama | cpu | 20.24 | 192.44 | 1801.77 | 260.92 | 2.11 | 5 |
| qwen35-2b | llama.cpp | cpu | 20.07 | 188.36 | 1797.46 | 213.95 | 2.18 | 1 |
| qwen35-2b | rbitnet | gpu | 56.14 | 666.00 | 1255.07 | 675.51 | 4.33 | 2526 |
| qwen35-2b | ollama | gpu | 216.49 | 63.66 | 238.48 | 57.67 | 1.05 | 2244 |
| qwen35-2b | llama.cpp | gpu | 238.68 | 16.25 | 171.70 | 42.75 | 2.37 | 2263 |
| gpt-oss-20b | rbitnet | cpu | 12.28 | 6076.00 | 8758.19 | 7239.24 | 10.15 | 5 |
| gpt-oss-20b | ollama | cpu | 17.10 | 859.21 | 2739.59 | 872.74 | 11.06 | 26 |
| gpt-oss-20b | llama.cpp | cpu | 17.64 | 858.78 | 2677.84 | 876.65 | 20.28 | 2 |
| gpt-oss-20b | rbitnet | gpu | 58.29 | 1512.00 | 2069.46 | 1821.66 | 21.30 | 11838 |
| gpt-oss-20b | ollama | gpu | 202.03 | 163.42 | 351.92 | 188.80 | 1.04 | 11100 |
| gpt-oss-20b | llama.cpp | gpu | 203.98 | 47.41 | 207.50 | 70.18 | 11.10 | 11151 |
| glm47-flash | rbitnet | cpu | 8.78 | 3418.00 | 7066.56 | 3404.56 | 14.26 | 0 |
| glm47-flash | ollama | cpu | 15.69 | 375.00 | 2423.65 | 415.70 | 17.28 | 1 |
| glm47-flash | llama.cpp | cpu | 13.76 | 409.74 | 2749.80 | 415.60 | 29.16 | 11 |
| glm47-flash | rbitnet | gpu | 14.19 | 2048.00 | 4305.91 | 2000.06 | 28.60 | 14381 |
| glm47-flash | ollama | gpu | 69.72 | 195.86 | 685.37 | 189.97 | 4.52 | 13994 |
| glm47-flash | llama.cpp | gpu | 76.50 | 87.41 | 509.45 | 108.88 | 16.71 | 14170 |

Les frontières des phases diffèrent selon les moteurs. Le probe SSE est une requête distincte de 16 tokens. La mémoire inclut le chargement et les fluctuations du bureau ; ces valeurs ne sont pas une mesure isolée des allocations du moteur.
