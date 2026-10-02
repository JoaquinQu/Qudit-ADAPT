#!/usr/bin/env bash
# QAOA con reinicios aleatorios en varias profundidades p (n = 5..10), para la
# figura estilo paper: mediana y rango intercuartil contra número de parámetros.
# Usa los núcleos 2-4 de cada bloque de L3; los núcleos 0-1 son de la etapa 3.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/Qudit-ADAPT/.venv/bin/python
NUC="2,3,4,10,11,12,18,19,20,26,27,28,34,35,36,42,43,44,50,51,52,58,59,60"
for p in 35 30 25 20 15 12 10 8 6 5 4 3 2 1; do
  echo "p = $p  $(date)"
  $PY cluster/qaoa_benchmark_mwnp.py --n 10 9 8 7 6 5 --ids 0-19 --p $p --reinicios 10 \
      --procesos 24 --nucleos $NUC --carpeta resultados/qaoa_barrido/p$p > resultados/logs/barrido_p$p.log 2>&1
done
echo "fin $(date)"
