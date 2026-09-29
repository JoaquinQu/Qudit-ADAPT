#!/usr/bin/env bash
# Benchmark ADAPT (k <= 80) contra QAOA (p = 40), por etapas y con procesos
# fijados a núcleos.
#
# BitWit tiene 8 bloques de 8 núcleos (bloque b: núcleos 8b..8b+7), cada uno
# con 32 MB de L3. A n = 12 el estado pesa 8.5 MB y el rendimiento lo decide
# ese caché: un proceso por bloque corre casi a velocidad libre, y dos se pisan.
# A n = 11 rinden dos por bloque. Ver cluster/calibrar_concurrencia.py.
#
#   etapa 1: ADAPT n = 10, 11  +  QAOA n = 5..11  (+ INTERP n = 5..9, liviano)
#   etapa 2: n = 12, ADAPT en 4 bloques y QAOA en los otros 4
#   etapa 3: INTERP n = 10..12
#
# Cada lanzador salta lo que ya existe, así que el script se puede relanzar.
#
#   setsid nohup bash cluster/lanzar_k80_qaoa.sh > resultados/logs/etapas.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/Qudit-ADAPT/.venv/bin/python}
L=resultados/logs
mkdir -p $L

UNO="0,8,16,24,32,40,48,56"      # un núcleo por bloque
DOS="1,9,17,25,33,41,49,57"      # un segundo núcleo por bloque
ADAPT="cluster/benchmark_mwnp.py --l 1 2 --ids 0-1999 --instancias datos/mwnp_ordenes_10a12.json \
       --carpeta resultados/mwnp_k80 --max_iteration 80 --barrido soporte --hilos 1"
QAOA="cluster/qaoa_benchmark_mwnp.py --ids 0-19 --p 40 --reinicios 10 --carpeta resultados/qaoa_p40"

echo "etapa 1: ADAPT n = 10, 11 | QAOA n = 5..11 | INTERP n = 5..9   $(date)"
$PY $ADAPT --n 10 11 --procesos 8 --nucleos $UNO > $L/etapa1_adapt.log 2>&1 &
$PY $QAOA --n 5 6 7 8 9 10 11 --procesos 8 --nucleos $DOS > $L/etapa1_qaoa.log 2>&1 &
$PY $QAOA --n 5 6 7 8 9 --solo_interp --procesos 4 --nucleos 2,18,34,50 > $L/etapa1_interp.log 2>&1 &
wait

echo "etapa 2: n = 12   $(date)"
$PY $ADAPT --n 12 --procesos 4 --nucleos 0,8,16,24 > $L/etapa2_adapt.log 2>&1 &
$PY $QAOA --n 12 --procesos 4 --nucleos 32,40,48,56 > $L/etapa2_qaoa.log 2>&1 &
wait

echo "etapa 3: INTERP n = 10..12   $(date)"
$PY $QAOA --n 12 --solo_interp --procesos 8 --nucleos $UNO > $L/etapa3_interp12.log 2>&1
$PY $QAOA --n 10 11 --solo_interp --procesos 16 --nucleos $UNO,$DOS > $L/etapa3_interp1011.log 2>&1

echo "fin   $(date)"
