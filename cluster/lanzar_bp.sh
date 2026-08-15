#!/bin/bash
# =====================================================================
# Lanza el estudio de entrenabilidad en un servidor SIN scheduler
# (tipo BitWit: se entra por SSH y los trabajos son procesos sueltos).
#
#   bash cluster/lanzar_bp.sh              # las 4 configuraciones
#   bash cluster/lanzar_bp.sh 3            # sólo la 3 (K6, l=2)
#   bash cluster/lanzar_bp.sh estado       # ver avance
#   bash cluster/lanzar_bp.sh detener      # detenerlas
#
# Los procesos se lanzan con setsid+nohup, así que sobreviven a cerrar la
# sesión SSH. No hace falta screen ni tmux.
#
# REANUDACIÓN: cada iteración escribe checkpoint. Si algo se corta, volver
# a lanzar el mismo comando continúa donde quedó, con resultado idéntico a
# una corrida sin interrupciones.
# =====================================================================

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p resultados/logs resultados/json

ARCHIVOS=(grafos_comparacion.txt grafos_comparacion.txt grafo_completo_n6.txt grafo_completo_n6.txt)
ELES=(1 2 1 2)
NOMBRES=(fig1a_l1 fig1a_l2 k6_l1 k6_l2)

MAX_ITER=35
N_RANDOM=100

if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi

# El cuello es overhead de Python, no BLAS: más hilos no acelera (medido
# 1.02x a n=6) y en un servidor compartido sólo molesta a los demás.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

estado() {
    printf "%-10s %-12s %s\n" "corrida" "estado" "última línea"
    for i in 0 1 2 3; do
        n="${NOMBRES[$i]}"; log="resultados/logs/$n.log"
        if [ ! -f "$log" ]; then est="sin lanzar"; ult=""
        elif grep -qa "Resultado guardado" "$log"; then est="TERMINADA"; ult=$(grep -a "Operadores en el ansatz" "$log" | tail -1)
        elif pgrep -f "output ${n}_bp100.json" >/dev/null; then est="corriendo"; ult=$(tail -1 "$log")
        else est="DETENIDA"; ult=$(tail -1 "$log"); fi
        printf "%-10s %-12s %s\n" "$n" "$est" "$(echo "$ult" | cut -c1-70)"
    done
}

detener() {
    for n in "${NOMBRES[@]}"; do pkill -f "output ${n}_bp100.json" 2>/dev/null; done
    echo "Señal de término enviada. Los checkpoints quedan intactos."
}

lanzar() {
    local i="$1" n="${NOMBRES[$i]}"
    if pgrep -f "output ${n}_bp100.json" >/dev/null; then
        echo "  $n ya está corriendo, no se relanza"; return
    fi
    setsid nohup "$PY" -u cluster/main_bp.py \
        --input_file "${ARCHIVOS[$i]}" --grafo 1 --l "${ELES[$i]}" \
        --max_iteration "$MAX_ITER" --n_random "$N_RANDOM" \
        --seed 0 --n_jobs 1 --output "${n}_bp100.json" \
        >> "resultados/logs/$n.log" 2>&1 < /dev/null &
    echo "  $n lanzada (${ARCHIVOS[$i]}, l=${ELES[$i]})"
}

case "${1:-todas}" in
    estado)  estado ;;
    detener) detener ;;
    todas)   echo "Lanzando las 4:"; for i in 0 1 2 3; do lanzar "$i"; done
             echo; echo "Ver avance:  bash cluster/lanzar_bp.sh estado" ;;
    [0-3])   lanzar "$1" ;;
    *)       echo "Uso: $0 [todas|0|1|2|3|estado|detener]"; exit 1 ;;
esac
