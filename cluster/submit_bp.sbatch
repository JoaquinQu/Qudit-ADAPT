#!/bin/bash
#SBATCH --job-name=qudit-bp
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=resultados/logs/slurm_%x_%a.log
#SBATCH --error=resultados/logs/slurm_%x_%a.log

# =====================================================================
# Estudio de entrenabilidad de Qudit-ADAPT: 4 configuraciones en paralelo
# como job array. Cada tarea del array es independiente.
#
#   0 -> grafo Fig. 1a, l=1        2 -> K6 completo, l=1
#   1 -> grafo Fig. 1a, l=2        3 -> K6 completo, l=2
#
# Uso:
#     sbatch cluster/submit_bp.sbatch
#     sbatch --array=3 cluster/submit_bp.sbatch     # sólo la de K6 l=2
#
# REANUDACIÓN: los scripts guardan checkpoint en cada iteración y retoman
# solos. Si un job muere por walltime, basta re-encolarlo con el mismo
# comando: continúa donde quedó, y el resultado es idéntico a una corrida
# sin cortes (se repone incluso el estado del generador aleatorio).
# =====================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p resultados/logs resultados/json

# --- entorno --------------------------------------------------------
# Ajustar a lo que ofrezca el cluster. Alternativas habituales:
#   module load python/3.11
#   source ~/venvs/qudit/bin/activate
#   conda activate qudit
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

# El cuello de botella es overhead de Python, no BLAS: más hilos no ayuda
# (se midió 1.02x a n=6) y sólo genera contención. Se fija a 1.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- configuraciones del array --------------------------------------
ARCHIVOS=(grafos_comparacion.txt grafos_comparacion.txt grafo_completo_n6.txt grafo_completo_n6.txt)
ELES=(1 2 1 2)
SALIDAS=(fig1a_l1_bp100.json fig1a_l2_bp100.json k6_l1_bp100.json k6_l2_bp100.json)

i="${SLURM_ARRAY_TASK_ID:-0}"

echo "=== tarea $i: ${ARCHIVOS[$i]}, l=${ELES[$i]} -> ${SALIDAS[$i]} ==="
echo "nodo: $(hostname)   inicio: $(date)"

$PY -u cluster/main_bp.py \
    --input_file "${ARCHIVOS[$i]}" \
    --grafo 1 \
    --l "${ELES[$i]}" \
    --max_iteration 35 \
    --n_random 100 \
    --seed 0 \
    --n_jobs 1 \
    --output "${SALIDAS[$i]}"

echo "fin: $(date)"
