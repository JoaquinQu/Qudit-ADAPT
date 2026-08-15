"""
The 300-random-graph ensemble behind Fig. 4.

Generates M random six-vertex graphs with a fixed seed and runs Qudit-ADAPT on
each with both pool truncations, so the two can be compared over a population
of instances rather than on hand-picked examples.

The paper reports the common-budget variant: every instance is capped at
k_max = 50 ADAPT iterations instead of running to its own convergence
threshold. That is deliberate. Stopping each run at its own convergence point
would compare the two pools at different ansatz sizes, and the interesting
question is what l = 2 buys you at equal ansatz growth -- which turns out to be
nothing for the first ~30 iterations, and everything after.

    python cluster/main.py

Set MAX_ITERATION to None to run each instance to convergence instead.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funciones.utilidades import generar_m_grafos, ejecutar_grafos

# =========================
# PARÁMETROS GENERALES
# =========================

N = 6
M = 300
MIN_EDGES = None
MAX_EDGES = None
SEED = 42

EPSILON = 1e-2
MAX_ITERATION = 50

GRAFOS_FILE = "grafos_n6.txt"

# =========================
# LEER ARGUMENTOS
# Uso:
# python3 main.py start_idx end_idx l
# Ejemplo:
# python3 main.py 1 20 1
# =========================

if len(sys.argv) != 4:
    raise ValueError(
        "Uso: python3 main.py <start_idx> <end_idx> <l>\n"
        "Ejemplo: python3 main.py 1 20 1"
    )

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])
l = int(sys.argv[3])

if l not in [1, 2]:
    raise ValueError("El parámetro l debe ser 1 o 2.")

# =========================
# GENERAR GRAFOS SOLO SI NO EXISTE EL ARCHIVO
# =========================

ruta_grafos = PROJECT_ROOT / "datos" / GRAFOS_FILE

if not ruta_grafos.exists():
    generar_m_grafos(
        n=N,
        m=M,
        min_edges=MIN_EDGES,
        max_edges=MAX_EDGES,
        seed=SEED,
        filename=GRAFOS_FILE
    )
    print(f"Grafos generados y guardados en '{ruta_grafos}'.")
else:
    print(f"El archivo '{ruta_grafos}' ya existe. Se reutilizarán esos grafos.")

# =========================
# NOMBRES DE SALIDA
# =========================

output_csv = f"resultados_resumen_n6_l{l}_{start_idx}_{end_idx}_bfgs1000.csv"
output_json = f"resultados_completos_n6_l{l}_{start_idx}_{end_idx}_bfgs1000.json"

# =========================
# EJECUTAR SOLO EL BLOQUE PEDIDO
# =========================
print("Emepezando ejecución")
ejecutar_grafos(
    input_file=GRAFOS_FILE,
    output_csv=output_csv,
    output_json=output_json,
    n=N,
    l=l,
    epsilon=EPSILON,
    max_iteration=MAX_ITERATION,
    show=True,
    start_idx=start_idx,
    end_idx=end_idx
)

print(
    f"Resultados de grafos {start_idx}-{end_idx} para l={l} "
    f"guardados en '{output_csv}' y '{output_json}'."
)
