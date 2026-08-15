"""
Qudit-ADAPT on the benchmark graphs: Figs. 1-3 and the ADAPT columns of Table I.

Runs the algorithm on the four irregular instances G_1..G_4 of
`datos/grafos_comparacion.txt` and on the regular ones of
`datos/grafos_regulares.txt`, for both pool truncations l = 1 and l = 2.

Writes a summary CSV and a full JSON with the energy trace, the selected
operators and the optimal parameters of every run, into `resultados/`. The
figures themselves are drawn later by `cuadernillos/comparacion_QAOA.ipynb`,
which overlays these curves with the QAOA baseline from `main_QAOA.py`.

Configuration is the block of constants below rather than command-line flags:
this is meant to be launched once and left alone.

    python cluster/main_comparaciones.py
"""

from pathlib import Path
import sys
import ast
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funciones.utilidades import cd_adapt_vqe_algorithm_profundo, to_jsonable


# ============================================================
# PARÁMETROS GENERALES
# ============================================================

N = 6
EPSILON = 1e-2
MAX_ITERATION = 40


# ============================================================
# LEER ARGUMENTOS
# Uso:
# python3 main_comparaciones.py l [grafos_file]
#
# Ejemplo:
# python3 main_comparaciones.py 1
# python3 main_comparaciones.py 1 grafos_regulares.txt
# ============================================================

if len(sys.argv) not in (2, 3):
    raise ValueError(
        "Uso: python3 main_comparaciones.py <l> [grafos_file]\n"
        "Ejemplo: python3 main_comparaciones.py 1\n"
        "         python3 main_comparaciones.py 1 grafos_regulares.txt"
    )

l = int(sys.argv[1])
GRAFOS_FILE = sys.argv[2] if len(sys.argv) == 3 else "grafos_comparacion.txt"

if l not in [1, 2]:
    raise ValueError("El parámetro l debe ser 1 o 2.")

# El nombre de salida se deriva del archivo de grafos, para que coincida
# automáticamente con lo que esperan los cuadernillos de análisis
# (grafos_comparacion.txt -> "..._comparacion_...", grafos_regulares.txt -> "..._regulares_...").
LABEL = Path(GRAFOS_FILE).stem
if LABEL.startswith("grafos_"):
    LABEL = LABEL[len("grafos_"):]


# ============================================================
# RUTAS
# ============================================================

DATOS_DIR = PROJECT_ROOT / "datos"
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
CSV_DIR = RESULTADOS_DIR / "csv"
JSON_DIR = RESULTADOS_DIR / "json"

CSV_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

input_path = DATOS_DIR / GRAFOS_FILE

if not input_path.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de grafos en: {input_path}\n"
        f"Revisa que '{GRAFOS_FILE}' exista dentro de {DATOS_DIR}."
    )

output_csv_path = CSV_DIR / f"resultados_resumen_{LABEL}_n{N}_l{l}.csv"
output_json_path = JSON_DIR / f"resultados_completos_{LABEL}_n{N}_l{l}.json"


# ============================================================
# FUNCIÓN PARA LEER CADA GRAFO
# ============================================================

def parse_graph_line(linea):
    """
    Permite leer grafos escritos de dos formas:

    Forma 1:
        (1,2), (1,3), (2,3)

    Forma 2:
        [(1,2), (1,3), (2,3)]
    """

    linea = linea.strip()

    if linea.startswith("["):
        edges = ast.literal_eval(linea)
    else:
        edges = ast.literal_eval("[" + linea + "]")

    return [(int(i), int(j)) for i, j in edges]


# ============================================================
# LEER GRAFOS
# ============================================================

with open(input_path, "r", encoding="utf-8") as f:
    lineas = f.readlines()

grafos = []

for linea in lineas:
    linea = linea.strip()

    if not linea:
        continue

    if linea.startswith("#"):
        continue

    grafos.append(parse_graph_line(linea))

print("=" * 70)
print(f"Ejecución profunda de grafos ({LABEL})")
print(f"N = {N}")
print(f"l = {l}")
print(f"epsilon = {EPSILON}")
print(f"max_iteration = {MAX_ITERATION}")
print(f"Archivo de grafos = {input_path}")
print(f"Número de grafos leídos = {len(grafos)}")
print("=" * 70)


# ============================================================
# CAMPOS DEL CSV
# ============================================================

csv_fieldnames = [
    "grafo_id",
    "n",
    "l",
    "num_edges",

    "initial_energy",
    "initial_problem_energy",
    "ground_energy",
    "first_excited_energy",
    "spectral_gap",
    "ground_degeneracy",

    "final_energy",
    "difference_ground",

    "iterations",
    "num_ansatz_ops",
    "pool_size",

    "final_gradient_norm",
    "final_max_gradient",

    "converged",
    "stop_reason",

    "optimizer_success",
    "optimizer_status",
    "optimizer_nfev",
    "optimizer_njev",
    "optimizer_nit",

    "runtime_min",
]


# ============================================================
# EJECUTAR GRAFOS Y GUARDAR RESULTADOS
# ============================================================

json_results = []

with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
    writer.writeheader()

    for grafo_id, edges in enumerate(grafos, start=1):

        print()
        print("=" * 70)
        print(f"Ejecutando grafo {grafo_id}/{len(grafos)}")
        print(f"Aristas: {edges}")
        print("=" * 70)

        try:
            result = cd_adapt_vqe_algorithm_profundo(
                n=N,
                edges=edges,
                l=l,
                epsilon=EPSILON,
                max_iteration=MAX_ITERATION,
                show=True
            )

            result_complete = {
                "grafo_id": grafo_id,
                **result
            }

            json_results.append(to_jsonable(result_complete))

            row_csv = {
                "grafo_id": grafo_id,
                "n": result["n"],
                "l": result["l"],
                "num_edges": result["num_edges"],

                "initial_energy": result["initial_energy"],
                "initial_problem_energy": result["initial_problem_energy"],
                "ground_energy": result["ground_energy"],
                "first_excited_energy": result["first_excited_energy"],
                "spectral_gap": result["spectral_gap"],
                "ground_degeneracy": result["ground_degeneracy"],

                "final_energy": result["final_energy"],
                "difference_ground": result["difference_ground"],

                "iterations": result["iterations"],
                "num_ansatz_ops": result["num_ansatz_ops"],
                "pool_size": result["pool_size"],

                "final_gradient_norm": result["final_gradient_norm"],
                "final_max_gradient": result["final_max_gradient"],

                "converged": result["converged"],
                "stop_reason": result["stop_reason"],

                "optimizer_success": result["optimizer_success"],
                "optimizer_status": result["optimizer_status"],
                "optimizer_nfev": result["optimizer_nfev"],
                "optimizer_njev": result["optimizer_njev"],
                "optimizer_nit": result["optimizer_nit"],

                "runtime_min": result["runtime_min"],
            }

            writer.writerow(row_csv)

            # Guardado parcial después de cada grafo.
            # Esto es importante para el cluster.
            with open(output_json_path, "w", encoding="utf-8") as jf:
                json.dump(json_results, jf, indent=4, ensure_ascii=False)

            print(f"Grafo {grafo_id} terminado y guardado parcialmente.")

        except Exception as e:
            print(f"ERROR en grafo {grafo_id}: {e}")

            error_result = {
                "grafo_id": grafo_id,
                "edges": edges,
                "error": str(e),
            }

            json_results.append(error_result)

            writer.writerow({
                "grafo_id": grafo_id,
                "n": N,
                "l": l,
                "num_edges": len(edges),

                "initial_energy": "ERROR",
                "initial_problem_energy": "ERROR",
                "ground_energy": "ERROR",
                "first_excited_energy": "ERROR",
                "spectral_gap": "ERROR",
                "ground_degeneracy": "ERROR",

                "final_energy": "ERROR",
                "difference_ground": "ERROR",

                "iterations": "ERROR",
                "num_ansatz_ops": "ERROR",
                "pool_size": "ERROR",

                "final_gradient_norm": "ERROR",
                "final_max_gradient": "ERROR",

                "converged": "ERROR",
                "stop_reason": "ERROR",

                "optimizer_success": "ERROR",
                "optimizer_status": "ERROR",
                "optimizer_nfev": "ERROR",
                "optimizer_njev": "ERROR",
                "optimizer_nit": "ERROR",

                "runtime_min": "ERROR",
            })

            with open(output_json_path, "w", encoding="utf-8") as jf:
                json.dump(json_results, jf, indent=4, ensure_ascii=False)


# ============================================================
# GUARDADO FINAL
# ============================================================

with open(output_json_path, "w", encoding="utf-8") as jf:
    json.dump(json_results, jf, indent=4, ensure_ascii=False)

print()
print("=" * 70)
print("Ejecución finalizada.")
print(f"CSV guardado en:  {output_csv_path}")
print(f"JSON guardado en: {output_json_path}")
print("=" * 70)