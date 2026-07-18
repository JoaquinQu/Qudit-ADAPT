"""
Ejecuta CD-ADAPT-VQE en grafos completos K_n para n = 4..10.

Uso:
    python run_kn_cluster.py --l 1
    python run_kn_cluster.py --l 2

Opciones adicionales:
    --n_min    primer valor de n  (default: 4)
    --n_max    último valor de n  (default: 10)
    --epsilon  umbral de convergencia (default: 1e-2)
    --max_iter máximo de iteraciones ADAPT (default: 50)
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rutas del proyecto  (el script vive en la raíz del proyecto)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = PROJECT_ROOT / "resultados"
CSV_DIR = RESULTADOS_DIR / "csv"
JSON_DIR = RESULTADOS_DIR / "json"

CSV_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Importar utilidades del proyecto
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))
from funciones.utilidades import cd_adapt_vqe_algorithm, to_jsonable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def complete_graph_edges(n: int):
    return list(combinations(range(1, n + 1), 2))


def max_3_cut_value(n: int) -> int:
    q, r = divmod(n, 3)
    sizes = [q + 1] * r + [q] * (3 - r)
    return int((n**2 - sum(s**2 for s in sizes)) // 2)


def exact_ground_energy(n: int) -> float:
    return float(-2 * max_3_cut_value(n))


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CD-ADAPT-VQE en grafos completos K_n")
    parser.add_argument("--l",        type=int,   required=True,  choices=[1, 2],
                        help="Orden del pool de operadores (1 o 2)")
    parser.add_argument("--n_min",    type=int,   default=4,      help="n mínimo (default: 4)")
    parser.add_argument("--n_max",    type=int,   default=10,     help="n máximo (default: 10)")
    parser.add_argument("--epsilon",  type=float, default=1e-2,   help="Umbral de convergencia (default: 1e-2)")
    parser.add_argument("--max_iter", type=int,   default=50,     help="Máximo de iteraciones (default: 50)")
    args = parser.parse_args()

    l         = args.l
    n_values  = list(range(args.n_min, args.n_max + 1))
    epsilon   = args.epsilon
    max_iter  = args.max_iter

    csv_path  = CSV_DIR  / f"kn_l{l}_n{args.n_min}-{args.n_max}.csv"
    json_path = JSON_DIR / f"kn_l{l}_n{args.n_min}-{args.n_max}.json"

    csv_fieldnames = [
        "n", "num_edges", "E0_exacta", "E_final", "diferencia",
        "ratio_E_E0", "iteraciones", "ops_ansatz", "pool_size",
        "grad_final_norm", "runtime_min",
        "optimizer_success", "optimizer_message"
    ]

    all_json = []

    log(f"Iniciando experimento  l={l}  |  n={n_values}  |  epsilon={epsilon}  |  max_iter={max_iter}")
    log(f"CSV  -> {csv_path}")
    log(f"JSON -> {json_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        csvfile.flush()

        for n in n_values:
            edges   = complete_graph_edges(n)
            E0_exact = exact_ground_energy(n)

            log(f"{'='*60}")
            log(f"Comenzando  K_{n}  |  l={l}  |  aristas={len(edges)}  |  E0_exacta={E0_exact}")
            log(f"{'='*60}")

            try:
                res = cd_adapt_vqe_algorithm(
                    n             = n,
                    edges         = edges,
                    l             = l,
                    epsilon       = epsilon,
                    max_iteration = max_iter,
                    show          = True
                )

                ratio = (res["final_energy"] / E0_exact
                         if E0_exact != 0 else float("nan"))

                row = {
                    "n"                 : n,
                    "num_edges"         : len(edges),
                    "E0_exacta"         : E0_exact,
                    "E_final"           : res["final_energy"],
                    "diferencia"        : res["difference_ground"],
                    "ratio_E_E0"        : ratio,
                    "iteraciones"       : res["iterations"],
                    "ops_ansatz"        : res["num_ansatz_ops"],
                    "pool_size"         : res["pool_size"],
                    "grad_final_norm"   : res["final_gradient_norm"],
                    "runtime_min"       : res["runtime_min"],
                    "optimizer_success" : res["optimizer_success"],
                    "optimizer_message" : res["optimizer_message"],
                }

                writer.writerow(row)
                csvfile.flush()

                json_entry = {"n": n, "l": l, **to_jsonable(res)}
                all_json.append(json_entry)

                # Sobrescribe el JSON completo después de cada n
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(all_json, jf, indent=4, ensure_ascii=False)

                log(f"K_{n} terminado | E_final={res['final_energy']:.6f} | "
                    f"ratio={ratio:.4f} | iters={res['iterations']} | "
                    f"runtime={res['runtime_min']:.2f} min")

            except Exception as e:
                log(f"ERROR en K_{n}: {e}")
                traceback.print_exc()

                writer.writerow({
                    "n": n, "num_edges": len(edges),
                    "E0_exacta": E0_exact,
                    "E_final": "ERROR", "diferencia": "ERROR",
                    "ratio_E_E0": "ERROR", "iteraciones": "ERROR",
                    "ops_ansatz": "ERROR", "pool_size": "ERROR",
                    "grad_final_norm": "ERROR", "runtime_min": "ERROR",
                    "optimizer_success": "ERROR",
                    "optimizer_message": str(e),
                })
                csvfile.flush()

                all_json.append({"n": n, "l": l, "error": str(e)})
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(all_json, jf, indent=4, ensure_ascii=False)

    log("Experimento finalizado.")
    log(f"CSV  -> {csv_path}")
    log(f"JSON -> {json_path}")


if __name__ == "__main__":
    main()
