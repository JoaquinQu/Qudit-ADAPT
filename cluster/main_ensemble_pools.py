"""
Comparación de pools contradiabáticos sobre un ensemble de grafos.

Corre CD-ADAPT-VQE sobre cada grafo de un archivo de `datos/`, una vez por
cada base del pool, y guarda las métricas de comparación. Pensado para dejarlo
corriendo desatendido en una workstation:

  - guarda un CSV y un JSON después de CADA corrida, así una interrupción no
    pierde nada;
  - hace `resume` automático: al relanzarlo salta los pares (grafo, base) que
    ya estén en el CSV;
  - se puede repartir por rangos de grafos entre varios procesos.

Ejemplos
--------
    # las tres bases sobre los primeros 50 grafos
    python cluster/main_ensemble_pools.py --start 1 --end 50 --l 2

    # repartir en 4 procesos
    for i in 0 1 2 3; do
      python cluster/main_ensemble_pools.py --start $((i*75+1)) --end $(((i+1)*75)) \
          --l 2 --output ensemble_l2_part$i.csv &
    done

    # sin instancias aleatorias (mucho más rápido): sólo compara convergencia
    python cluster/main_ensemble_pools.py --start 1 --end 300 --n_random 0
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import ast
import csv
import json
import time
import traceback

import numpy as np

from funciones.utilidades_bp import (
    adapt_bp_scan,
    costo_compuertas,
    costo_mediciones,
    peso_operador,
)

DATOS_DIR = PROJECT_ROOT / "datos"
CSV_DIR = PROJECT_ROOT / "resultados" / "csv"
JSON_DIR = PROJECT_ROOT / "resultados" / "json"

CAMPOS = [
    "grafo_id", "base", "n", "num_edges", "l",
    "pool_size", "num_ansatz_ops", "stop_reason",
    "ground_energy", "error_final",
    "costo_compuertas", "costo_mediciones",
    "k_1e-2", "med_1e-2", "comp_1e-2",
    "k_1e-4", "med_1e-4", "comp_1e-4",
    "peso_medio", "runtime_min",
]


def leer_grafos(input_file):
    path = Path(input_file)
    if not path.is_absolute():
        path = DATOS_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return [ast.literal_eval("[" + ln.strip() + "]") for ln in f if ln.strip()]


def hechos(csv_path):
    """Pares (grafo_id, base) ya calculados, para poder reanudar."""
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        return {(int(r["grafo_id"]), r["base"]) for r in csv.DictReader(f)
                if r.get("grafo_id", "").isdigit()}


def umbral(err, med, comp, tol):
    """Primer índice donde el error baja de `tol`, con sus costos asociados."""
    bajo = np.where(err < tol)[0]
    if len(bajo) == 0:
        return "", "", ""
    k = int(bajo[0])
    return k, int(med[min(k, len(med) - 1)]), int(comp[min(k, len(comp) - 1)])


def fila_de(gid, base, res):
    E0 = float(res["ground_energy"])
    err = np.abs(np.asarray(res["recycled_energy"], float) - E0)
    comp = costo_compuertas(res)
    med = costo_mediciones(res)
    pesos = [peso_operador(x) for x in res["ansatz_op_labels"]]

    k2, m2, c2 = umbral(err, med, comp, 1e-2)
    k4, m4, c4 = umbral(err, med, comp, 1e-4)

    return {
        "grafo_id": gid, "base": base, "n": res["n"],
        "num_edges": res["num_edges"], "l": res["l"],
        "pool_size": res["pool_size"], "num_ansatz_ops": res["num_ansatz_ops"],
        "stop_reason": res["stop_reason"],
        "ground_energy": E0, "error_final": float(err[-1]),
        "costo_compuertas": int(comp[-1]), "costo_mediciones": int(med[-1]),
        "k_1e-2": k2, "med_1e-2": m2, "comp_1e-2": c2,
        "k_1e-4": k4, "med_1e-4": m4, "comp_1e-4": c4,
        "peso_medio": float(np.mean(pesos)) if pesos else 0.0,
        "runtime_min": round(float(res["runtime_min"]), 3),
    }


def main():
    p = argparse.ArgumentParser(description="Comparación de pools sobre un ensemble")
    p.add_argument("--input_file", type=str, default="grafos_n6.txt")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--l", type=int, default=2, choices=[1, 2])
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--bases", type=str, nargs="+",
                   default=["angular", "gellmann", "heisenberg"])
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--max_iteration", type=int, default=30)
    p.add_argument("--n_random", type=int, default=0,
                   help="0 = sólo comparar convergencia (mucho más rápido)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    grafos = leer_grafos(args.input_file)
    fin = args.end or len(grafos)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    nombre = args.output or f"ensemble_pools_n{args.n}_l{args.l}_{args.start}-{fin}.csv"
    csv_path = CSV_DIR / nombre
    json_path = JSON_DIR / (Path(nombre).stem + ".json")

    ya = hechos(csv_path)
    if ya:
        print(f"Reanudando: {len(ya)} pares (grafo, base) ya calculados.")

    nuevo = not csv_path.exists()
    filas_json = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            filas_json = json.load(f)

    t_ini = time.time()
    total = (fin - args.start + 1) * len(args.bases)
    hechas = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
            fcsv.flush()

        for gid in range(args.start, fin + 1):
            edges = grafos[gid - 1]

            for base in args.bases:
                hechas += 1
                if (gid, base) in ya:
                    continue

                etiqueta = f"[{hechas}/{total}] grafo {gid} | {base}"
                try:
                    res = adapt_bp_scan(
                        n=args.n, edges=edges, l=args.l,
                        epsilon=args.epsilon, max_iteration=args.max_iteration,
                        n_random=args.n_random, seed=args.seed,
                        show=False, base=base,
                    )
                    fila = fila_de(gid, base, res)
                    w.writerow(fila)
                    fcsv.flush()

                    filas_json.append(fila)
                    with open(json_path, "w", encoding="utf-8") as fj:
                        json.dump(filas_json, fj, indent=2, ensure_ascii=False)

                    print(f"{etiqueta}: pool={fila['pool_size']} "
                          f"ops={fila['num_ansatz_ops']} err={fila['error_final']:.3e} "
                          f"med={fila['costo_mediciones']} "
                          f"({fila['runtime_min']:.1f} min)", flush=True)

                except Exception as exc:
                    print(f"{etiqueta}: ERROR {exc}", flush=True)
                    traceback.print_exc()

    print(f"\nCSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Tiempo total: {(time.time() - t_ini) / 60:.1f} min")


if __name__ == "__main__":
    main()
