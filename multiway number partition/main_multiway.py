"""
main_multiway.py

Corre CD-ADAPT-VQE (ver utilidades/utilidades_multiway.py, formulacion_qubo.tex) sobre
una lista de instancias de multiway number partitioning (k=3), una por
linea de un archivo generado con generar_instancias_multiway (ver
datos/casos_n5.txt, datos/casos_n6.txt). Guarda un CSV con lo mas
importante y un JSON con la traza completa por cada instancia -- mismo
patron que cluster/main_factorizacion.py en la raiz del proyecto.

Uso tipico desde esta carpeta ("multiway number partition/"):

    python main_multiway.py --instancias_file datos/casos_n5.txt --l 1
    python main_multiway.py --instancias_file datos/casos_n6.txt --l 2

Para correr en background en el cluster:

    nohup python3 main_multiway.py --instancias_file datos/casos_n6.txt --l 1 \
        > logs/multiway_n6_l1.log 2>&1 &

Aviso sobre --l 2: el pool O1+O3 crece muy rapido. Para n=5 ya da 1755
operadores (~10s armarlo, ~30s la corrida completa); para n=6 va a ser
bastante mas grande -- probar primero con una sola instancia antes de
lanzar las 5 de una vez si se corre l=2 para n=6.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# ============================================================
# Asegurar que el proyecto (y esta carpeta) esten en el path
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from funciones.utilidades import to_jsonable  # noqa: E402
from utilidades.utilidades_multiway import (  # noqa: E402
    Hp_multiway,
    build_pool_multiway,
    initial_state,
    cd_adapt_vqe_multiway,
    top_partitions,
    leer_instancias_multiway,
)


# ============================================================
# Utilidades locales
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Corre CD-ADAPT-VQE de multiway number partitioning sobre una lista de instancias."
    )

    parser.add_argument(
        "--instancias_file", type=str, required=True,
        help="Archivo .txt con una instancia por linea (enteros separados "
             "por coma), generado con generar_instancias_multiway. Ej.: "
             "datos/casos_n5.txt",
    )
    parser.add_argument(
        "--l", type=int, default=1, choices=[1, 2],
        help="Orden del pool: l=1 -> O1, l=2 -> O1+O3 (ver build_pool_multiway; "
             "a diferencia de factorizacion, aca solo se admiten 1 o 2). Default: 1",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3,
                         help="Umbral de convergencia (norma del gradiente). Default: 1e-3")
    parser.add_argument("--max_iteration", type=int, default=30,
                         help="Maximo de operadores en el ansatz. Default: 30")
    parser.add_argument("--quiet", action="store_true",
                         help="Reduce impresion en pantalla.")
    parser.add_argument("--output_csv", type=str, default=None,
                         help="Ruta del CSV de salida. Default automatico en resultados/csv/")
    parser.add_argument("--output_json", type=str, default=None,
                         help="Ruta del JSON de salida. Default automatico en resultados/json/")

    return parser.parse_args()


def read_instancias(args):
    path = Path(args.instancias_file)
    if not path.is_absolute():
        path = SCRIPT_DIR / args.instancias_file

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de instancias: {path}")

    return leer_instancias_multiway(path)


def build_default_output_paths(args, instancias):
    resultados_dir = SCRIPT_DIR / "resultados"
    csv_dir = resultados_dir / "csv"
    json_dir = resultados_dir / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    ns = sorted({len(a) for a in instancias})
    n_tag = "-".join(str(n) for n in ns)
    base_name = f"resultados_multiway_l{args.l}_n{n_tag}_{len(instancias)}casos"

    output_csv = Path(args.output_csv) if args.output_csv else csv_dir / f"{base_name}.csv"
    output_json = Path(args.output_json) if args.output_json else json_dir / f"{base_name}.json"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    return output_csv, output_json


def run_one(a, l, epsilon, max_iteration, show):
    t0 = time.time()
    n = len(a)

    Hf = Hp_multiway(n, a)
    operator_pool, operator_pool_labels = build_pool_multiway(n, a, l=l, show=show)
    psi_0, initial_ground_energy = initial_state(n)

    result = cd_adapt_vqe_multiway(
        Hf=Hf,
        psi_0=psi_0,
        operator_pool=operator_pool,
        operator_pool_labels=operator_pool_labels,
        epsilon=epsilon,
        max_iteration=max_iteration,
        show=show,
    )

    # top_k=3**n: se guardan TODOS los estados computacionales (no solo el
    # top-5), mismo criterio que top_candidates en main_factorizacion.py --
    # para no tener que reconstruir psi_final si despues hace falta la
    # probabilidad de una particion que no quedo entre las mas probables.
    candidates = top_partitions(result["psi_final"], n, a, top_k=3 ** n)
    best = candidates[0]

    tol = 1e-6
    found_solution = bool(result["difference_ground"] < tol)

    del result["psi_final"]  # no es serializable, ya extrajimos lo que hace falta
    result.update({
        "a": a,
        "n": n,
        "l": l,
        "mu": sum(a) / 3,
        "best_digits": list(best["digits"]),
        "best_sumas": list(best["sumas"]),
        "best_imbalance": best["imbalance"],
        "best_prob": best["prob"],
        "found_solution": found_solution,
        "top_candidates": candidates,
        "initial_energy": initial_ground_energy,
        "total_runtime_min": (time.time() - t0) / 60.0,
    })
    return result


CSV_FIELDNAMES = [
    "a", "n", "l", "mu", "pool_size",
    "epsilon", "max_iteration", "iterations", "num_ansatz_ops",
    "final_energy", "ground_energy", "difference_ground", "final_gradient_norm",
    "found_solution", "best_imbalance", "best_prob",
    "optimizer_success", "runtime_min", "total_runtime_min",
]


def to_csv_row(result):
    row = {k: result.get(k) for k in CSV_FIELDNAMES}
    row["a"] = str(result.get("a"))
    return row


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    show = not args.quiet

    instancias = read_instancias(args)
    output_csv, output_json = build_default_output_paths(args, instancias)

    print("=" * 90)
    print("CD-ADAPT-VQE PARA MULTIWAY NUMBER PARTITIONING (k=3)")
    print("=" * 90)
    print(f"instancias_file = {args.instancias_file}  ({len(instancias)} instancias)")
    print(f"l               = {args.l}")
    print(f"epsilon         = {args.epsilon}")
    print(f"max_iteration   = {args.max_iteration}")
    print(f"output_csv      = {output_csv}")
    print(f"output_json     = {output_json}")
    print("=" * 90)

    global_start = time.time()
    all_json_results = []
    errors = []

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for idx, a in enumerate(instancias, start=1):
            print("\n" + "#" * 90)
            print(f"Instancia {idx}/{len(instancias)}: a = {a}  (n={len(a)})")
            print("#" * 90)

            try:
                result = run_one(a, args.l, args.epsilon, args.max_iteration, show)

                writer.writerow(to_csv_row(result))
                csvfile.flush()

                all_json_results.append(to_jsonable(result))
                with open(output_json, "w", encoding="utf-8") as jf:
                    json.dump(all_json_results, jf, indent=4, ensure_ascii=False)

                estado = "OK (imbalance optimo)" if result["found_solution"] else "no optimo"
                print(f"\na={a}: mejor particion sumas={result['best_sumas']} "
                      f"imbalance={result['best_imbalance']:.4f} prob={result['best_prob']:.4f}  [{estado}]")

            except Exception as exc:
                error_item = {"a": a, "error": str(exc)}
                errors.append(error_item)
                print(f"\nERROR en a={a}: {exc}")

    if errors:
        error_path = output_json.with_name(output_json.stem + "_errors.json")
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(errors), f, indent=4, ensure_ascii=False)
        print(f"Errores guardados en: {error_path}")

    runtime_min = (time.time() - global_start) / 60.0

    print("\n" + "=" * 90)
    print("FIN")
    print("=" * 90)
    print(f"Tiempo total: {runtime_min:.2f} min")
    print(f"CSV final:  {output_csv}")
    print(f"JSON final: {output_json}")

    n_ok = sum(1 for r in all_json_results if r.get("found_solution"))
    print(f"\nOptimas encontradas: {n_ok}/{len(all_json_results)}")


if __name__ == "__main__":
    main()
