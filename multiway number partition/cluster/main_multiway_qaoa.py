"""
main_multiway_qaoa.py

Corre QAOA (ver utilidades/utilidades_multiway.py) sobre una lista de instancias de
multiway number partitioning (k=3), una por linea de un archivo generado con
generar_instancias_multiway (ver datos/casos_n5.txt, datos/casos_n6.txt).
Barre profundidad p=1..p_max, guarda un CSV resumen y un JSON con la traza
completa (incluye todos los restarts de cada p) por cada instancia -- mismo
patron que main_multiway.py (CD-ADAPT-VQE) y cluster/main_QAOA.py (QAOA para
Max-3-Cut).

Defaults = lo que se valido en comparacion_multiway.ipynb: method="CMA-ES",
sin warm-start entre profundidades (cada p corre 25 reinicios totalmente
independientes). Ahi se probo primero L-BFGS-B y rendia mal -- paisaje de
(gamma,beta) muy oscilante porque H_p tiene rango de energia grande -- y se
cambio a CMA-ES siguiendo la recomendacion de Deller et al. (arXiv:2204.00340,
Ap. B.2) para este tipo de paisaje multimodal.

Uso tipico desde "multiway number partition/":

    python cluster/main_multiway_qaoa.py --instancias_file datos/casos_n5.txt --p_max 10

Para correr solo una instancia (1-indexado, inclusive) -- util para paralelizar
lanzando un proceso por instancia:

    python cluster/main_multiway_qaoa.py --instancias_file datos/casos_n6.txt --p_max 10 \
        --start_idx 3 --end_idx 3

Para correr en background en el cluster:

    nohup python3 cluster/main_multiway_qaoa.py --instancias_file datos/casos_n6.txt --p_max 10 \
        > logs/qaoa_n6.log 2>&1 &
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# Asegurar que el proyecto, "multiway number partition/" (utilidades/) y
# esta carpeta esten en el path
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MULTIWAY_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MULTIWAY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MULTIWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIWAY_ROOT))

from funciones.utilidades import to_jsonable  # noqa: E402
from utilidades.utilidades_multiway import (  # noqa: E402
    Hp_multiway,
    scan_qaoa_p_multiway,
    leer_instancias_multiway,
)


# ============================================================
# Utilidades locales
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Corre QAOA de multiway number partitioning sobre una lista de instancias."
    )

    parser.add_argument(
        "--instancias_file", type=str, required=True,
        help="Archivo .txt con una instancia por linea (enteros separados "
             "por coma), generado con generar_instancias_multiway. Ej.: "
             "datos/casos_n5.txt",
    )
    parser.add_argument("--p_max", type=int, default=10,
                         help="Profundidad maxima QAOA. Default: 10")
    parser.add_argument("--mixer", type=str, default="jx", choices=["jx", "custom"],
                         help="Mixer a usar. Default: jx")
    parser.add_argument("--method", type=str, default="CMA-ES",
                         help="'CMA-ES' o un metodo de scipy.optimize.minimize "
                              "(ej. 'L-BFGS-B'). Default: CMA-ES")
    parser.add_argument("--num_restarts", type=int, default=25,
                         help="Reinicios aleatorios por cada p. Default: 25")
    parser.add_argument("--maxiter", type=int, default=500,
                         help="CMA-ES: maxfevals por restart. scipy: maxiter. Default: 500")
    parser.add_argument("--bounds_scale", type=float, default=float(np.pi),
                         help="Cota de parametros: [-bounds_scale, bounds_scale]. Default: pi")
    parser.add_argument("--seed", type=int, default=7,
                         help="Semilla aleatoria base. Default: 7")
    parser.add_argument("--use_warmstart", action="store_true",
                         help="Activa warm-start entre p y p+1 (default: desactivado, "
                              "protocolo de Deller et al. -- cada p corre independiente).")
    parser.add_argument("--start_idx", type=int, default=1,
                         help="Indice inicial de instancia, empezando desde 1. Default: 1")
    parser.add_argument("--end_idx", type=int, default=None,
                         help="Indice final de instancia, inclusivo. Default: todas")
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
        path = MULTIWAY_ROOT / args.instancias_file

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de instancias: {path}")

    return leer_instancias_multiway(path)


def select_instance_range(instancias, start_idx, end_idx):
    if start_idx < 1:
        raise ValueError("start_idx debe ser >= 1")

    if end_idx is None:
        end_idx = len(instancias)

    if end_idx < start_idx:
        raise ValueError("end_idx debe ser >= start_idx")

    if end_idx > len(instancias):
        raise ValueError(
            f"end_idx={end_idx} excede la cantidad de instancias: {len(instancias)}"
        )

    return [(idx, instancias[idx - 1]) for idx in range(start_idx, end_idx + 1)]


def build_default_output_paths(args, instancias_seleccionadas):
    resultados_dir = MULTIWAY_ROOT / "resultados"
    csv_dir = resultados_dir / "csv"
    json_dir = resultados_dir / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    ns = sorted({len(a) for _, a in instancias_seleccionadas})
    n_tag = "-".join(str(n) for n in ns)
    end_label = "all" if args.end_idx is None else str(args.end_idx)
    range_label = f"{args.start_idx}_{end_label}"
    base_name = (
        f"resultados_multiway_qaoa_n{n_tag}_p{args.p_max}_"
        f"{args.mixer}_{args.method}_r{args.num_restarts}_casos_{range_label}"
    )

    output_csv = Path(args.output_csv) if args.output_csv else csv_dir / f"{base_name}.csv"
    output_json = Path(args.output_json) if args.output_json else json_dir / f"{base_name}.json"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    return output_csv, output_json


CSV_FIELDNAMES = [
    "instancia_idx", "a", "n", "mu",
    "p", "num_params", "mixer", "method",
    "ground_energy", "initial_problem_energy", "initial_mixer_energy",
    "best_energy", "absolute_error", "relative_error",
    "num_restarts", "maxiter", "used_warmstart", "runtime_min",
]


def to_csv_row(item):
    row = {k: item.get(k) for k in CSV_FIELDNAMES}
    row["a"] = str(item.get("a"))
    return row


def run_one(instancia_idx, a, args, show):
    n = len(a)
    Hc = Hp_multiway(n, a)

    results_p = scan_qaoa_p_multiway(
        n=n,
        Hc=Hc,
        p_max=args.p_max,
        mixer=args.mixer,
        num_restarts=args.num_restarts,
        maxiter=args.maxiter,
        seed=args.seed + 1000 * instancia_idx,
        bounds_scale=args.bounds_scale,
        method=args.method,
        use_warmstart=args.use_warmstart,
        show=show,
    )

    mu = sum(a) / 3.0
    for item in results_p:
        item["instancia_idx"] = int(instancia_idx)
        item["a"] = list(a)
        item["mu"] = float(mu)
        item["num_restarts"] = int(args.num_restarts)
        item["maxiter"] = int(args.maxiter)
        item["used_warmstart"] = bool(args.use_warmstart)

    return results_p


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    show = not args.quiet

    instancias = read_instancias(args)
    seleccionadas = select_instance_range(instancias, args.start_idx, args.end_idx)
    output_csv, output_json = build_default_output_paths(args, seleccionadas)

    print("=" * 90)
    print("QAOA PARA MULTIWAY NUMBER PARTITIONING (k=3)")
    print("=" * 90)
    print(f"instancias_file = {args.instancias_file}  ({len(seleccionadas)}/{len(instancias)} seleccionadas)")
    print(f"p_max           = {args.p_max}")
    print(f"mixer           = {args.mixer}")
    print(f"method          = {args.method}")
    print(f"num_restarts    = {args.num_restarts}")
    print(f"maxiter         = {args.maxiter}")
    print(f"use_warmstart   = {args.use_warmstart}")
    print(f"output_csv      = {output_csv}")
    print(f"output_json     = {output_json}")
    print("=" * 90)

    global_start = time.time()
    all_json_results = []
    errors = []

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for instancia_idx, a in seleccionadas:
            print("\n" + "#" * 90)
            print(f"Instancia {instancia_idx}: a = {a}  (n={len(a)})")
            print("#" * 90)

            try:
                t0 = time.time()
                results_p = run_one(instancia_idx, a, args, show)
                runtime_min = (time.time() - t0) / 60.0

                for row in results_p:
                    writer.writerow(to_csv_row(row))
                csvfile.flush()

                all_json_results.extend(to_jsonable(results_p))
                with open(output_json, "w", encoding="utf-8") as jf:
                    json.dump(all_json_results, jf, indent=4, ensure_ascii=False)

                best_p = results_p[-1]
                print(f"\nInstancia {instancia_idx} (a={a}): "
                      f"error_abs(p={best_p['p']}) = {best_p['absolute_error']:.6f}  "
                      f"[{runtime_min:.2f} min]")

            except Exception as exc:
                error_item = {"instancia_idx": instancia_idx, "a": a, "error": str(exc)}
                errors.append(error_item)
                print(f"\nERROR en instancia {instancia_idx} (a={a}): {exc}")

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


if __name__ == "__main__":
    main()
