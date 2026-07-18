"""
main_QAOA.py

Ejecuta QAOA/QOAO para qudits sobre grafos guardados en un archivo .txt,
guarda un CSV con la información más importante y un JSON con toda la información.

Uso típico desde la raíz del proyecto:

    python3 main_QAOA.py --n 6 --p_max 15 --num_restarts 25 --maxiter 500

Para ejecutar solo un rango de grafos:

    python3 main_QAOA.py --n 6 --start_idx 1 --end_idx 4 --p_max 15

Para correr en background:

    nohup python3 main_QAOA.py --n 6 --p_max 15 --num_restarts 25 --maxiter 500 > logs/qaoa.log 2>&1 &
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# Asegurar que el proyecto esté en el path
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funciones.utilidades_QAOA import (  # noqa: E402
    leer_grafos,
    scan_qaoa_p,
    qaoa_results_to_dataframe,
    to_jsonable,
)


# ============================================================
# Utilidades locales
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta QAOA/QOAO para qudits sobre varios grafos."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=str(PROJECT_ROOT / "datos" / "grafos_regulares.txt"),
        help="Archivo .txt con los grafos. Default: datos/grafos_regulares.txt",
    )

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Número de nodos/qutrits de los grafos.",
    )

    parser.add_argument(
        "--p_max",
        type=int,
        default=15,
        help="Profundidad máxima QAOA. Default: 15",
    )

    parser.add_argument(
        "--num_restarts",
        type=int,
        default=25,
        help="Número de reinicios aleatorios por cada p. Default: 25",
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Máximo de iteraciones del optimizador local. Default: 500",
    )

    parser.add_argument(
        "--mixer",
        type=str,
        default="jx",
        choices=["jx", "custom"],
        help="Mixer a usar: 'jx' para sum Jx_j, 'custom' para tu X local. Default: jx",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="L-BFGS-B",
        help="Método de scipy.optimize.minimize. Default: L-BFGS-B",
    )

    parser.add_argument(
        "--bounds_scale",
        type=float,
        default=float(np.pi),
        help="Cota de parámetros: [-bounds_scale, bounds_scale]. Default: pi",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Semilla aleatoria base. Default: 123",
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=1,
        help="Índice inicial de grafo, empezando desde 1. Default: 1",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Índice final de grafo, inclusivo. Default: todos",
    )

    parser.add_argument(
        "--no_warmstart",
        action="store_true",
        help="Desactiva warm-start entre p y p+1.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce impresión en pantalla.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Ruta del CSV de salida. Default automático en resultados/csv/",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Ruta del JSON de salida. Default automático en resultados/json/",
    )

    return parser.parse_args()


def build_default_output_paths(args):
    resultados_dir = PROJECT_ROOT / "resultados"
    csv_dir = resultados_dir / "csv"
    json_dir = resultados_dir / "json"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    end_label = "all" if args.end_idx is None else str(args.end_idx)
    range_label = f"{args.start_idx}_{end_label}"

    base_name = (
        f"resultados_qaoa_n{args.n}_p{args.p_max}_"
        f"{args.mixer}_r{args.num_restarts}_grafos_{range_label}"
    )

    output_csv = Path(args.output_csv) if args.output_csv else csv_dir / f"{base_name}.csv"
    output_json = Path(args.output_json) if args.output_json else json_dir / f"{base_name}.json"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    return output_csv, output_json


def select_graph_range(grafos, start_idx, end_idx):
    if start_idx < 1:
        raise ValueError("start_idx debe ser >= 1")

    if end_idx is None:
        end_idx = len(grafos)

    if end_idx < start_idx:
        raise ValueError("end_idx debe ser >= start_idx")

    if end_idx > len(grafos):
        raise ValueError(
            f"end_idx={end_idx} excede la cantidad de grafos: {len(grafos)}"
        )

    selected = []
    for grafo_id in range(start_idx, end_idx + 1):
        edges = grafos[grafo_id - 1]
        selected.append((grafo_id, edges))

    return selected


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    show = not args.quiet
    use_warmstart = not args.no_warmstart

    output_csv, output_json = build_default_output_paths(args)

    print("=" * 90)
    print("EJECUCIÓN QAOA/QOAO PARA QUDITS")
    print("=" * 90)
    print(f"input_file    = {args.input_file}")
    print(f"n             = {args.n}")
    print(f"p_max         = {args.p_max}")
    print(f"num_restarts  = {args.num_restarts}")
    print(f"maxiter       = {args.maxiter}")
    print(f"mixer         = {args.mixer}")
    print(f"method        = {args.method}")
    print(f"bounds_scale  = {args.bounds_scale}")
    print(f"seed          = {args.seed}")
    print(f"start_idx     = {args.start_idx}")
    print(f"end_idx       = {args.end_idx}")
    print(f"warmstart     = {use_warmstart}")
    print(f"output_csv    = {output_csv}")
    print(f"output_json   = {output_json}")
    print("=" * 90)

    global_start = time.time()

    grafos = leer_grafos(args.input_file)
    selected_graphs = select_graph_range(grafos, args.start_idx, args.end_idx)

    print(f"Cantidad total de grafos en archivo: {len(grafos)}")
    print(f"Cantidad de grafos seleccionados: {len(selected_graphs)}")

    all_results = []
    errors = []

    for grafo_id, edges in selected_graphs:
        print("\n" + "#" * 90)
        print(f"GRAFO {grafo_id} | n = {args.n} | num_edges = {len(edges)}")
        print(f"edges = {edges}")
        print("#" * 90)

        try:
            results_grafo = scan_qaoa_p(
                n=args.n,
                edges=edges,
                p_max=args.p_max,
                mixer=args.mixer,
                num_restarts=args.num_restarts,
                maxiter=args.maxiter,
                seed=args.seed + 1000 * grafo_id,
                bounds_scale=args.bounds_scale,
                method=args.method,
                use_warmstart=use_warmstart,
                show=show,
            )

            for item in results_grafo:
                item["grafo_id"] = int(grafo_id)

            all_results.extend(results_grafo)

            # Guardado parcial después de cada grafo, por seguridad en cluster.
            df_partial = qaoa_results_to_dataframe(all_results)
            df_partial.to_csv(output_csv, index=False)

            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(
                    to_jsonable(all_results),
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            print(f"\nGuardado parcial actualizado:")
            print(f"CSV  -> {output_csv}")
            print(f"JSON -> {output_json}")

        except Exception as exc:
            error_item = {
                "grafo_id": int(grafo_id),
                "n": int(args.n),
                "edges": edges,
                "num_edges": int(len(edges)),
                "error": str(exc),
            }
            errors.append(error_item)
            print(f"\nERROR en grafo {grafo_id}: {exc}")

    # Guardado final
    if all_results:
        df = qaoa_results_to_dataframe(all_results)
        df.to_csv(output_csv, index=False)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(all_results),
                f,
                indent=4,
                ensure_ascii=False,
            )
    else:
        df = None
        print("No hubo resultados exitosos para guardar en CSV/JSON.")

    if errors:
        error_path = output_json.with_name(output_json.stem + "_errors.json")
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(errors),
                f,
                indent=4,
                ensure_ascii=False,
            )
        print(f"Errores guardados en: {error_path}")

    runtime_min = (time.time() - global_start) / 60.0

    print("\n" + "=" * 90)
    print("FIN EJECUCIÓN QAOA/QOAO")
    print("=" * 90)
    print(f"Tiempo total: {runtime_min:.4f} min")
    print(f"CSV final:  {output_csv}")
    print(f"JSON final: {output_json}")

    if df is not None:
        print("\nResumen final:")
        print(df[[
            "grafo_id",
            "p",
            "num_params",
            "ground_energy",
            "energy_final",
            "relative_error",
            "runtime_min",
        ]])


if __name__ == "__main__":
    main()
