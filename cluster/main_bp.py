"""
Estudio de barren plateaus / paisajes rugosos para Qudit-ADAPT-VQE.

Corre `adapt_bp_scan` sobre un grafo de un archivo de `datos/` y guarda el
resultado completo (curvas reciclada/fría + historias de coste de los
reinicios aleatorios) en `resultados/json/`.

Ejemplos:

    python cluster/main_bp.py --grafo 2 --l 2 --max_iteration 30 --n_random 10
    python cluster/main_bp.py --grafo 1 --l 1 --max_iteration 20 --n_random 10
    python cluster/main_bp.py --input_file grafos_regulares.txt --grafo 3 --l 1
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import ast

from funciones.utilidades_bp import adapt_bp_scan, save_bp_result

DATOS_DIR = PROJECT_ROOT / "datos"


def leer_grafo(input_file, grafo_id):
    """Lee la línea `grafo_id` (1-indexada) de un archivo de grafos de datos/."""
    path = Path(input_file)
    if not path.is_absolute():
        path = DATOS_DIR / path

    with open(path, "r", encoding="utf-8") as f:
        lineas = [ln.strip() for ln in f if ln.strip()]

    if not 1 <= grafo_id <= len(lineas):
        raise ValueError(f"grafo_id={grafo_id} fuera de rango (1..{len(lineas)}) en {path.name}")

    return ast.literal_eval("[" + lineas[grafo_id - 1] + "]")


def main():
    parser = argparse.ArgumentParser(description="Barren plateaus en Qudit-ADAPT-VQE")
    parser.add_argument("--input_file", type=str, default="grafos_comparacion.txt")
    parser.add_argument("--grafo", type=int, default=2, help="índice del grafo (1-indexado)")
    parser.add_argument("--n", type=int, default=6, help="número de qutrits")
    parser.add_argument("--l", type=int, default=2, choices=[1, 2])
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--max_iteration", type=int, default=30)
    parser.add_argument("--n_random", type=int, default=100,
                        help="instancias con parámetros aleatorios por iteración")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="hilos para los reinicios (por defecto min(6, n_cpu))")
    parser.add_argument("--store_history", action="store_true",
                        help="guardar todos los valores visitados, no sólo el óptimo final")
    parser.add_argument("--base", type=str, default="angular",
                        choices=["angular", "gellmann", "gellmann_local", "heisenberg"],
                        help="base en la que se descompone el pool contradiabático")
    parser.add_argument("--no_resume", action="store_true",
                        help="ignorar el checkpoint previo y empezar de cero")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    edges = leer_grafo(args.input_file, args.grafo)

    sufijo = "" if args.base == "angular" else f"_{args.base}"
    nombre = args.output or (
        f"bp_n{args.n}_l{args.l}_grafo{args.grafo}_"
        f"r{args.n_random}_k{args.max_iteration}{sufijo}.json"
    )

    print(f"Archivo: {args.input_file} | grafo {args.grafo} | n = {args.n}")
    print(f"Aristas: {edges}")
    print(f"Salida (se guarda en cada iteración): {nombre}")

    result = adapt_bp_scan(
        n=args.n,
        edges=edges,
        l=args.l,
        epsilon=args.epsilon,
        max_iteration=args.max_iteration,
        n_random=args.n_random,
        seed=args.seed,
        show=True,
        store_history=args.store_history,
        n_jobs=args.n_jobs,
        checkpoint_path=nombre,
        base=args.base,
        resume=not args.no_resume,
    )

    result["input_file"] = args.input_file
    result["grafo_id"] = args.grafo

    path = save_bp_result(result, nombre)
    print(f"\nResultado guardado en: {path}")


if __name__ == "__main__":
    main()
