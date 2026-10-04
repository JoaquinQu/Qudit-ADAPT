"""
Cada instancia bajo varios ÓRDENES de sus números.

Permutar qué número va en qué qutrit no cambia el problema, pero sí puede
cambiar la trayectoria de ADAPT: cuando varios operadores empatan en el
gradiente, argmax toma el de menor índice, y el índice depende del etiquetado.
Este archivo permite medir cuánto del resultado depende de esa elección
arbitraria. El orden 0 es el ordenado de menor a mayor que usa el benchmark;
los demás son permutaciones aleatorias deterministas por (seed, instancia, orden).

    python cluster/instancias_ordenes_mwnp.py --n 6 --ids 0-19 --ordenes 10
    python cluster/instancias_ordenes_mwnp.py --n 5 6 7 8 9 --ordenes 5 --salida datos/mwnp_ordenes_5a9.json
    python cluster/instancias_ordenes_mwnp.py --n 10 11 12 --ordenes 5 \
        --base datos/mwnp_instancias_5a12.json --salida datos/mwnp_ordenes_10a12.json
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse, json
import numpy as np
from funciones.utilidades_mwnp import D, hamiltoniano_diag, particion_desde_indice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[6])
    p.add_argument("--ids", type=str, default="0-19")
    p.add_argument("--ordenes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=str, default="datos/mwnp_instancias.json",
                   help="archivo con las instancias base")
    p.add_argument("--salida", type=str, default=None,
                   help="por defecto datos/mwnp_ordenes_n{n}.json (un solo n)")
    args = p.parse_args()
    a0, b0 = map(int, args.ids.split("-"))
    base = [i for i in json.load(open(PROJECT_ROOT / args.base))["instancias"]
            if i["n"] in args.n and a0 <= i["id"] <= b0]
    fuera = []
    for inst in base:
        for r in range(args.ordenes):
            n = inst["n"]
            rng = np.random.default_rng([args.seed, n, inst["id"], r])
            a = list(inst["a"]) if r == 0 else [int(x) for x in rng.permutation(inst["a"])]
            h = hamiltoniano_diag(a)
            c = particion_desde_indice(int(np.flatnonzero(np.isclose(h, h.min()))[0]), n)
            fuera.append({**inst, "a": a, "id": inst["id"] * 100 + r,
                          "instancia_base": inst["id"], "orden": r,
                          "particion_optima": "".join(map(str, c)),
                          "cajas_optimas": [[a[i] for i in range(n) if c[i] == k] for k in range(D)]})
    salida = (PROJECT_ROOT / args.salida if args.salida
              else PROJECT_ROOT / "datos" / f"mwnp_ordenes_n{args.n[0]}.json")
    json.dump({"instancias": fuera}, open(salida, "w", encoding="utf-8"), indent=1)
    print(f"{len(fuera)} (instancia, orden) -> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
