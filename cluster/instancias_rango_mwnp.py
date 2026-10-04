"""
Instancias para aislar el efecto del RANGO de los números sobre la dificultad.

Con n = 6, números distintos y solución única, a rango [1, 8] existen
exactamente 9 instancias —se enumeran todas, no se sortean— y a [1, 18], que es
el rango 3n de la fase 1, existen 967. Si nuestro motor resuelve las de [1, 8]
tanto mejor que las de [1, 18], la dificultad extra de nuestras instancias viene
de la magnitud de los números y no del criterio de solución única.

    python cluster/instancias_rango_mwnp.py --n 6 --rango 8
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import itertools
import json

import numpy as np

from funciones.utilidades_mwnp import (D, energia_a_objetivo, hamiltoniano_diag,
                                       particion_desde_indice)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--rango", type=int, default=8)
    args = p.parse_args()
    n, R = args.n, args.rango

    fuera = []
    for a in itertools.combinations(range(1, R + 1), n):
        a = list(a)
        if sum(a) % 3:
            continue
        h = hamiltoniano_diag(a)
        E0 = float(h.min())
        if not np.isclose(energia_a_objetivo(E0, a), sum(a) ** 2 / 3):
            continue
        opt = np.flatnonzero(np.isclose(h, E0))
        if len(opt) != 6:
            continue
        c = particion_desde_indice(int(opt[0]), n)
        fuera.append({"n": n, "id": len(fuera), "rango": R, "a": a,
                      "suma": sum(a), "suma_por_caja": sum(a) // 3, "degeneracion": 6,
                      "particion_optima": "".join(map(str, c)),
                      "cajas_optimas": [[a[i] for i in range(n) if c[i] == k] for k in range(D)],
                      "p_azar": 6.0 / D ** n})
    salida = PROJECT_ROOT / "datos" / f"mwnp_rango{R}_n{n}.json"
    json.dump({"criterios": {"numeros_distintos": True, "rango": f"[1, {R}]",
                             "degeneracion": 6, "enumeracion": "exhaustiva"},
               "instancias": fuera}, open(salida, "w", encoding="utf-8"), indent=1)
    print(f"{len(fuera)} instancias -> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
