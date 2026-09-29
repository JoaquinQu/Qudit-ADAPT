"""
¿Cuántos procesos simultáneos rinden más en BitWit?

Corre K copias a la vez de la operación que domina cada lote a n = 12
(energía + gradiente de ADAPT con k = 40, y de QAOA con p = 40) y mide el
tiempo por evaluación de cada copia. El rendimiento total es K / tiempo:
pasado cierto K la memoria se satura y más procesos no rinden más.

    python cluster/calibrar_concurrencia.py
"""

import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from multiprocessing import get_context

import numpy as np

from funciones import qaoa_mwnp as Q
from funciones.utilidades_mwnp import (energia_y_grad, estado_referencia, hamiltoniano_diag,
                                       hamiltoniano_joaquin, preparar_pool)

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 12
A = [i for i in json.load(open(PROJECT_ROOT / "datos" / "mwnp_instancias_5a12.json"))["instancias"]
     if i["n"] == N][0]["a"]
H = hamiltoniano_diag(A)
HJ = hamiltoniano_joaquin(A)
HJ = HJ / HJ.std()
POOL = preparar_pool(N, 1)


def nucleo(i):
    """Núcleo físico del proceso i: primero uno por bloque de L3 (CCD), luego dos, etc."""
    return 8 * (i % 8) + (i // 8)


def trabajo(args):
    tipo, semilla, reps, fijar = args
    if fijar:
        os.sched_setaffinity(0, {nucleo(semilla)})
    rng = np.random.default_rng(semilla)
    if tipo == "adapt":
        ops = [POOL[i] for i in rng.choice(len(POOL), 40, replace=False)]
        f = lambda: energia_y_grad(rng.normal(size=40) * 0.1, ops, estado_referencia(N), H, N)
    else:
        f = lambda: Q.energia_y_grad(rng.uniform(-np.pi, np.pi, 80), HJ, 40, "jx")
    f()
    t = time.time()
    for _ in range(reps):
        f()
    return (time.time() - t) / reps


def main():
    fijar = "--fijar" in sys.argv
    print("procesos fijados a un núcleo, repartidos por bloque de L3" if fijar else "sin fijar", flush=True)
    for tipo, reps in (("adapt", 6), ("qaoa", 3)):
        base = None
        for K in ((1, 8, 16, 24, 32, 40) if fijar else (1, 8, 16, 24, 32, 48)):
            with get_context("fork").Pool(K) as pool:
                ts = pool.map(trabajo, [(tipo, s, reps, fijar) for s in range(K)], chunksize=1)
            t = float(np.mean(ts))
            base = t if base is None else base
            print(f"{tipo:5s} K={K:2d}: {t:6.2f} s por evaluación   rendimiento = {K * base / t:5.1f} "
                  f"procesos a velocidad libre", flush=True)


if __name__ == "__main__":
    main()
