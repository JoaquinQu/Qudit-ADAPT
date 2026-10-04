"""
Verificación del motor de QAOA (`funciones/qaoa_mwnp.py`) y medición de su costo.

1. El estado coincide con una construcción densa independiente (matrices de
   3^n x 3^n y expm) en n = 5, con los dos mezcladores.
2. El gradiente adjunto coincide con diferencias finitas centradas.
3. Tiempo de una evaluación de energía + gradiente con p = 40, n = 10..12.

    python cluster/verificar_qaoa_mwnp.py
"""

import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from functools import reduce

import numpy as np
from scipy.linalg import expm

from funciones.qaoa_mwnp import GENERADORES, energia_y_grad, estado_uniforme, evolucionar
from funciones.utilidades_mwnp import hamiltoniano_joaquin, instancia_unica


def denso(params, h, p, mezclador, n):
    G = GENERADORES[mezclador]
    HM = sum(reduce(np.kron, [G if j == s else np.eye(3) for j in range(n)]) for s in range(n))
    psi = estado_uniforme(n)
    for l in range(p):
        psi = np.exp(-1j * params[l] * h) * psi
        psi = expm(-1j * params[p + l] * HM) @ psi
    return psi


def main():
    rng = np.random.default_rng(1)
    a = instancia_unica(5, 0)["a"]
    h = hamiltoniano_joaquin(a)
    h = h / h.std()
    p = 4
    for mezclador in ("jx", "x"):
        x = rng.uniform(-np.pi, np.pi, 2 * p)
        dif = np.max(np.abs(evolucionar(x, h, p, mezclador) - denso(x, h, p, mezclador, 5)))
        E, g = energia_y_grad(x, h, p, mezclador)
        eps = 1e-6
        gfd = np.array([(energia_y_grad(x + eps * e, h, p, mezclador)[0]
                         - energia_y_grad(x - eps * e, h, p, mezclador)[0]) / (2 * eps)
                        for e in np.eye(2 * p)])
        print(f"mezclador {mezclador}: |estado - denso| = {dif:.1e}   "
              f"|grad - dif. finitas| = {np.max(np.abs(g - gfd)):.1e}   (|grad| ~ {np.abs(g).max():.2f})",
              flush=True)

    for n in (10, 11, 12):
        a = instancia_unica(n, 0)["a"]
        h = hamiltoniano_joaquin(a)
        h = h / h.std()
        x = rng.uniform(-np.pi, np.pi, 80)
        t = time.time()
        energia_y_grad(x, h, 40, "jx")
        print(f"n={n}: energía + gradiente, p=40: {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
