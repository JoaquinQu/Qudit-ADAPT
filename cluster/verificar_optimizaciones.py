"""
Verifica las dos optimizaciones antes de usarlas en el benchmark, y mide su costo.

ADAPT: `gradientes_por_soporte` contra `gradientes_pool` (el barrido original)
  1. mismo vector de gradientes en estados aleatorios, n = 5..9, l = 1 y 2;
  2. corridas ya hechas (l = 2, tope 150) repetidas con el barrido nuevo hasta
     k = 80: ¿mismos operadores elegidos y mismo p_éxito?
QAOA: mezclador por grupos contra la versión por sitio y la construcción densa;
  gradiente contra diferencias finitas; energías de los reinicios de n = 12 ya
  guardados, recalculadas desde sus parámetros.
Tiempos con la máquina libre.

    python cluster/verificar_optimizaciones.py
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
from functools import reduce
from multiprocessing import get_context

import numpy as np
from scipy.linalg import expm

from funciones import qaoa_mwnp as Q
from funciones.utilidades_mwnp import (adapt_mwnp, gradientes_por_soporte, gradientes_pool,
                                       hamiltoniano_diag, hamiltoniano_joaquin, preparar_barrido,
                                       preparar_pool)

INST = json.load(open(PROJECT_ROOT / "datos" / "mwnp_instancias_5a12.json"))["instancias"]
ORD = json.load(open(PROJECT_ROOT / "datos" / "mwnp_ordenes_5a9.json"))["instancias"]


def instancia(n):
    return [i for i in INST if i["n"] == n][0]["a"]


def barrido_aleatorio():
    rng = np.random.default_rng(0)
    print("1) barrido por soporte contra barrido por operador (estado aleatorio)", flush=True)
    for n in (5, 6, 7, 8, 9):
        h = hamiltoniano_diag(instancia(n))
        psi = rng.normal(size=3 ** n) + 1j * rng.normal(size=3 ** n)
        psi /= np.linalg.norm(psi)
        for l in (1, 2):
            pool = preparar_pool(n, l)
            g0 = gradientes_pool(psi, pool, h, n)
            g1 = gradientes_por_soporte(psi, preparar_barrido(pool, n), h, n)
            print(f"   n={n} l={l}: max|dif| = {np.max(np.abs(g1 - g0)):.1e}  "
                  f"(max|g| = {np.max(np.abs(g0)):.1e}, pool {len(pool)})", flush=True)


def repetir(args):
    n, base, orden = args
    ident = base * 100 + orden
    viejo = json.load(open(PROJECT_ROOT / "resultados" / "mwnp_ordenes_5a9" / f"n{n}_l2_i{ident:02d}.json"))
    inst = [i for i in ORD if i["n"] == n and i["id"] == ident][0]
    pool = preparar_pool(n, 2)
    r = adapt_mwnp(inst["a"], l=2, epsilon=1e-3 / 2.0, max_iteration=80, pool=pool,
                   mostrar=False, barrido=preparar_barrido(pool, n))
    k = min(80, len(viejo["trazas"]["indices_pool"]))
    iguales = r["ansatz_op_indices"][:k] == viejo["trazas"]["indices_pool"][:k]
    primera = next((i for i, (x, y) in enumerate(zip(r["ansatz_op_indices"], viejo["trazas"]["indices_pool"]))
                    if x != y), None)
    p_viejo = viejo["trazas"]["p_exito"][min(80, len(viejo["trazas"]["p_exito"]) - 1)]
    return (n, ident, iguales, primera, p_viejo, r["prob_subespacio_optimo"])


def repeticiones():
    print("\n2) corridas l=2 repetidas con el barrido nuevo (k <= 80)", flush=True)
    casos = [(6, b, 0) for b in range(5)] + [(7, b, 0) for b in range(5)] + [(8, b, 0) for b in range(4)]
    with get_context("fork").Pool(14) as pool:
        for n, ident, iguales, primera, pv, pn in pool.imap_unordered(repetir, casos):
            print(f"   n={n} #{ident:<4d} misma secuencia: {iguales}"
                  + ("" if iguales else f" (difiere desde k={primera + 1})")
                  + f"   p_éxito viejo {pv:.4f}  nuevo {pn:.4f}", flush=True)


def qaoa():
    print("\n3) QAOA por grupos", flush=True)
    rng = np.random.default_rng(1)
    for n in (5, 6):
        h = hamiltoniano_joaquin(instancia(n))
        h = h / h.std()
        for mez in ("jx", "x"):
            x = rng.uniform(-np.pi, np.pi, 8)
            G = Q.GENERADORES[mez]
            HM = sum(reduce(np.kron, [G if j == s else np.eye(3) for j in range(n)]) for s in range(n))
            psi = Q.estado_uniforme(n)
            for l in range(4):
                psi = expm(-1j * x[4 + l] * HM) @ (np.exp(-1j * x[l] * h) * psi)
            dif = np.max(np.abs(Q.evolucionar(x, h, 4, mez) - psi))
            E, g = Q.energia_y_grad(x, h, 4, mez)
            gfd = np.array([(Q.energia_y_grad(x + 1e-6 * e, h, 4, mez)[0]
                             - Q.energia_y_grad(x - 1e-6 * e, h, 4, mez)[0]) / 2e-6 for e in np.eye(8)])
            print(f"   n={n} {mez}: |estado - denso| = {dif:.1e}   |grad - dif. finitas| = "
                  f"{np.max(np.abs(g - gfd)):.1e}", flush=True)
    n = 8
    pila = rng.normal(size=(2, 3 ** n)) + 1j * rng.normal(size=(2, 3 ** n))
    U = Q._unitario(*np.linalg.eigh(Q.LX), 0.7)
    print(f"   n=8: grupos contra sitio por sitio: "
          f"{np.max(np.abs(Q._en_cada_sitio(pila, U, n) - Q._en_cada_sitio_por_sitio(pila, U, n))):.1e}   "
          f"<H_M>: {abs(Q._esperado_suma_local(pila[1], pila[0], Q.LX, n) - np.vdot(pila[1], Q._suma_local(pila[0], Q.LX, n))):.1e}",
          flush=True)
    for f in sorted((PROJECT_ROOT / "resultados" / "qaoa_p40").glob("n12_*_r*.json")):
        d = json.load(open(f))
        h = hamiltoniano_joaquin(d["instancia"]["a"])
        E, _ = Q.energia_y_grad(np.array(d["parametros"]["x"]), h / h.std(), 40, "jx")
        print(f"   {f.name}: E guardada {d['resultado']['E_norm']:.12f}  recalculada {E:.12f}", flush=True)


def tiempos():
    print("\n4) tiempos, máquina libre", flush=True)
    rng = np.random.default_rng(2)
    for n in (10, 11, 12):
        h = hamiltoniano_diag(instancia(n))
        psi = rng.normal(size=3 ** n) + 1j * rng.normal(size=3 ** n)
        psi /= np.linalg.norm(psi)
        pool = preparar_pool(n, 2)
        b = preparar_barrido(pool, n)
        t = time.time(); gradientes_por_soporte(psi, b, h, n); t1 = time.time() - t
        hj = hamiltoniano_joaquin(instancia(n)); hj = hj / hj.std()
        t = time.time(); Q.energia_y_grad(rng.uniform(-np.pi, np.pi, 80), hj, 40, "jx"); t2 = time.time() - t
        print(f"   n={n}: barrido l=2 por soporte {t1:.1f} s ({len(b['soportes'])} soportes)   "
              f"QAOA p=40 energía+gradiente {t2:.1f} s", flush=True)
        del pool, b


if __name__ == "__main__":
    barrido_aleatorio()
    repeticiones()
    qaoa()
    tiempos()
