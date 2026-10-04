"""
Conteo exacto de t_w^{(l)}, los operadores del pool contradiabático por
subconjunto de w qutrits, con coeficientes numéricos en vez de simbólicos.

Reproduce las reglas del motor simbólico (`funciones/utilidades.py`):
  * un operador es una palabra de factores (sitio, eje), ordenada de forma
    estable por sitio (dentro de un sitio se conserva el orden);
  * [a, palabra] para un factor a se calcula con la regla de Leibniz, usando
    [L_x, L_y] = i L_z y cíclicas; [T, M] para un producto T = a * resto es
    a [resto, M] + [a, M] resto;
  * el pool toma las palabras con coeficiente no nulo en O_1, O_3, ...,
    O_{2l-1} y las identifica por su multiconjunto de factores.

Los coeficientes de O_k son polinomios en lambda y en los pesos; se evalúan en
valores numéricos genéricos (lambda y pesos al azar), de modo que un
coeficiente distinto de cero como polinomio es distinto de cero en el punto
elegido con probabilidad 1. Las cancelaciones estructurales sí se anulan.

Dos reducciones, ambas exactas porque el soporte de una palabra nunca se
achica al conmutar:
  * t_w se obtiene en K_w: las palabras con soporte exactamente S sólo usan
    términos de H_ad dentro de S, salvo los términos de un sitio z_j^2, cuyo
    coeficiente depende de n pero es genérico;
  * en el paso k se descartan las palabras con soporte s < w - (2l - 1 - k),
    porque ya no pueden llegar a cubrir los w sitios.

    python cluster/pool_numerico.py --l 4 --w 1 2 3 4 5 6 7 8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

TABLA = {("x", "y"): ("z", 1j), ("y", "z"): ("x", 1j), ("z", "x"): ("y", 1j),
         ("y", "x"): ("z", -1j), ("z", "y"): ("x", -1j), ("x", "z"): ("y", -1j)}


def canon(palabra):
    return tuple(sorted(palabra, key=lambda f: f[0]))


def comm_local(a, palabra):
    out = {}
    for k, (s, eje) in enumerate(palabra):
        if s == a[0] and eje != a[1]:
            r, c = TABLA[(a[1], eje)]
            w = canon(palabra[:k] + ((s, r),) + palabra[k + 1:])
            out[w] = out.get(w, 0) + c
    return out


def comm_mono(m1, palabra):
    if len(m1) == 1:
        return comm_local(m1[0], palabra)
    a, resto = m1[0], m1[1:]
    out = {}
    for w, c in comm_mono(resto, palabra).items():
        w2 = canon((a,) + w)
        out[w2] = out.get(w2, 0) + c
    for w, c in comm_local(a, palabra).items():
        w2 = canon(w + resto)
        out[w2] = out.get(w2, 0) + c
    return out


def terminos(n, lam, pesos, n_total=None):
    """
    H_ad = (1 - lam) H_M + lam H_C y O_0 = H_C - H_M, como listas (monomio, coef).

    Con n_total > n se simula el subconjunto de n sitios dentro de K_{n_total}:
    los pares hacia los n_total - n sitios de afuera sólo aportan su parte de un
    sitio, -2 z_j^2 por cada par, que se suma acá.
    """
    HM, HC = {}, {}
    if n_total is not None and n_total > n:
        for j in range(1, n + 1):
            HC[((j, "z"), (j, "z"))] = -2.0 * (n_total - n)
    for j in range(1, n + 1):
        HM[((j, "x"),)] = HM.get(((j, "x"),), 0) - np.sqrt(2)
        HM[((j, "z"), (j, "z"))] = HM.get(((j, "z"), (j, "z")), 0) - 1
    for (i, j), w in pesos.items():
        for m, c in ((((i, "z"), (j, "z")), 1), (((i, "z"), (i, "z")), -2), (((j, "z"), (j, "z")), -2),
                     (((i, "z"), (i, "z"), (j, "z"), (j, "z")), 3)):
            HC[m] = HC.get(m, 0) + w * c
    Had, O0 = {}, {}
    for m, c in HM.items():
        Had[m] = Had.get(m, 0) + (1 - lam) * c
        O0[m] = O0.get(m, 0) - c
    for m, c in HC.items():
        Had[m] = Had.get(m, 0) + lam * c
        O0[m] = O0.get(m, 0) + c
    return list(Had.items()), O0


def soporte(palabra):
    return len({s for s, _ in palabra})


def contar(l, w, pesos_iguales=False, semilla=0, tol=1e-12, n_total=None):
    from math import comb
    rng = np.random.default_rng(semilla)
    lam = rng.uniform(0.2, 0.8)
    # Con w = 1 hace falta al menos un par para que haya H_C; se usa K_2 y se
    # divide por los C(2, 1) = 2 sitios.
    nsis = max(w, 2)
    pares = [(i, j) for i in range(1, nsis + 1) for j in range(i + 1, nsis + 1)]
    pesos = {e: (1.0 if pesos_iguales else rng.uniform(0.5, 2.0)) for e in pares}
    Had, O = terminos(nsis, lam, pesos, n_total)
    kmax = 2 * l - 1
    pool = set()
    for k in range(1, kmax + 1):
        nuevo = {}
        for m, c in Had:
            for pal, cp in O.items():
                for r, cr in comm_mono(m, pal).items():
                    nuevo[r] = nuevo.get(r, 0) + c * cp * cr
        minimo = w - (kmax - k)                # soporte mínimo para poder llegar a w
        escala = max((abs(v) for v in nuevo.values()), default=0.0)
        O = {p: v for p, v in nuevo.items() if abs(v) > tol * escala and soporte(p) >= minimo}
        if k % 2 == 1:
            pool |= {tuple(sorted(p)) for p in O if soporte(p) == w}
    return len(pool) // comb(nsis, w)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--l", type=int, nargs="+", required=True)
    pa.add_argument("--w", type=int, nargs="+", default=None)
    pa.add_argument("--iguales", action="store_true", help="pesos 1 (K_n sin pesos, como el código)")
    pa.add_argument("--salida", type=str, default=None)
    pa.add_argument("--tol", type=float, default=1e-12)
    pa.add_argument("--semilla", type=int, default=0)
    pa.add_argument("--n_total", type=int, default=None, help="tamaño del K_n que se simula")
    args = pa.parse_args()
    filas = []
    for l in args.l:
        for w in (args.w or range(1, 2 * l + 1)):
            if w > 2 * l:
                continue
            t0 = time.time()
            t = contar(l, w, pesos_iguales=args.iguales, semilla=args.semilla, tol=args.tol,
                       n_total=args.n_total)
            filas.append({"l": l, "w": w, "t_w": t, "tol": args.tol, "semilla": args.semilla,
                          "n_total": args.n_total, "t_s": time.time() - t0})
            print(f"l={l} w={w}: t_w = {t}   ({time.time() - t0:.1f} s)", flush=True)
            if args.salida:
                json.dump(filas, open(args.salida, "w"), indent=1)


if __name__ == "__main__":
    main()
