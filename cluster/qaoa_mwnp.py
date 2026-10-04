"""
QAOA bien optimizado sobre las instancias de Joaquín (n = 5, 6).

Su QAOA (CMA-ES, 25 reinicios con parámetros sorteados en [-pi, pi], hasta 500
evaluaciones por reinicio) empeora al agregar capas, lo que no puede pasar si
el optimizador converge: el circuito de p capas contiene al de p - 1. El
problema es de escala: H_C tiene energías de cientos, así que el gamma útil es
del orden de 1/std(H_C) y un sorteo en [-pi, pi] cae casi siempre en una
región que oscila demasiado rápido.

Acá se usa la misma forma de circuito (mixer sum_j J_x, estado inicial
|+>^n, params = [gammas, betas]; ver `qaoa_joaquin.py`) con:

  * p = 1: barrido en grilla de (gamma, beta), con gamma en unidades de
    1/std(H_C), y BFGS desde el mejor punto;
  * p -> p + 1: inicialización INTERP (Zhou et al., PRX 10, 021067 (2020)),
    que interpola linealmente los parámetros óptimos de p capas, y BFGS.

    python cluster/qaoa_mwnp.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cluster"))

import json
import time

import numpy as np
from scipy.optimize import minimize

from funciones.utilidades_mwnp import hamiltoniano_joaquin
from qaoa_joaquin import estado_qaoa, jx_un_sitio

P_MAX = 10


def interp(x, p):
    """Parámetros óptimos de p capas -> punto inicial de p + 1 capas (INTERP)."""
    def f(v):
        v = np.concatenate([[0.0], v, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * v[i - 1] + (p - i + 1) / p * v[i]
    return np.concatenate([f(x[:p]), f(x[p:])])


def main():
    w, V = np.linalg.eigh(jx_un_sitio())
    u_mezcla = lambda b: (V * np.exp(-1j * b * w)) @ V.conj().T

    def energia(x, p, h):
        psi = estado_qaoa(x, p, h, u_mezcla)
        return float(np.real(np.vdot(psi, h * psi)))

    suyas = json.load(open(PROJECT_ROOT / "resultados" / "json" / "qaoa_joaquin.json"))["instancias"]
    filas = []
    for f in suyas:
        t0 = time.time()
        a = f["a"]
        h = hamiltoniano_joaquin(a)
        optimo = np.isclose(h, h.min())
        escala = 1.0 / np.std(h)
        G, B = np.meshgrid(np.linspace(0.02, 3.0, 40) * escala, np.linspace(-np.pi / 2, np.pi / 2, 40))
        E = [energia([g, b], 1, h) for g, b in zip(G.ravel(), B.ravel())]
        x = np.array([G.ravel()[np.argmin(E)], B.ravel()[np.argmin(E)]])
        curva = []
        for p in range(1, P_MAX + 1):
            if p > 1:
                x = interp(x, p - 1)
            r = minimize(energia, x, args=(p, h), method="BFGS", options={"gtol": 1e-8})
            x = r.x
            psi = estado_qaoa(x, p, h, u_mezcla)
            curva.append({"p": p, "parametros": 2 * p, "E_j": float(r.fun),
                          "p_exito": float(np.sum(np.abs(psi[optimo]) ** 2)),
                          "nfev": int(r.nfev), "params": x.tolist()})
        filas.append({"archivo": f["archivo"], "balanceada": f["balanceada"], "n": f["n"], "a": a,
                      "p_azar": f["p_azar"], "curva": curva, "t_s": time.time() - t0})
        print(f"n={f['n']} {str(a):24s} E p=1 {curva[0]['E_j']:7.1f}  p={P_MAX} {curva[-1]['E_j']:6.1f}"
              f"   p_exito p={P_MAX} {curva[-1]['p_exito']:.3f}   ({time.time()-t0:.0f} s)", flush=True)

    salida = PROJECT_ROOT / "resultados" / "json" / "qaoa_reoptimizado.json"
    json.dump({"metodo": "grilla en p=1 + INTERP + BFGS", "p_max": P_MAX, "instancias": filas},
              open(salida, "w", encoding="utf-8"), indent=1)
    print(f"\n{len(filas)} instancias -> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
