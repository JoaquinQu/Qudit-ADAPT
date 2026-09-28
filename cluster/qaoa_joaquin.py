"""
QAOA de Joaquín, reconstruido desde sus parámetros: ¿qué probabilidad de éxito da?

Joaquín corrió QAOA para qutrits (mixer sum_j J_x, CMA-ES con 25 reinicios,
p = 1..10) sobre sus mismas instancias de n = 5 y 6. Guardó la energía y los
parámetros óptimos de cada p, pero no la probabilidad de medir la solución.
Como el estado QAOA queda fijado por los parámetros,

    |psi> = prod_{l=1}^{p} exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^n ,
    params = [gamma_1..gamma_p, beta_1..beta_p],

se reconstruye acá con su H_C (ec. de formulacion_qubo.pdf) y su H_M, y se
verifica contra la energía que él reportó antes de calcular p_éxito.

Junta el resultado con el de ADAPT (el suyo reconstruido y el nuestro) sobre
las mismas instancias, desde `resultados/json/comparacion_joaquin.json`.

    python cluster/qaoa_joaquin.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from functools import reduce

import numpy as np
import scipy.linalg as sla

from funciones.utilidades import Jx_site
from funciones.utilidades_mwnp import hamiltoniano_joaquin

CARPETA_JM = PROJECT_ROOT / "multiway number partition" / "resultados" / "json"
D = 3


def jx_un_sitio():
    """El J_x de un qutrit en la convención de Joaquín, leído de su propio operador."""
    return Jx_site(1, 1).full()


def estado_qaoa(params, p, h_c, u_mezcla):
    """
    Estado QAOA con params = [gammas, betas]. H_C es diagonal; H_M es suma de
    términos de un sitio que conmutan, así que exp(-i beta H_M) es producto
    tensorial de exp(-i beta J_x).
    """
    n = round(np.log(len(h_c)) / np.log(D))
    psi = np.full(D ** n, D ** (-n / 2), dtype=complex)
    for capa in range(p):
        gamma, beta = params[capa], params[p + capa]
        psi = np.exp(-1j * gamma * h_c) * psi
        u = u_mezcla(beta)
        psi = psi.reshape([D] * n)
        for sitio in range(n):
            psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [sitio])), 0, sitio)
        psi = psi.reshape(-1)
    return psi


def main():
    jx = jx_un_sitio()
    u_mezcla = lambda beta: sla.expm(-1j * beta * jx)

    nuestros = {(tuple(f["a"]), f["l"]): f
                for f in json.load(open(PROJECT_ROOT / "resultados" / "json" / "comparacion_joaquin.json"))}

    filas = []
    max_dif = 0.0
    for ruta in sorted(CARPETA_JM.glob("resultados_multiway_qaoa_*.json")):
        balanceada = "balanceados" in ruta.name
        corridas = json.load(open(ruta, encoding="utf-8"))
        # Un archivo trae una o varias instancias; cada entrada es un (instancia, p).
        por_instancia = {}
        for r in corridas:
            por_instancia.setdefault((tuple(r["a"]), r.get("instancia_idx")), []).append(r)

        for (a, idx), rs in por_instancia.items():
            h_c = hamiltoniano_joaquin(a)
            optimo = np.isclose(h_c, h_c.min())
            curva = []
            for r in sorted(rs, key=lambda r: r["p"]):
                psi = estado_qaoa(r["best_params"], r["p"], h_c, u_mezcla)
                e = float(np.real(np.vdot(psi, h_c * psi)))
                max_dif = max(max_dif, abs(e - r["best_energy"]))
                curva.append({
                    "p": r["p"], "parametros": 2 * r["p"],
                    "E_j": e, "E_j_suyo": r["best_energy"],
                    "p_exito": float(np.sum(np.abs(psi[optimo]) ** 2)),
                    "nfev": r["optimizer_nfev"], "runtime_min": r["runtime_min"],
                })
            adapt = {l: nuestros.get((a, l)) for l in (1, 2)}
            filas.append({
                "archivo": ruta.name, "balanceada": balanceada, "n": len(a), "a": list(a),
                "degeneracion": int(optimo.sum()), "p_azar": float(optimo.sum() / D ** len(a)),
                "qaoa": curva,
                "adapt": {str(l): None if f is None else {
                    "suyo": {"p_exito": f["suyo"]["p_exito"], "k": f["suyo"]["k"]},
                    "nuestro": {"p_exito": f["nuestro"]["p_exito"], "k": f["nuestro"]["k"]},
                } for l, f in adapt.items()},
            })

    print(f"energías reconstruidas contra las suyas: diferencia máxima {max_dif:.1e}\n")
    print(f"{'n':>2s} {'bal':>3s} {'a':24s} {'azar':>7s} {'QAOA p=1':>9s} {'p=5':>7s} {'p=10':>7s}"
          f" {'máx':>7s} | {'ADAPT l=1':>9s} {'k':>3s} {'l=2':>7s} {'k':>3s}")
    for f in sorted(filas, key=lambda f: (f["n"], not f["balanceada"])):
        q = {c["p"]: c["p_exito"] for c in f["qaoa"]}
        ad = f["adapt"]
        a1 = ad["1"]["nuestro"] if ad["1"] else None
        a2 = ad["2"]["nuestro"] if ad["2"] else None
        print(f"{f['n']:2d} {'sí' if f['balanceada'] else 'no':>3s} {str(f['a']):24s} {f['p_azar']:7.3f}"
              f" {q.get(1, np.nan):9.3f} {q.get(5, np.nan):7.3f} {q.get(10, np.nan):7.3f} {max(q.values()):7.3f} |"
              f" {a1['p_exito'] if a1 else np.nan:9.3f} {a1['k'] if a1 else 0:3d}"
              f" {a2['p_exito'] if a2 else np.nan:7.3f} {a2['k'] if a2 else 0:3d}")

    salida = PROJECT_ROOT / "resultados" / "json" / "qaoa_joaquin.json"
    json.dump({"max_dif_energia": max_dif, "instancias": filas},
              open(salida, "w", encoding="utf-8"), indent=1)
    print(f"\n{len(filas)} instancias -> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
