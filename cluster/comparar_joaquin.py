"""
Nuestro ADAPT contra el de Joaquín, sobre SUS instancias y con SUS ajustes.

Joaquín corrió n = 5 y 6, con l = 1 y 2, cinco instancias por archivo
(genéricas y balanceadas), epsilon = 1e-3 en su escala y tope de 30 iteraciones.
Nuestras instancias de la fase 1 son otras —números distintos, solución única—,
así que la comparación honesta es correr nuestro motor sobre las suyas.

De su lado no hace falta su código: sus JSON guardan los operadores y los
parámetros finales, y con eso se reconstruye su estado final exacto (ver
`sanidad_joaquin.py`, que lo valida a 1e-12). De ahí sale su probabilidad de
éxito, que él no guardó, calculada igual que la nuestra.

Los dos ADAPT no pueden elegir exactamente los mismos operadores: su pool trae
n operadores de un sitio más que el nuestro, porque sus proyectores son
cuadráticos en el operador dígito y la expansión simbólica produce más monomios
distintos. Lo que se compara es el resultado sobre el mismo problema.

    python cluster/comparar_joaquin.py
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

from funciones.utilidades_mwnp import (
    adapt_mwnp,
    hamiltoniano_diag,
    leer_estado,
    preparar_pool,
)
from sanidad_joaquin import a_escala_joaquin, reconstruir

CARPETA_JM = PROJECT_ROOT / "multiway number partition" / "resultados" / "json"


def p_exito(psi, a):
    """Peso del estado sobre el subespacio fundamental."""
    h = hamiltoniano_diag(a)
    return float(np.sum(np.abs(psi[np.isclose(h, h.min())]) ** 2))


def main():
    filas = []
    archivos = sorted(CARPETA_JM.glob("resultados_multiway_l*_n*.json"))
    for ruta in archivos:
        suyas = json.load(open(ruta, encoding="utf-8"))
        balanceada = "balanceados" in ruta.name
        for r in suyas:
            a, n, l = r["a"], r["n"], r["l"]

            # Su estado final, reconstruido desde sus operadores y parámetros.
            psi_suyo = reconstruir(r["ansatz_op_labels"], r["params"], n)
            top_suyo = leer_estado(psi_suyo, a, cuantos=1)[0]

            # Nuestro ADAPT con sus mismos ajustes: epsilon 1e-3 en su escala
            # (5e-4 en la nuestra, donde los gradientes valen la mitad) y tope 30.
            t = time.time()
            nuestro = adapt_mwnp(a, l=l, epsilon=r["epsilon"] / 2.0,
                                 max_iteration=r["max_iteration"],
                                 pool=preparar_pool(n, l), mostrar=False)
            h = hamiltoniano_diag(a)

            filas.append({
                "archivo": ruta.name, "balanceada": balanceada,
                "n": n, "l": l, "a": a,
                "repetidos": len(set(a)) < len(a),
                "degeneracion": int(np.isclose(h, h.min()).sum()),
                "suyo": {
                    "k": r["num_ansatz_ops"],
                    "E_final_j": r["final_energy"],
                    "error_abs_j": r["difference_ground"],
                    "p_exito": p_exito(psi_suyo, a),
                    "desbalance_top": top_suyo["desbalance"],
                    "stop": ("gradient_norm_below_epsilon"
                             if r["final_gradient_norm"] < r["epsilon"] else "max_iteration"),
                },
                "nuestro": {
                    "k": nuestro["num_ansatz_ops"],
                    "E_final_j": a_escala_joaquin(nuestro["final_energy"], a),
                    "error_abs_j": (a_escala_joaquin(nuestro["final_energy"], a)
                                    - a_escala_joaquin(nuestro["ground_energy"], a)),
                    "p_exito": nuestro["prob_subespacio_optimo"],
                    "desbalance_top": nuestro["desbalance"],
                    "stop": nuestro["stop_reason"],
                    "t_s": time.time() - t,
                },
            })
            f = filas[-1]
            print(f"{ruta.name[20:62]:42s} {str(a):22s}"
                  f"  p suyo {f['suyo']['p_exito']:.3f}  p nuestro {f['nuestro']['p_exito']:.3f}"
                  f"  k {f['suyo']['k']:2d}/{f['nuestro']['k']:2d}", flush=True)

    salida = PROJECT_ROOT / "resultados" / "json" / "comparacion_joaquin.json"
    json.dump(filas, open(salida, "w", encoding="utf-8"), indent=1)
    print(f"\n{len(filas)} instancias -> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
