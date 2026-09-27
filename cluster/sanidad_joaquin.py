"""
Prueba de sanidad contra los resultados de Joaquín, operador por operador.

Su código no corre desde un clon limpio (importa `utilidades_factorizacion`,
que no está en el repositorio), pero sus JSON guardan la lista exacta de
operadores que eligió su ADAPT y los parámetros óptimos. Eso permite una
comprobación más fuerte que comparar resultados finales: se RECONSTRUYE su
ansatz, con sus operadores y sus ángulos, sobre nuestro motor, y se verifica
que el estado resultante tenga exactamente su energía.

Si coincide a precisión de máquina, las dos implementaciones concuerdan en el
estado de referencia, la hermitización de los operadores del pool, la
convención de signo y de orden del ansatz, y el Hamiltoniano de costo.

La comparación se hace en SU escala de energía,
    E_Joaquín = 2 E_nuestra + (2/3) (sum a)^2,
porque es la honesta para este problema: su E0 es el desbalance residual.

    python cluster/sanidad_joaquin.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np

from funciones.utilidades_mwnp import (
    D,
    _aplicar_bloque,
    bloque_local,
    estado_referencia,
    hamiltoniano_diag,
    leer_estado,
)

CARPETA_JM = PROJECT_ROOT / "multiway number partition" / "resultados" / "json"


def a_escala_joaquin(E, a):
    """Energía de nuestra escala llevada a la de Joaquín."""
    return 2.0 * E + (2.0 / 3.0) * float(np.sum(a)) ** 2


def reconstruir(labels, params, n):
    """Aplica exp(-i theta_j A_j) en orden, desde |+3>^n."""
    psi = estado_referencia(n)
    for label, theta in zip(labels, params):
        sitios, B = bloque_local(label)
        w, V = np.linalg.eigh(B)
        U = (V * np.exp(-1j * float(theta) * w)) @ V.conj().T
        psi = _aplicar_bloque(psi, [s - 1 for s in sitios], U, n)
    return psi


def main():
    archivos = sorted(p for p in CARPETA_JM.glob("resultados_multiway_l*_n*.json"))
    if not archivos:
        raise SystemExit(f"no encontré resultados de Joaquín en {CARPETA_JM}")

    print(f"{'archivo':46s}{'inst':>5s}{'E suya':>14s}{'E nuestra':>14s}"
          f"{'|dif|':>11s}{'p mejor':>9s}{'cadena':>9s}")

    errores, filas = [], []
    for ruta in archivos:
        datos = json.load(open(ruta, encoding="utf-8"))
        corridas = datos if isinstance(datos, list) else [datos]
        for i, r in enumerate(corridas):
            a, n = r["a"], r["n"]
            psi = reconstruir(r["ansatz_op_labels"], r["params"], n)
            h = hamiltoniano_diag(a)
            E_nuestra = a_escala_joaquin(float(np.real(np.vdot(psi, h * psi))), a)
            dif = abs(E_nuestra - r["final_energy"])
            errores.append(dif)

            # Además: ¿el estado reconstruido da su misma cadena más probable?
            top = leer_estado(psi, a, cuantos=1)[0]
            misma = top["clases"] == list(r["best_digits"])
            # Con degeneración, la cadena más probable puede ser otra del mismo
            # nivel: comparamos entonces la probabilidad, que no depende de eso.
            p_ok = abs(top["probabilidad"] - r["best_prob"]) < 1e-8

            filas.append({"archivo": ruta.name, "instancia": i, "a": a,
                          "E_suya": r["final_energy"], "E_nuestra": E_nuestra,
                          "dif": dif, "misma_cadena": misma, "misma_prob": p_ok})
            print(f"{ruta.name[20:66]:46s}{i:5d}{r['final_energy']:14.8f}"
                  f"{E_nuestra:14.8f}{dif:11.2e}{'ok' if p_ok else 'NO':>9s}"
                  f"{'ok' if misma else '(deg)':>9s}")

    print(f"\n{len(errores)} ansätze reconstruidos")
    print(f"diferencia de energía máxima : {max(errores):.2e}")
    print(f"probabilidad más alta igual  : {sum(f['misma_prob'] for f in filas)}/{len(filas)}")

    salida = PROJECT_ROOT / "resultados" / "json" / "sanidad_joaquin.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"max_dif_energia": max(errores), "filas": filas},
              open(salida, "w", encoding="utf-8"), indent=1)
    print(f"guardado en {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
