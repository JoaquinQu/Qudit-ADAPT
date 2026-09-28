"""
¿Por qué los números se sortean en [1, 3n]?

Estima, por Monte Carlo, qué fracción de conjuntos de n enteros DISTINTOS
sorteados en [1, R] tiene partición perfecta en tres cajas y cuál la tiene
ÚNICA (degeneración 6), para varios rangos R. Con R chico las sumas coinciden
a menudo y aparecen varias particiones perfectas; con R grande la partición
perfecta es rara. El benchmark necesita instancias de solución única en todos
los tamaños n = 5..10.

    python cluster/rangos_mwnp.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import json

import numpy as np

MUESTRAS = 2000
RANGOS = {"2n": lambda n: 2 * n, "3n": lambda n: 3 * n, "4n": lambda n: 4 * n,
          "6n": lambda n: 6 * n, "15": lambda n: 15}


def main():
    rng = np.random.default_rng(0)
    filas = []
    for n in range(5, 11):
        idx = np.arange(3 ** n)
        etiquetas = np.stack([(idx // 3 ** (n - 1 - i)) % 3 for i in range(n)], axis=1)
        caja0 = (etiquetas == 0).astype(float)
        caja1 = (etiquetas == 1).astype(float)
        for nombre, R in RANGOS.items():
            R = R(n)
            perfecta = unica = 0
            for _ in range(MUESTRAS):
                a = rng.choice(np.arange(1, R + 1), n, replace=False).astype(float)
                S = a.sum()
                if S % 3:
                    continue
                # partición perfecta: las cajas 0 y 1 suman S/3 (y entonces la 2 también)
                optimos = int(np.sum((caja0 @ a == S / 3) & (caja1 @ a == S / 3)))
                perfecta += optimos > 0
                unica += optimos == 6
            filas.append({"n": n, "rango": nombre, "R": R,
                          "frac_perfecta": perfecta / MUESTRAS, "frac_unica": unica / MUESTRAS})
            print(f"n={n:2d} R={nombre:>3s} ({R:3d})  perfecta {perfecta/MUESTRAS:.3f}"
                  f"  única {unica/MUESTRAS:.3f}", flush=True)
    salida = PROJECT_ROOT / "resultados" / "json" / "rangos_instancias.json"
    json.dump({"muestras": MUESTRAS, "filas": filas}, open(salida, "w", encoding="utf-8"), indent=1)
    print(f"-> {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
