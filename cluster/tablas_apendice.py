"""
Tables II and III of Appendix B: angular-momentum vs Gell-Mann operator pools.

Table II compares the two pools by what they cost to *run*: pool size, number
of variational parameters, relative error reached, accumulated two-qutrit
entangling gates and cumulative gradient measurements.

Table III compares them by what they cost to *compile*, at the moderate target
precision eps_rel <= 1e-3, splitting the count into single-qudit two-level
rotations and two-qudit Molmer-Sorensen gates.

Both read the two runs on instance G_2 with the l = 2 pool. The gate counts
come from `conteo_compuertas_ansatz`, which compiles every single-qutrit
unitary with the decomposition algorithm of Ringbauer et al. (see
funciones/utilidades_ringbauer.py).

    python cluster/tablas_apendice.py
    python cluster/tablas_apendice.py --latex     # ready to paste into the paper
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np

from funciones.utilidades_bp import (
    conteo_compuertas,
    conteo_compuertas_ansatz,
    costo_compuertas,
    costo_mediciones,
    load_bp_result,
)

# Las dos corridas sobre la misma instancia G_2 con l = 2. Lo único que cambia
# entre ellas es la base en que se descomponen los conmutadores anidados.
POOLS = {
    "Angular momentum": "bp_n6_l2_grafo2_r20_k30.json",
    "Gell-Mann": "bp_n6_l2_grafo2_r20_k30_gellmann.json",
}

# Precisión objetivo de la Tabla III. Es un régimen de precisión moderada, que
# es donde la ventaja de compilación de Gell-Mann es relevante: a precisión de
# máquina el pool angular converge en muchos menos parámetros.
UMBRAL = 1e-3


def cargar():
    datos = {}
    for etiqueta, archivo in POOLS.items():
        res = load_bp_result(archivo)
        E0 = float(res["ground_energy"])
        datos[etiqueta] = {
            "res": res,
            "eps": np.abs(np.asarray(res["recycled_energy"], float) - E0) / abs(E0),
            "gates": costo_compuertas(res),
            "meas": costo_mediciones(res),
            "nativo": conteo_compuertas_ansatz(res),
        }
    return datos


def _en_umbral(d):
    """Índice del primer ansatz que baja del umbral, o None si nunca lo hace."""
    bajo = np.where(d["eps"] < UMBRAL)[0]
    return int(bajo[0]) if len(bajo) else None


def tabla_ii(datos, latex=False):
    filas = []
    for etiqueta, d in datos.items():
        r = d["res"]
        filas.append((etiqueta, r["pool_size"], r["num_ansatz_ops"], d["eps"][-1],
                      int(d["gates"][-1]), int(d["meas"][-1]), r["stop_reason"]))

    if latex:
        print(r"% Table II")
        for e, pool, k, eps, g, m, _ in filas:
            mant, exp = f"{eps:.1e}".split("e")
            print(f"{e} & {pool} & {k} & ${mant}\\times10^{{{int(exp)}}}$ & {g} & "
                  f"{m:,} \\\\".replace(",", r"\,"))
        return

    print("\nTABLE II  --  pool comparison (n = 6 qutrits, instance G_2, l = 2)")
    print(f"  {'Pool basis':18s}{'size':>7s}{'params':>8s}{'eps_rel':>12s}"
          f"{'ent. gates':>12s}{'grad. meas':>12s}")
    for e, pool, k, eps, g, m, stop in filas:
        print(f"  {e:18s}{pool:>7d}{k:>8d}{eps:>12.1e}{g:>12d}{m:>12d}   ({stop})")


def tabla_iii(datos, latex=False):
    filas = []
    for etiqueta, d in datos.items():
        k = _en_umbral(d)
        if k is None:
            print(f"  {etiqueta}: never reaches eps_rel <= {UMBRAL:g}")
            continue
        n = d["nativo"]
        filas.append((etiqueta, k, int(n["r_dos_niveles"][k]), int(n["ms"][k]),
                      int(n["total"][k]), int(d["meas"][k])))

    if latex:
        print(r"% Table III")
        for e, k, r, ms, tot, m in filas:
            destaca = r"\textbf{%d}" % tot if e == "Gell-Mann" else str(tot)
            print(f"{e} & {k} & {r} & {ms} & {destaca} & {m:,} \\\\".replace(",", r"\,"))
        return

    print(f"\nTABLE III  --  native gate decomposition at eps_rel <= {UMBRAL:g}")
    print(f"  {'Pool basis':18s}{'params':>8s}{'R(i,j)':>9s}{'MS':>6s}"
          f"{'total':>8s}{'grad. meas':>12s}")
    for e, k, r, ms, tot, m in filas:
        print(f"  {e:18s}{k:>8d}{r:>9d}{ms:>6d}{tot:>8d}{m:>12d}")

    if len(filas) == 2:
        a, g = filas[0][4], filas[1][4]
        print(f"\n  Gell-Mann reduces the total native gate count from {a} to {g}"
              f"  ({(a - g) / a * 100:.0f} %)")


def detalle(datos):
    """Los dos números que el texto del apéndice cita fuera de las tablas."""
    print("\nQuoted in the text:")
    for etiqueta, d in datos.items():
        r = d["res"]
        base = r.get("base") or "angular"
        cuentas = [conteo_compuertas(l, base=base)["r_dos_niveles"]
                   for l in r["ansatz_op_labels"]]
        # Generadores cuyo (O + O^dag)/2 NO factoriza: ahí la escalera no
        # aplica y haría falta trotterizar, un costo que no está en las tablas.
        no_prod = sum(0 if conteo_compuertas(l, base=base)["es_producto"] else 1
                      for l in r["ansatz_op_labels"])
        print(f"  {etiqueta:18s} mean two-level rotations per generator: "
              f"{np.mean(cuentas):5.2f}   needing Trotterization: {no_prod}/{len(cuentas)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--latex", action="store_true",
                   help="emit the table rows as LaTeX instead of a text table")
    args = p.parse_args()

    datos = cargar()
    tabla_ii(datos, latex=args.latex)
    tabla_iii(datos, latex=args.latex)
    if not args.latex:
        detalle(datos)


if __name__ == "__main__":
    main()
