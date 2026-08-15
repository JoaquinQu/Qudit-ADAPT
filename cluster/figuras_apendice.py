"""
Appendix figures: Gell-Mann pool reformulation and native-gate benchmarking.

Both figures span the full text width (`figure*`), so they are drawn at
textwidth rather than column width, with type sizes chosen to match the body
font once placed at 100 %.

Everything is reported in the relative error used throughout the manuscript,
    eps_rel = |E - E0| / |E0|,
which is the metric of Eq. (32); the earlier versions of these figures used the
absolute error and are not directly comparable with the rest of the paper.

    python cluster/figuras_apendice.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from funciones.utilidades_bp import (
    bp_error_curves,
    conteo_compuertas,
    conteo_compuertas_ansatz,
    costo_compuertas,
    costo_mediciones,
    load_bp_result,
)
from cluster.conteo_qaoa import G2, conteo_qaoa

JSON_DIR = PROJECT_ROOT / "resultados" / "json"
IMG_DIR = PROJECT_ROOT / "resultados" / "images"

# Full text width of a two-column REVTeX page.
ANCHO_TEXTO = 7.00

POOLS = {
    "Angular momentum": "bp_n6_l2_grafo2_r20_k30.json",
    "Gell-Mann": "bp_n6_l2_grafo2_r20_k30_gellmann.json",
}
COLORES = {"Angular momentum": "blue", "Gell-Mann": "red"}
MARCAS = {"Angular momentum": "x", "Gell-Mann": "o"}

UMBRAL = 1e-3          # target relative precision used in the appendix tables

# QAOA reference at a matched parameter budget. 16 is the number of parameters
# at which the angular pool reaches the target precision, i.e. the SMALLEST of
# the two budgets in the appendix tables, so the comparison is the conservative
# one. No QAOA optimisation is involved: the gate count of a QAOA circuit is
# fixed by the graph and the number of layers (see cluster/conteo_qaoa.py).
#
# La comparación con QAOA vive en la Tabla IV del manuscrito, así que la figura
# va sin ella: con la recta dentro hay que pasar el eje x a escala logarítmica
# —QAOA queda entre cinco y nueve veces más a la derecha— y eso aplasta las dos
# curvas de ADAPT, que son lo que la figura tiene que mostrar.
CON_QAOA = False
QAOA_PARAMS = 16
COLOR_QAOA = "0.35"


# Se reutiliza exactamente la configuración del manuscrito.
from cluster.figuras_paper import estilo, TAM_ETIQUETA, TAM_LEYENDA, TAM_TICK_MAYOR, TAM_TICK_MENOR  # noqa: E402


def cargar():
    datos = {}
    for etiqueta, archivo in POOLS.items():
        ruta = JSON_DIR / archivo
        if not ruta.exists():
            raise FileNotFoundError(ruta)
        res = load_bp_result(ruta)
        E0 = float(res["ground_energy"])
        datos[etiqueta] = {
            "res": res,
            "eps": np.abs(np.asarray(res["recycled_energy"], float) - E0) / abs(E0),
            "gates": costo_compuertas(res),
            "meas": costo_mediciones(res),
            "nativo": conteo_compuertas_ansatz(res),
        }
    return datos


def _ejes(ax, xlabel):
    """Ejes con el formato del manuscrito."""
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=TAM_ETIQUETA)
    ax.set_ylabel(r"$\epsilon_{\mathrm{rel}}$", fontsize=TAM_ETIQUETA + 5)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    ax.tick_params(axis="both", which="major", labelsize=TAM_TICK_MAYOR)
    ax.tick_params(axis="both", which="minor", labelsize=TAM_TICK_MENOR)
    ax.grid(True, which="both", alpha=0.1)
    ax.legend(loc="lower left", fontsize=TAM_LEYENDA, frameon=True,
              framealpha=0.9)


def _traza(ax, x, y, etiqueta, umbral=None):
    m = min(len(x), len(y))
    ax.plot(x[:m], y[:m], color=COLORES[etiqueta], lw=1.8, ls="--",
            marker=MARCAS[etiqueta], ms=8, mew=1.8, label=etiqueta, zorder=4)
    if umbral is not None:
        bajo = np.where(y[:m] < umbral)[0]
        if len(bajo):
            k = int(bajo[0])
            ax.plot([x[k]], [y[k]], marker="*", ms=26, color=COLORES[etiqueta],
                    markeredgecolor="white", markeredgewidth=1.4, zorder=6)


def figura_convergencia(datos, salida="fig_ap_pools"):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    ejes = [
        ("Number of variational parameters", lambda d: np.arange(len(d["eps"]))),
        ("Accumulated two-qutrit entangling gates", lambda d: d["gates"]),
        ("Accumulated gradient measurements", lambda d: d["meas"]),
    ]

    for ax, (xlabel, getx) in zip(axes, ejes):
        for etiqueta, d in datos.items():
            _traza(ax, getx(d), d["eps"], etiqueta)
        _ejes(ax, xlabel)

    axes[2].set_xscale("log")

    fig.tight_layout(w_pad=3.0)
    for ext in ("pdf", "png"):
        fig.savefig(IMG_DIR / f"{salida}.{ext}")
    plt.close(fig)
    return salida


def figura_compuertas(datos, salida="fig_ap_native"):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    claves = [
        ("r_dos_niveles", r"Single-qudit rotations $R^{(i,j)}$"),
        ("ms", r"M\o{}lmer-S\o{}rensen gates"),
        ("total", "Total native gates"),
    ]
    claves[1] = ("ms", "Mølmer-Sørensen gates")

    qaoa = conteo_qaoa(6, G2, QAOA_PARAMS, base="gellmann") if CON_QAOA else None

    for ax, (clave, xlabel) in zip(axes, claves):
        for etiqueta, d in datos.items():
            _traza(ax, d["nativo"][clave], d["eps"], etiqueta, umbral=UMBRAL)
        ax.axhline(UMBRAL, color="0.55", ls=":", lw=1.2, zorder=1)

        if CON_QAOA:
            # Recta de referencia de QAOA con el mismo presupuesto de
            # parámetros. Va como vertical y no como punto porque no se
            # optimizó nada: el circuito tiene un costo definido pero no una
            # energía asociada. Obliga a escala logarítmica en x.
            ax.axvline(qaoa[clave], color=COLOR_QAOA, ls="-.", lw=2.0, zorder=3,
                       label=rf"QAOA, {QAOA_PARAMS} parameters ({qaoa[clave]})")
            ax.set_xscale("log")
            ax.set_xlim(left=3.0, right=qaoa[clave] * 1.6)

        _ejes(ax, xlabel + " (accumulated)")

    fig.tight_layout(w_pad=3.0)
    for ext in ("pdf", "png"):
        fig.savefig(IMG_DIR / f"{salida}.{ext}")
    plt.close(fig)
    return salida


def tablas(datos):
    """Numbers for the appendix tables, all in relative error."""
    print(f"\n{'':18s}{'pool':>7s}{'k':>4s}{'eps_rel final':>15s}"
          f"{'gates':>7s}{'meas':>8s}")
    for etiqueta, d in datos.items():
        r = d["res"]
        print(f"{etiqueta:18s}{r['pool_size']:>7d}{r['num_ansatz_ops']:>4d}"
              f"{d['eps'][-1]:>15.3e}{int(d['gates'][-1]):>7d}{int(d['meas'][-1]):>8d}"
              f"   ({r['stop_reason']})")

    print(f"\nAt eps_rel <= {UMBRAL:g}:")
    print(f"{'':18s}{'k':>4s}{'R':>6s}{'MS':>5s}{'total':>7s}{'meas':>8s}")
    for etiqueta, d in datos.items():
        bajo = np.where(d["eps"] < UMBRAL)[0]
        if not len(bajo):
            print(f"{etiqueta:18s}  never reaches the threshold")
            continue
        k = int(bajo[0])
        n = d["nativo"]
        print(f"{etiqueta:18s}{k:>4d}{int(n['r_dos_niveles'][k]):>6d}"
              f"{int(n['ms'][k]):>5d}{int(n['total'][k]):>7d}"
              f"{int(d['meas'][k]):>8d}")

    print("\nTwo-level rotations per generator:")
    for etiqueta, d in datos.items():
        r = d["res"]
        base = r.get("base", "angular")
        c = [conteo_compuertas(l, base=base)["r_dos_niveles"]
             for l in r["ansatz_op_labels"]]
        nop = sum(0 if conteo_compuertas(l, base=base)["es_producto"] else 1
                  for l in r["ansatz_op_labels"])
        print(f"  {etiqueta:18s} mean {np.mean(c):.2f}   "
              f"generators needing Trotterization: {nop}/{len(c)}")


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    estilo()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    datos = cargar()
    print(figura_convergencia(datos) + ".pdf")
    print(figura_compuertas(datos) + ".pdf")
    tablas(datos)


if __name__ == "__main__":
    main()
