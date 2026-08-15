"""
Trainability figures in the style of the Qudit-ADAPT manuscript.

Produces one figure per graph instance, with two stacked panels a) and b)
corresponding to the l = 1 and l = 2 truncations of the approximate AGP. Each
panel shows the relative error against the number of variational parameters
for the warm-start strategy that Qudit-ADAPT actually uses, for a cold restart
at theta = 0, and for 100 independent random parameter initialisations.

Formatting follows the manuscript: panel labels outside the axes, the graph
instance as an inset, English axis labels and a Computer Modern math font.

    python cluster/figuras_paper.py
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
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from funciones.utilidades_bp import (
    bp_error_cloud,
    bp_error_curves,
    layout_poligono,
    load_bp_result,
)

JSON_DIR = PROJECT_ROOT / "resultados" / "json"
IMG_DIR = PROJECT_ROOT / "resultados" / "images"

# Legend labels requested for the manuscript.
_BM = r"\bm" if False else r"\boldsymbol"   # \bm sólo existe con usetex
LBL_WARM = r"Warm start: $\boldsymbol{\theta}_k = (\boldsymbol{\theta}_{k-1}, 0)$"
LBL_COLD = r"Cold start: $\boldsymbol{\theta}_k = \boldsymbol{0}_k$"
LBL_RAND = r"Random restarts ($100$ per $k$)"

# Fracción de la altura del panel reservada abajo para el grafo y la leyenda.
FRAC_BANDA = 0.24

# Ancho de columna de REVTeX a dos columnas (246 pt). Las figuras se diseñan
# EXACTAMENTE a esta medida para que LaTeX las coloque sin reescalar: si se
# diseñan más anchas y luego se reducen a \linewidth, toda la tipografía se
# encoge en la misma proporción y deja de coincidir con la del texto.
ANCHO_COLUMNA = 3.40

# Proporción alto/ancho. Se fija en 0.88 a propósito: en `overpic` las
# coordenadas van en centésimas del ANCHO, de modo que el \put(2,88) que el
# manuscrito ya usa para las etiquetas a)/b) cae justo en el borde superior
# sólo si la figura tiene esta proporción. Así no hay que tocar el .tex.
PROPORCION = 0.88

# Un archivo por panel: el manuscrito compone las figuras con `overpic` y
# coloca las etiquetas a) / b) desde LaTeX, así que NO deben venir en el PDF.
# Se usan las MISMAS imágenes de grafo que el manuscrito, ya renderizadas en
# datos/imagenes/, en vez de volver a dibujarlas con networkx: así los insets
# son idénticos a los de las figuras 1-3.
IMG_GRAFOS = PROJECT_ROOT / "datos" / "imagenes"

PANELES = [
    ("fig_bp_1a_l1", "fig1a_l1_bp100.json", "comparaciones/grafo_1.png"),
    ("fig_bp_1a_l2", "fig1a_l2_bp100.json", "comparaciones/grafo_1.png"),
    ("fig_bp_k6_l1", "k6_l1_bp100.json", "regulares/grado_5.png"),
    ("fig_bp_k6_l2", "k6_l2_bp100.json", "regulares/grado_5.png"),
]


# Tamaños del script de figuras del manuscrito, para que al colocarse a
# \linewidth el resultado coincida con las figuras 1-4.
TAM_ETIQUETA = 20
TAM_LEYENDA = 13
TAM_TICK_MAYOR = 16
TAM_TICK_MENOR = 14

# El manuscrito usa text.usetex con txfonts. matplotlib exige además el paquete
# cm-super de LaTeX, que no está instalado en esta máquina y necesita permisos
# de root (`sudo apt install cm-super`). Con USETEX=False se emplea STIX, que
# es la aproximación más cercana: txfonts y STIX son ambas de base Times.
USETEX = True


def estilo():
    """Réplica de la configuración de figuras del manuscrito."""
    mpl.rcdefaults()
    comun = {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 15,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
    if USETEX:
        comun.update({
            "text.usetex": True,
            "text.latex.preamble": r"""
                \usepackage{amsmath}
                \usepackage{amsfonts}
                \usepackage{amssymb}
                \usepackage{bm}
                \usepackage{txfonts}
            """,
        })
    else:
        comun.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.serif": ["STIXGeneral"],
        })
    mpl.rcParams.update(comun)


def _transparentar(img, umbral=0.93):
    """
    Vuelve transparente el fondo blanco de la imagen del grafo.

    Las imágenes de datos/imagenes/ no tienen canal alfa y traen fondo blanco,
    que dentro del panel se ve como un recuadro. Se calcula el alfa con una
    rampa suave sobre el mínimo de los canales RGB —blanco puro da alfa 0 y
    cualquier píxel con algún canal por debajo del umbral da alfa 1— de modo
    que los bordes suavizados de nodos y aristas se conservan en vez de
    quedar dentados.
    """
    rgb = img[..., :3].astype(float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    minimo = rgb.min(axis=-1)
    alfa = np.clip((1.0 - minimo) / (1.0 - umbral), 0.0, 1.0)
    return np.dstack([rgb, alfa])


def agregar_imagen_dentro(ax, ruta, zoom=0.10, xy=(0.155, 0.40), transparente=True):
    """
    Inserta la imagen del grafo dentro del eje, igual que el script de figuras
    del manuscrito: (0,0) es la esquina inferior izquierda y (1,1) la superior
    derecha en fracción de eje.
    """
    img = mpimg.imread(ruta)
    if transparente:
        img = _transparentar(img)
    imagen = OffsetImage(img, zoom=zoom)
    ax.add_artist(AnnotationBbox(
        imagen, xy, xycoords="axes fraction", frameon=False,
        box_alignment=(0.5, 0.5), zorder=3,
    ))


def panel(ax, res, cmap="turbo", floor=1e-16, imagen_grafo=None, zoom=0.10):
    E0 = float(res["ground_energy"])

    ks, errs = bp_error_cloud(res, floor=floor, mode="final", relativo=True)
    curvas = bp_error_curves(res, floor=floor, relativo=True)

    # cloud of converged optima from the random initialisations
    if len(errs):
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.16, 0.16, size=len(ks))
        ax.scatter(ks + jitter, errs, c=errs, cmap=cmap,
                   norm=LogNorm(vmin=max(errs.min(), floor), vmax=errs.max()),
                   marker="_", s=95, linewidths=1.3, alpha=0.55, zorder=2,
                   rasterized=True, label=LBL_RAND)

    k = curvas["k"]
    ax.plot(k, curvas["cold"], color="red", lw=1.8, ls="--",
            marker="o", ms=7, markevery=2, zorder=4, label=LBL_COLD)
    ax.plot(k, curvas["recycled"], color="blue", lw=1.8, ls="--",
            marker="x", ms=8, mew=1.8, markevery=1, zorder=5, label=LBL_WARM)

    ax.set_yscale("log")
    ax.set_xlabel("Number of variational parameters", fontsize=TAM_ETIQUETA)
    ax.set_ylabel(r"$\epsilon_{\mathrm{rel}}$", fontsize=TAM_ETIQUETA + 5)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    ax.tick_params(axis="both", which="major", labelsize=TAM_TICK_MAYOR)
    ax.tick_params(axis="both", which="minor", labelsize=TAM_TICK_MENOR)
    ax.grid(True, which="both", alpha=0.1)
    ax.set_xlim(-1.0, float(k[-1]) + 1.0)

    # El eje x cuenta parámetros, que son enteros. Sin esto el localizador
    # automático elige pasos de 2.5 cuando el ansatz es corto —como en K6 con
    # l = 2, que converge en 18— y rotula 15.0, 17.5, etc.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Se abre una banda libre en la parte inferior para alojar el grafo y la
    # leyenda sin taparle nada a las curvas. En escala logarítmica basta con
    # bajar el límite inferior en la fracción de décadas correspondiente.
    todos = np.concatenate([errs, curvas["recycled"], curvas["cold"]]) if len(errs) \
        else np.concatenate([curvas["recycled"], curvas["cold"]])
    lo, hi = float(np.min(todos)), float(np.max(todos))
    decadas = np.log10(hi / lo)
    # Con topes: sobre un rango de muchas décadas la fracción sola abriría un
    # hueco enorme (en K6, l=2 el rango es de ~16 décadas), y ahí la banda
    # inferior ya está libre de por sí.
    extra = min(max(FRAC_BANDA / (1.0 - FRAC_BANDA) * decadas, 0.55), 1.30)
    ax.set_ylim(lo / 10 ** extra, hi * 10 ** (0.06 * decadas))

    if imagen_grafo is not None:
        agregar_imagen_dentro(ax, imagen_grafo, zoom=zoom)

    # La leyenda va FUERA del área de datos, debajo de los ejes.
    #
    # Con las etiquetas completas la caja mide ~55% del ancho del panel, y la
    # única región libre dentro es un triángulo en la esquina inferior
    # izquierda: las curvas bajan en diagonal y la nube de mínimos ocupa la
    # mitad superior derecha. Ahí no cabe. A la izquierda pisa el grafo y a la
    # derecha pisa la cola de las curvas, así que se coloca fuera y el inset
    # se queda solo con la esquina.
    handles, labels = ax.get_legend_handles_labels()
    orden = [labels.index(x) for x in (LBL_WARM, LBL_COLD, LBL_RAND) if x in labels]
    ax.legend([handles[i] for i in orden], [labels[i] for i in orden],
              loc="lower left", fontsize=TAM_LEYENDA, frameon=True,
              framealpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmap", type=str, default="turbo")
    args = p.parse_args()

    estilo()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for salida, archivo, img in PANELES:
        ruta = JSON_DIR / archivo
        if not ruta.exists():
            print(f"[saltado] {salida}: falta {archivo}")
            continue

        res = load_bp_result(ruta)
        fig, ax = plt.subplots(figsize=(8, 7))
        panel(ax, res, cmap=args.cmap, imagen_grafo=IMG_GRAFOS / img)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(IMG_DIR / f"{salida}.{ext}")
        plt.close(fig)

        c = bp_error_curves(res, relativo=True)
        fin = np.abs(np.array([x["final_energy"] for x in res["random_runs"][-1]], float)
                     - res["ground_energy"]) / abs(res["ground_energy"])
        print(f"{salida}.pdf   k={res['num_ansatz_ops']:2d}  pool={res['pool_size']:5d}  "
              f"warm={c['recycled'][-1]:.3e}  cold={c['cold'][-1]:.3e}  "
              f"median={np.median(fin):.3e}  unique={len(np.unique(np.round(fin,10)))}")


if __name__ == "__main__":
    main()
