"""
Figura de entrenabilidad en cuatro paneles.

Compara el paisaje de optimización de Qudit-ADAPT sobre dos instancias —el
grafo de la Fig. 1a del paper y el grafo completo K6— con los dos pools
contradiabáticos (l=1 y l=2).

Cada panel muestra, para cada tamaño de ansatz, el óptimo final de 100
instancias con parámetros iniciales aleatorios (las marcas horizontales),
contra la curva de ADAPT con parámetros reciclados (verde) y la de reinicio
frío en theta=0 (rojo punteada). La métrica es el error relativo del paper,
eps_rel = |E - E0| / |E0|.

Funciona con corridas parciales: los checkpoints se escriben en cada
iteración, así que la figura se puede generar antes de que terminen.

    python cluster/figura_4paneles.py
    python cluster/figura_4paneles.py --salida mi_figura
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
import numpy as np

from funciones.utilidades_bp import (
    bp_error_curves,
    load_bp_result,
    plot_bp_landscape,
)

JSON_DIR = PROJECT_ROOT / "resultados" / "json"
IMG_DIR = PROJECT_ROOT / "resultados" / "images"

PANELES = [
    ("fig1a_l1_bp100.json", r"Grafo Fig. 1a (10 aristas), $\ell=1$"),
    ("fig1a_l2_bp100.json", r"Grafo Fig. 1a (10 aristas), $\ell=2$"),
    ("k6_l1_bp100.json", r"Grafo completo $K_6$ (15 aristas), $\ell=1$"),
    ("k6_l2_bp100.json", r"Grafo completo $K_6$ (15 aristas), $\ell=2$"),
]


def estilo():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "serif",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "savefig.dpi": 200,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", type=str, default="bp_4paneles")
    p.add_argument("--cmap", type=str, default="turbo")
    args = p.parse_args()

    estilo()

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))
    resumen = []
    faltan = []

    for ax, (archivo, titulo) in zip(axes.ravel(), PANELES):
        ruta = JSON_DIR / archivo
        if not ruta.exists():
            ax.text(0.5, 0.5, f"falta\n{archivo}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
            faltan.append(archivo)
            continue

        res = load_bp_result(ruta)
        plot_bp_landscape(res, ax=ax, cmap=args.cmap, mode="final",
                          inset_grafo=False, relativo=True)

        k = int(res["num_ansatz_ops"])
        parcial = res["stop_reason"] == "en_progreso"
        ax.set_title(f"{titulo}\n{k}/{res['max_iteration']} parámetros"
                     + ("  (parcial)" if parcial else ""))

        curvas = bp_error_curves(res, relativo=True)
        E0 = float(res["ground_energy"])
        finales = np.abs(
            np.array([x["final_energy"] for x in res["random_runs"][-1]], float) - E0
        ) / abs(E0)

        resumen.append({
            "panel": titulo, "k": k, "pool": res["pool_size"],
            "adapt": curvas["recycled"][-1], "fria": curvas["cold"][-1],
            "mejor": float(finales.min()), "mediana": float(np.median(finales)),
            "peor": float(finales.max()),
            "unicos": int(len(np.unique(np.round(finales, 10)))),
            "parcial": parcial,
        })

    fig.suptitle(
        "Entrenabilidad de Qudit-ADAPT: parámetros reciclados frente a 100 "
        "instancias con inicialización aleatoria",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(IMG_DIR / f"{args.salida}.{ext}")

    print(f"Figura: {IMG_DIR / (args.salida + '.pdf')}")
    if faltan:
        print(f"Paneles sin datos todavía: {faltan}")

    print()
    cab = f'{"panel":42s}{"k":>4s}{"pool":>7s}{"ADAPT":>11s}{"fría":>11s}{"mejor":>11s}{"mediana":>11s}{"peor":>11s}{"únicos":>8s}'
    print(cab); print("-" * len(cab))
    for r in resumen:
        marca = " *" if r["parcial"] else ""
        print(f'{r["panel"][:40]:42s}{r["k"]:>4d}{r["pool"]:>7d}{r["adapt"]:>11.3e}'
              f'{r["fria"]:>11.3e}{r["mejor"]:>11.3e}{r["mediana"]:>11.3e}'
              f'{r["peor"]:>11.3e}{r["unicos"]:>8d}{marca}')
    if any(r["parcial"] for r in resumen):
        print("\n(*) corrida aún en progreso")


if __name__ == "__main__":
    main()
